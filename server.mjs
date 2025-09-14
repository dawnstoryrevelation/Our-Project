// server.mjs
import 'dotenv/config';
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import { OpenAI } from 'openai';

const app = express();
const PORT = process.env.PORT || 8787;
const ORIGIN = process.env.ORIGIN || 'http://localhost:8080';
const MODEL  = process.env.OPENAI_MODEL || 'gpt-5-mini';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.use(helmet({
  contentSecurityPolicy: false, // keep simple; configure CSP on your host if needed
  crossOriginEmbedderPolicy: false
}));
app.use(cors({ origin: ORIGIN, credentials: false }));
app.use(express.json({ limit: '1mb' }));
app.use(rateLimit({ windowMs: 60_000, max: 60 }));

// ----------- MCTS PLANNER (compact & domain aware) ---------------------------
// State = { condition, step, flags }
// Actions depend on condition; transitions set flags and accrue reward.
// Reward heuristic balances: clinical safety, time-to-care, accessibility uplift.
const ACTIONS = [
  { id:'urgent_triage',      label:'Urgent triage at ER/ophthalmology',       cost: 5, benefit: 18, riskDrop: 0.9 },
  { id:'book_specialist',    label:'Book ophthalmologist within 1–2 weeks',    cost: 2, benefit: 9,  riskDrop: 0.5 },
  { id:'lab_tests',          label:'Lab tests & OCT/fundus imaging',           cost: 2, benefit: 6,  riskDrop: 0.3 },
  { id:'meds_dr',            label:'Discuss anti-VEGF/steroids (DR/DME)',      cost: 3, benefit: 10, riskDrop: 0.5 },
  { id:'glycemic_bp',        label:'Tight glycemic & BP control plan',         cost: 3, benefit: 8,  riskDrop: 0.35 },
  { id:'low_vision_tools',   label:'Enable low-vision tools & contrast modes', cost: 1, benefit: 5,  accessBoost: 0.5 },
  { id:'screen_reader',      label:'Set up screen reader + TTS everywhere',    cost: 1, benefit: 6,  accessBoost: 0.7 },
  { id:'wearables',          label:'Voice/haptic wearables for navigation',    cost: 2, benefit: 6,  accessBoost: 0.6 },
  { id:'followup_6w',        label:'Schedule follow-up in 6–8 weeks',          cost: 1, benefit: 4,  riskDrop: 0.2 }
];

function possibleActions(condition){
  const c = (condition||'').toLowerCase();
  // DR/DME bias clinical actions higher; otherwise accessibility-first
  if (c.includes('dr') || c.includes('retinopathy') || c.includes('dme')) {
    return ['urgent_triage','book_specialist','lab_tests','meds_dr','glycemic_bp','screen_reader','low_vision_tools','followup_6w','wearables'];
  }
  return ['book_specialist','lab_tests','screen_reader','low_vision_tools','wearables','followup_6w'];
}

function rolloutReward(seq, baseRisk=0.6){
  // Monte Carlo heuristic: risk multiplies down by (1-riskDrop), accessibility sums
  let risk = baseRisk, access=0;
  let utility=0, cost=0;
  for(const a of seq){
    risk *= (1 - (a.riskDrop||0));
    access += (a.accessBoost||0);
    cost   += a.cost||0;
    utility += (a.benefit||0);
  }
  // Final score: lower risk + higher access + utility − cost
  const score = (1 - Math.min(1, risk)) * 0.55 + Math.min(1, access/2) * 0.25 + (utility/40)*0.2 - (cost/20)*0.1;
  return score;
}

function mctsPlan(condition, iters=200, maxDepth=5){
  const acts = possibleActions(condition).map(id => ACTIONS.find(a=>a.id===id));
  let bestSeq = [], bestScore = -1;
  for (let i=0; i<iters; i++){
    const seq = [];
    const depth = 2 + Math.floor(Math.random()*maxDepth);
    // random playout with bias toward earlier clinical steps
    const bag = [...acts];
    for (let d=0; d<depth && bag.length; d++){
      const pick = (Math.random()<0.6) ? bag.shift() : bag.splice(Math.floor(Math.random()*bag.length),1)[0];
      if (pick) seq.push(pick);
    }
    const score = rolloutReward(seq, /*baseRisk*/ (condition?.toLowerCase().includes('dme')||condition?.toLowerCase().includes('dr'))?0.7:0.4);
    if (score > bestScore){ bestScore = score; bestSeq = seq; }
  }
  const summary = bestSeq.map(a=>a.label).join(' → ');
  return {
    title: `Optimized next steps for ${condition || 'your eye health'}`,
    summary,
    steps: bestSeq.map(a=>({ id:a.id, label:a.label })),
    score: Number(bestScore.toFixed(3))
  };
}

// --------------------- OpenAI: safe, accessible assistant --------------------
const SYS = `
You are WhalaBot, an accessibility-first health advisor. Be supportive, plain, and safe.
CRITICAL:
- You are NOT a medical device. Include disclaimers and urge professional care for concerning symptoms.
- Use the provided condition (from CNN or user) and the MCTS plan to craft a stepwise, long-horizon strategy.
- Provide alternatives for visually-impaired users: TTS-first instructions, screen-reader / high-contrast steps, haptic/voice tools.
- Prefer concise bullets, checklists, timelines, and cue words for audio navigation.
- If the user typed "Normal"/"Unknown", emphasize routine screening and accessibility setup.
- Avoid definitive diagnoses; use likelihood language; never prescribe meds directly—recommend consulting clinicians.
- Include a brief "Next 48 hours" and "6–8 weeks" plan, plus resources.
`;

async function aiRespond({condition, cnn, message, plan}){
  const context = `
Condition: ${condition || 'Unknown'}
CNN: ${cnn ? JSON.stringify(cnn) : '{}'}
MCTS Plan: ${plan ? JSON.stringify(plan) : '{}'}
User message: ${message || '(none)'}
`;
  const res = await client.chat.completions.create({
    model: MODEL,
    temperature: 0.4,
    messages: [
      { role:'system', content: SYS },
      { role:'user', content: context }
    ]
  });
  const reply = res.choices?.[0]?.message?.content?.trim() || "I'm here to help.";
  return reply;
}

// --------------------------- Routes ------------------------------------------
app.get('/api/health', (req,res)=> res.json({ ok:true, model:MODEL, time:new Date().toISOString() }));

app.post('/api/plan_and_chat', async (req,res)=>{
  try{
    const { condition, cnn, user } = req.body || {};
    const plan = mctsPlan(condition);
    const intro = await aiRespond({ condition, cnn, message: "Provide a brief welcome + summary of next steps for TTS.", plan });
    const chart = { classProbs: cnn?.probs || [0.5,0.5] };
    res.json({ plan, intro, chart });
  }catch(e){
    console.error(e);
    res.status(500).json({ error:'plan_and_chat_failed' });
  }
});

app.post('/api/chat', async (req,res)=>{
  try{
    const { condition, cnn, message } = req.body || {};
    const plan = mctsPlan(condition);
    const reply = await aiRespond({ condition, cnn, message, plan });
    res.json({ reply, plan });
  }catch(e){
    console.error(e);
    res.status(500).json({ error:'chat_failed' });
  }
});

app.listen(PORT, ()=> console.log(`Whala server running on http://localhost:${PORT}`));
