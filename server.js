// server.js
// Production-ready Express server for WhalaApp (WhalaBot + B-Eye-O-Marker V1)
// Usage:
//   1) npm init -y
//   2) npm i express dotenv openai helmet cors express-rate-limit morgan
//   3) echo "OPENAI_API_KEY=sk-...." > .env
//   4) node server.js
//
// Files expected next to this script (no folders):
//   - index.html           (your landing page)
//   - application.html     (the app below)
//   - b_eye_o_marker_v1.onnx  (CNN model file)
//   - logo.png (optional)

import 'dotenv/config';
import express from 'express';
import path from 'path';
import helmet from 'helmet';
import cors from 'cors';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import { fileURLToPath } from 'url';
import { OpenAI } from 'openai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// --- Security hardening
app.use(helmet({
  crossOriginEmbedderPolicy: false, // allow WASM
  contentSecurityPolicy: {
    useDefaults: true,
    directives: {
      "default-src": ["'self'"],
      "script-src": [
        "'self'",
        "'unsafe-inline'", // needed for on-the-fly UI (kept minimal)
        "https://cdn.jsdelivr.net",
        "https://unpkg.com"
      ],
      "connect-src": [
        "'self'",
        "https://api.openai.com",
        "https://cdn.jsdelivr.net",
        "https://unpkg.com"
      ],
      "img-src": ["'self'", "data:", "blob:"],
      "style-src": ["'self'", "'unsafe-inline'"],
      "font-src": ["'self'", "data:", "https://cdn.jsdelivr.net"],
      "worker-src": ["'self'", "blob:"],
      "frame-ancestors": ["'self'"]
    }
  }
}));
app.use(cors());
app.use(morgan('combined'));
app.use(express.json({ limit: '2mb' }));

// --- Rate limit API
const limiter = rateLimit({ windowMs: 60_000, max: 60 });
app.use('/api/', limiter);

// --- Serve the two html files and assets from the flat repo
const sendFile = (res, name) => res.sendFile(path.join(__dirname, name));
app.get('/', (_, res) => sendFile(res, 'index.html'));
app.get('/application.html', (_, res) => sendFile(res, 'application.html'));
app.get('/b_eye_o_marker_v1.onnx', (_, res) => sendFile(res, 'b_eye_o_marker_v1.onnx'));
app.get('/logo.png', (_, res) => sendFile(res, 'logo.png')); // optional

// --- OpenAI setup (server-side; never expose key client-side)
const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error('Missing OPENAI_API_KEY in .env');
}
const openai = new OpenAI({ apiKey });

// --- Safety / compliance note:
// This system is research/assistive. It does NOT give medical diagnoses.
// Always recommend professional care and emergency escalation where appropriate.

// Monte Carlo Tree Search primitive (domain: ocular health & accessibility).
// We use a lightweight MCTS to propose candidate action plans, then pass the
// top plan to GPT-5 Mini for reasoning & report generation. This keeps "MCTS present"
// and auditable, while the LLM handles the prose and personalization.
class MCTSNode {
  constructor(state, parent = null, action = null) {
    this.state = state;        // { condition, severity, prefs, stepIndex, plan[] }
    this.parent = parent;
    this.action = action;
    this.children = [];
    this.visits = 0;
    this.value = 0;
  }
}

function uct(child, totalVisits, c = 1.4) {
  if (child.visits === 0) return Infinity;
  return (child.value / child.visits) + c * Math.sqrt(Math.log(totalVisits) / child.visits);
}

// Domain-specific action space
const ACTIONS = [
  { key: 'book_ophthalmology', cost: 2, benefit: 9, text: 'Book an ophthalmology appointment (within 1–2 weeks)' },
  { key: 'urgent_referral',   cost: 3, benefit: 12, text: 'Seek urgent care if symptoms acute (flashes, curtain, severe pain)' },
  { key: 'oct_scan_followup', cost: 2, benefit: 8, text: 'Schedule OCT follow-up imaging (6–8 weeks)' },
  { key: 'glycemic_control',  cost: 1, benefit: 7, text: 'Tighten glycemic control & BP (log with reminders)' },
  { key: 'vitreous_floater_tx', cost: 2, benefit: 5, text: 'Discuss floaters/vision changes triage' },
  { key: 'a11y_screenreader', cost: 1, benefit: 6, text: 'Enable screen reader + larger text + contrast' },
  { key: 'tts_everywhere',    cost: 1, benefit: 6, text: 'Use TTS on all outputs and device a11y shortcuts' },
  { key: 'low_vision_tools',  cost: 2, benefit: 7, text: 'Trial magnifier, high-contrast themes, voice control' },
  { key: 'diet_plan',         cost: 2, benefit: 6, text: 'Adopt low-GI diet with hydration reminders' },
  { key: 'sleep_schedule',    cost: 1, benefit: 4, text: 'Regular sleep and screen-time hygiene' }
];

// Heuristic reward based on condition
function rewardFor(state, action) {
  const { condition = 'unknown', severity = 'moderate' } = state;
  const w = (k) => ({
    dr: { book_ophthalmology: 10, oct_scan_followup: 8, glycemic_control: 7 },
    dme: { book_ophthalmology: 10, oct_scan_followup: 8, glycemic_control: 7 },
    normal: { a11y_screenreader: 6, tts_everywhere: 6, low_vision_tools: 6 }
  }[condition]?.[k] ?? 5);
  const sev = severity === 'severe' ? 1.4 : severity === 'mild' ? 0.9 : 1.0;
  return (action.benefit - action.cost) * sev + w(action.key);
}

function expand(node) {
  const used = new Set(node.state.plan?.map(p => p.key));
  const options = ACTIONS.filter(a => !used.has(a.key));
  if (options.length === 0) return null;
  const pick = options[Math.floor(Math.random() * options.length)];
  const next = new MCTSNode(
    {
      ...node.state,
      stepIndex: node.state.stepIndex + 1,
      plan: [...(node.state.plan || []), pick]
    },
    node,
    pick
  );
  node.children.push(next);
  return next;
}

function simulate(state) {
  let total = 0;
  const depth = 4;
  let used = new Set(state.plan?.map(p => p.key));
  for (let i = 0; i < depth; i++) {
    const options = ACTIONS.filter(a => !used.has(a.key));
    if (options.length === 0) break;
    const a = options[Math.floor(Math.random() * options.length)];
    total += rewardFor(state, a);
    used.add(a.key);
  }
  return total / Math.max(1, depth);
}

function backpropagate(node, value) {
  while (node) {
    node.visits += 1;
    node.value += value;
    node = node.parent;
  }
}

function bestChildByUCT(node) {
  let best = null, bestUct = -Infinity;
  for (const child of node.children) {
    const u = uct(child, Math.max(1, node.visits));
    if (u > bestUct) { bestUct = u; best = child; }
  }
  return best;
}

function runMCTS({ condition, severity, prefs }, iters = 200) {
  const root = new MCTSNode({ condition, severity, prefs, stepIndex: 0, plan: [] });
  for (let i = 0; i < iters; i++) {
    // Selection
    let node = root;
    while (node.children.length > 0) node = bestChildByUCT(node);
    // Expansion
    node = expand(node) || node;
    // Simulation
    const v = simulate(node.state);
    // Backprop
    backpropagate(node, v);
  }
  // Pick best plan from root’s children by mean value
  let best = null, score = -Infinity;
  for (const child of root.children) {
    const s = child.value / Math.max(1, child.visits);
    if (s > score) { score = s; best = child; }
  }
  return (best?.state.plan || []).map(a => a.text);
}

// --- Chat API: consumes user messages + CNN context, returns assistant text
app.post('/api/chat', async (req, res) => {
  try {
    if (!apiKey) return res.status(500).json({ error: 'Server missing API key' });

    const { messages = [], cnn = null, manual_condition = null, user_prefs = {} } = req.body || {};
    const condition = (manual_condition?.toLowerCase?.() || cnn?.label || 'unknown')
      .replace(/[^a-z]/g, '');
    const severity = cnn?.severity || 'moderate';
    const plan = runMCTS({ condition, severity, prefs: user_prefs }, 240);

    // Compose a strong system prompt (safety + accessibility)
    const system = [
      "Role: Assistive health & accessibility guide. You are not a doctor and do NOT provide medical diagnoses.",
      "Always caution users to seek professional medical care; escalate emergencies.",
      "Use the provided CNN context conservatively; it may be wrong. Offer uncertainty handling and alternatives.",
      "For accessibility, provide: concise summary first; numbered action steps; long-form details; and a TL;DR.",
      "Include accessible descriptions of any charts/diagrams you suggest.",
      "Tone: calm, clear, empowering. Avoid jargon unless explained.",
      "Respect privacy. No data retention. No disallowed content.",
      "If user is visually impaired, adapt instructions for audio/screen readers.",
      "Propose concrete follow-ups with timelines (e.g., 1–2 weeks, 6–8 weeks).",
      "You can suggest simple pseudo-graphs (ASCII) when useful."
    ].join(' ');

    const context = {
      cnn_summary: cnn || null,
      assumed_condition: condition,
      mcts_candidate_plan: plan
    };

    // Use GPT-5 Mini (fast, cheap) — see OpenAI docs for GPT-5 family and minis.
    // Node SDK "responses" API (preferred in latest docs) or fallback to chat.completions.
    const response = await openai.responses.create({
      model: 'gpt-5-mini',
      temperature: 0.3,
      max_output_tokens: 1200,
      system,
      input: [
        { role: 'user', content: "You will receive JSON context followed by the chat so far." },
        { role: 'user', content: "JSON context:" },
        { role: 'user', content: JSON.stringify(context, null, 2) },
        { role: 'user', content: "Chat messages:" },
        ...messages
      ]
    });

    const text =
      response.output_text ??
      response?.output?.[0]?.content?.[0]?.text ??
      'Sorry, I could not generate a response right now.';

    res.json({ ok: true, text, plan, condition });
  } catch (err) {
    console.error(err);
    res.status(500).json({ ok: false, error: 'Chat generation failed.' });
  }
});

// --- Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Whala server listening on http://localhost:${PORT}`);
});
