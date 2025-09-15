# train_b_eye_o_marker_v2.py
# B-Eye-O-Marker CNN-V2 — multi-source, multi-label ophthalmology training
# Target: a2-highgpu-4g (4x A100). Robust, fast, DDP, AMP, Triton BiNLOP-3, masked loss, fair splits.
import os, math, time, random, hashlib, io, json, gc, sys, subprocess, shutil
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ===================== ALL CACHES / ARTIFACTS TO /content =====================
BASE_DIR = "/content"
os.makedirs(BASE_DIR, exist_ok=True)
CACHE_ROOT = os.path.join(BASE_DIR, ".cache")
os.makedirs(CACHE_ROOT, exist_ok=True)

ENV_MAP = {
    "HF_HOME":                 os.path.join(CACHE_ROOT, "hf"),
    "HUGGINGFACE_HUB_CACHE":   os.path.join(CACHE_ROOT, "hf", "hub"),
    "HF_DATASETS_CACHE":       os.path.join(CACHE_ROOT, "hf", "datasets"),
    "TRANSFORMERS_CACHE":      os.path.join(CACHE_ROOT, "hf", "transformers"),
    "TORCH_HOME":              os.path.join(CACHE_ROOT, "torch"),
    "XDG_CACHE_HOME":          os.path.join(CACHE_ROOT, "xdg"),
    "TRITON_CACHE_DIR":        os.path.join(CACHE_ROOT, "triton"),
    "CUDA_CACHE_PATH":         os.path.join(CACHE_ROOT, "cuda"),
    "MPLCONFIGDIR":            os.path.join(CACHE_ROOT, "mpl"),
    "PIP_CACHE_DIR":           os.path.join(CACHE_ROOT, "pip"),
}
for k, v in ENV_MAP.items():
    os.makedirs(v, exist_ok=True)
    os.environ[k] = v

# ============================== Environment & perf =============================
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision", "torchaudio", "--cache-dir", ENV_MAP["PIP_CACHE_DIR"]])
    import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# Triton (for BiNLOP-3)
HAS_TRITON = False
if torch.cuda.is_available():
    try:
        import triton
        import triton.language as tl
        HAS_TRITON = True
    except Exception:
        HAS_TRITON = False

# DDP init
import torch.distributed as dist
def ddp_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", timeout=torch.distributed.timedelta(seconds=1800))
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        return True, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    return False, 0, 1, 0
IS_DDP, RANK, WORLD, LOCAL_RANK = ddp_init()
DEVICE = torch.device("cuda", LOCAL_RANK) if torch.cuda.is_available() else torch.device("cpu")
def is_main(): return (not IS_DDP) or (RANK == 0)
def barrier():
    if IS_DDP: dist.barrier()

# Helper to safe-install deps with pip cache in /content
def ensure_pkg(pkg: str, extras: List[str] = None):
    try:
        __import__(pkg)
    except Exception:
        if is_main():
            cmd = [sys.executable, "-m", "pip", "install", "-q", pkg, "--cache-dir", ENV_MAP["PIP_CACHE_DIR"]]
            if extras: cmd += extras
            subprocess.check_call(cmd)
        barrier()
        __import__(pkg)

# Core libs (cached to /content via env)
ensure_pkg("datasets")
ensure_pkg("safetensors")
ensure_pkg("PIL")
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from safetensors.torch import save_file as safetensors_save

try:
    import thop
except Exception:
    ensure_pkg("thop")
    import thop

# =================================== Config ===================================
@dataclass
class CFG:
    seed: int = 42
    image_size: int = 384
    epochs: int = 5
    batch_size: int = 64              # per GPU
    num_workers: int = 6              # per process
    lr: float = 2.0e-3
    weight_decay: float = 0.04
    warmup_epochs: int = 1
    amp_dtype: str = "bf16"           # on A100

    # dataset caps
    eyepacs_cap: int = 5000
    olives_dc_cap: int = 20000
    olives_bio_cap: int = 20000

    # label ontology
    classes: Tuple[str, ...] = (
        "Normal", "DiabeticRetinopathy", "DiabeticMacularEdema",
        "AgeRelatedMacularDegeneration", "Glaucoma", "RetinalDetachment",
        "EpiretinalMembrane", "VitreousDetachment", "Other"
    )

    # ALL artifacts into /content
    root_dir: str = BASE_DIR
    cache_dir: str = ENV_MAP["HF_DATASETS_CACHE"]
    ckpt_dir: str = os.path.join(BASE_DIR, "checkpoints")
    metrics_csv: str = os.path.join(BASE_DIR, "metrics_B-Eye-O-Marker_CNN-V2.csv")
    plots_dir: str = os.path.join(BASE_DIR, "plots")
    model_name: str = "B-Eye-O-Marker_CNN-V2"

cfg = CFG()
os.makedirs(cfg.ckpt_dir, exist_ok=True)
os.makedirs(cfg.plots_dir, exist_ok=True)

# Seed
def set_seed(seed:int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(cfg.seed)

# ============================= BiNLOP-3 (Triton) ==============================
class BiNLOP3(nn.Module):
    def __init__(self, channels:int, gamma_min:float=0.5, per_channel:bool=True,
                 init_g1:float=0.75, init_g2:float=0.6, init_k1:float=1.0, init_k2:float=6.0):
        super().__init__()
        self.channels = channels
        self.gamma_min = float(gamma_min)
        shape = (channels,) if per_channel else (1,)
        def inv_sig(y):
            y = max(min(y, 1-1e-6), 1e-6)
            return math.log(y/(1-y))
        g1_target = (init_g1 - self.gamma_min) / (1 - self.gamma_min)
        g2_target = (init_g2 - self.gamma_min) / (init_g1 - self.gamma_min)
        s1_hat_init = math.log(max(init_k1, 1e-4))
        d_hat_init  = math.log(max(init_k2 - init_k1, 1e-4))
        self.g1_hat = nn.Parameter(torch.full(shape, inv_sig(g1_target), dtype=torch.float32))
        self.g2_hat = nn.Parameter(torch.full(shape, inv_sig(g2_target), dtype=torch.float32))
        self.s1_hat = nn.Parameter(torch.full(shape, s1_hat_init, dtype=torch.float32))
        self.d_hat  = nn.Parameter(torch.full(shape, d_hat_init, dtype=torch.float32))
    def _params(self):
        gamma_min = self.gamma_min
        g1 = gamma_min + (1 - gamma_min) * torch.sigmoid(self.g1_hat)
        g2 = gamma_min + (g1 - gamma_min) * torch.sigmoid(self.g2_hat)
        k1 = torch.exp(self.s1_hat)
        k2 = k1 + torch.exp(self.d_hat)
        return g1, g2, k1, k2
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g1, g2, k1, k2 = self._params()
        if k1.dim() == 1:
            g1 = g1.view(1, -1, 1, 1); g2 = g2.view(1, -1, 1, 1)
            k1 = k1.view(1, -1, 1, 1); k2 = k2.view(1, -1, 1, 1)
        return g2 * x + (1.0 - g1) * torch.clamp(x, min=-k1, max=k1) \
                       + (g1 - g2) * torch.clamp(x, min=-k2, max=k2)

if HAS_TRITON:
    import triton.language as tl
    @triton.jit
    def _binlop3_kernel(X, G1, G2, K1, K2, Y, n_elements, H, W, C, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask, other=0.0)
        hw = H * W
        chw = C * hw
        rem1 = offs % chw
        c_idx = rem1 // hw
        g1 = tl.load(G1 + c_idx, mask=mask)
        g2 = tl.load(G2 + c_idx, mask=mask)
        k1 = tl.load(K1 + c_idx, mask=mask)
        k2 = tl.load(K2 + c_idx, mask=mask)
        cl1 = tl.minimum(tl.maximum(x, -k1), k1)
        cl2 = tl.minimum(tl.maximum(x, -k2), k2)
        y = g2 * x + (1.0 - g1) * cl1 + (g1 - g2) * cl2
        tl.store(Y + offs, y, mask=mask)
    class BiNLOP3_Triton(BiNLOP3):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
                g1, g2, k1, k2 = self._params()
                x_c = x.contiguous(); N, C, H, W = x_c.shape
                y = torch.empty_like(x_c)
                BLOCK = 1024
                n = x_c.numel()
                grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
                _binlop3_kernel[grid](x_c, g1.contiguous(), g2.contiguous(),
                                      k1.contiguous(), k2.contiguous(),
                                      y, n, H, W, C, BLOCK_SIZE=BLOCK)
                return y
            return super().forward(x)
else:
    BiNLOP3_Triton = BiNLOP3

# ================================== Model =====================================
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act_layer=BiNLOP3_Triton):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1)
        self.act = act_layer(out_ch)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act_layer=BiNLOP3_Triton):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, stride=stride, act_layer=act_layer)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1)
        self.act = act_layer(out_ch)
        self.short = nn.Identity() if (in_ch == out_ch and stride == 1) else \
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1))
    def forward(self, x):
        s = self.short(x)
        x = self.bn2(self.conv1(x))
        x = self.act(x + s)
        return x

class B_Eye_O_Marker_CNN_V2(nn.Module):
    def __init__(self, num_classes: int, chs=(96, 192, 384, 640)):
        super().__init__()
        # ~20M params for 3x384x384
        self.stem = ConvBNAct(3, chs[0], stride=2)
        self.stage1 = nn.Sequential(BasicBlock(chs[0], chs[0]), BasicBlock(chs[0], chs[0]))
        self.stage2 = nn.Sequential(BasicBlock(chs[0], chs[1], stride=2), BasicBlock(chs[1], chs[1]))
        self.stage3 = nn.Sequential(BasicBlock(chs[1], chs[2], stride=2), BasicBlock(chs[2], chs[2]))
        self.stage4 = nn.Sequential(BasicBlock(chs[2], chs[3], stride=2), BasicBlock(chs[3], chs[3]))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(chs[3], num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        x = self.head(x)
        return x

def count_params(m): return sum(p.numel() for p in m.parameters())
_tmp = B_Eye_O_Marker_CNN_V2(num_classes=9)
if is_main(): print(f"[Model] Params: {count_params(_tmp)/1e6:.2f}M (target ~20M)")
del _tmp

# ================================= Transforms =================================
train_tfms = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
])
eval_tfms = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])

# ===================== Label mapping & dataset loaders =========================
CLASSES = list(cfg.classes)
IDX = {k:i for i,k in enumerate(CLASSES)}

def empty_label():
    y = torch.full((len(CLASSES),), -1, dtype=torch.int8)  # -1 unknown / masked
    return y
def set_pos(y, cls): y[IDX[cls]] = 1
def set_neg(y, cls): y[IDX[cls]] = 0

OCT_TRIGGERS = {
    "normal": ["normal retinal thickness", "normal retinal", "normal"],
    "dr": ["diabetic retinopathy", "diabetic retinal"],
    "dme": ["cystoid macular edema", "macular edema", "dme"],
    "amd": ["macular degeneration", "amd"],
    "rd": ["retinal detachment", "detachment"],
    "erm": ["epiretinal membrane", "erm"],
    "vd": ["vitreous detachment"]
}
def map_oct_text(text: str):
    s = text.lower()
    y = empty_label(); hit = False
    if any(t in s for t in OCT_TRIGGERS["normal"]): set_pos(y, "Normal"); hit = True
    if any(t in s for t in OCT_TRIGGERS["dr"]): set_pos(y, "DiabeticRetinopathy"); set_neg(y,"Normal"); hit = True
    if any(t in s for t in OCT_TRIGGERS["dme"]): set_pos(y, "DiabeticMacularEdema"); set_neg(y,"Normal"); hit = True
    if any(t in s for t in OCT_TRIGGERS["amd"]): set_pos(y, "AgeRelatedMacularDegeneration"); set_neg(y,"Normal"); hit = True
    if any(t in s for t in OCT_TRIGGERS["rd"]): set_pos(y, "RetinalDetachment"); set_neg(y,"Normal"); hit = True
    if any(t in s for t in OCT_TRIGGERS["erm"]): set_pos(y, "EpiretinalMembrane"); set_neg(y,"Normal"); hit = True
    if any(t in s for t in OCT_TRIGGERS["vd"]): set_pos(y, "VitreousDetachment"); set_neg(y,"Normal"); hit = True
    return y

def map_aptos_or_eyepacs(label_code:int):
    y = empty_label()
    if label_code == 0:
        set_pos(y, "Normal"); set_neg(y, "DiabeticRetinopathy")
    else:
        set_pos(y, "DiabeticRetinopathy"); set_neg(y,"Normal")
    return y

def map_smdg(sample: dict):
    y = empty_label()
    # string labels first
    for k in sample.keys():
        v = sample[k]
        if isinstance(v, str):
            l = v.strip().lower()
            if l in ("glaucoma", "non_glaucoma", "glaucoma_suspect"):
                if l == "glaucoma": set_pos(y, "Glaucoma")
                elif l == "non_glaucoma": set_neg(y, "Glaucoma")
                return y
    # numeric fallback {1,0,-1}
    for k in sample.keys():
        v = sample[k]
        if isinstance(v, int) and v in (-1,0,1):
            if v == 1: set_pos(y, "Glaucoma")
            elif v == 0: set_neg(y, "Glaucoma")
            return y
    return y

def map_olives_dc(sample: dict):
    y = empty_label()
    dl_key = "Disease Label" if "Disease Label" in sample else "DiseaseLabel"
    if dl_key in sample:
        v = int(sample[dl_key])
        if v == 0: set_pos(y, "DiabeticRetinopathy"); set_neg(y,"Normal")
        elif v == 1: set_pos(y, "DiabeticMacularEdema"); set_neg(y,"Normal")
    return y

def map_olives_bio(sample: dict):
    y = empty_label()
    # Treat B5>0 as DME present
    b5 = None
    for k in sample.keys():
        if k.strip().upper() == "B5" or k.strip().upper().startswith("DRT"):
            b5 = sample[k]; break
    if b5 is not None:
        try:
            if float(b5) > 0:
                set_pos(y, "DiabeticMacularEdema"); set_neg(y,"Normal")
        except: pass
    return y

# -------------------------- HF utility helpers (cached) ------------------------
def take_first_n(dataset, n: int):
    n = min(n, len(dataset))
    return dataset.select(range(n))

def pil_from_any(img):
    if isinstance(img, Image.Image): return img
    if isinstance(img, bytes): return Image.open(io.BytesIO(img)).convert("RGB")
    try:
        return Image.fromarray(img)
    except Exception:
        # If hf image feature returns dict {bytes,...}
        return Image.open(io.BytesIO(img)).convert("RGB")

def hash_image(img_pil: Image.Image) -> str:
    thumb = img_pil.convert("RGB").resize((64,64), Image.BILINEAR)
    return hashlib.sha1(thumb.tobytes()).hexdigest()

class Entry:
    __slots__ = ("src","idx","conf","patient","eye")
    def __init__(self, src, idx, conf=None, patient=None, eye=None):
        self.src=src; self.idx=idx; self.conf=conf; self.patient=patient; self.eye=eye

class UnifiedHFDataset(Dataset):
    """ Harmonized, deduped manifest across 5 datasets, fair splits, cached in /content """
    def __init__(self, split: str, manifest: List[Entry], datasets_cache: Dict[str, any],
                 tfm, ontology: List[str]):
        self.split = split
        self.manifest = manifest
        self.ds = datasets_cache
        self.tfm = tfm
        self.ontology = ontology
    def __len__(self): return len(self.manifest)
    def _load_sample(self, e: Entry):
        ds = self.ds[e.src] if e.conf is None else self.ds[f"{e.src}::{e.conf}"]
        sample = ds[e.idx]
        # image
        img = None
        for k in ("Image", "image", "img"):
            if k in sample:
                img = sample[k]; break
        if img is None:
            for k,v in sample.items():
                if isinstance(v, Image.Image):
                    img = v; break
        img = pil_from_any(img).convert("RGB")  # force 3ch
        # label mapping
        if e.src == "aptos" or e.src == "eyepacs":
            y = map_aptos_or_eyepacs(int(sample.get("label_code", sample.get("label", 0))))
        elif e.src == "smdg":
            y = map_smdg(sample)
        elif e.src == "retinal_oct":
            y = map_oct_text(str(sample.get("text", "")))
        elif e.src == "olives" and e.conf == "disease_classification":
            y = map_olives_dc(sample)
        elif e.src == "olives" and e.conf == "biomarker_detection":
            y = map_olives_bio(sample)
        else:
            y = empty_label()
        m = (y != -1)
        x = self.tfm(img)
        return x, y.to(torch.int8), m
    def __getitem__(self, i):
        return self._load_sample(self.manifest[i])

def build_datasets_and_splits():
    # All datasets download/cache to /content via cache_dir + env above
    dm = "reuse_dataset_if_exists"
    aptos = load_dataset("bumbledeep/aptos", split="train", cache_dir=cfg.cache_dir, download_mode=dm)
    eyepacs = load_dataset("bumbledeep/eyepacs", split="train", cache_dir=cfg.cache_dir, download_mode=dm)
    eyepacs = take_first_n(eyepacs, cfg.eyepacs_cap)
    smdg = load_dataset("bumbledeep/smdg-full-dataset", split="train", cache_dir=cfg.cache_dir, download_mode=dm)
    octset = load_dataset("aditya11997/retinal_oct_analysis2", split="train", cache_dir=cfg.cache_dir, download_mode=dm)
    olives_dc = load_dataset("gOLIVES/OLIVES_Dataset", "disease_classification", split="train", cache_dir=cfg.cache_dir, download_mode=dm)
    olives_dc = take_first_n(olives_dc, cfg.olives_dc_cap)
    olives_bio = load_dataset("gOLIVES/OLIVES_Dataset", "biomarker_detection", split="train", cache_dir=cfg.cache_dir, download_mode=dm)
    olives_bio = take_first_n(olives_bio, cfg.olives_bio_cap)

    ds_cache = {
        "aptos": aptos,
        "eyepacs": eyepacs,
        "smdg": smdg,
        "retinal_oct": octset,
        "olives::disease_classification": olives_dc,
        "olives::biomarker_detection": olives_bio,
    }

    # Manifest building with dedupe + patient-aware splits for OLIVES
    seen = set()
    man_train, man_val, man_test = [], [], []

    def patient_split(idxs_by_group: Dict[str, List[int]], ratios=(0.8,0.1,0.1)):
        groups = list(idxs_by_group.keys()); random.shuffle(groups)
        n = len(groups); n_tr = int(n*ratios[0]); n_val = int(n*ratios[1])
        gtr = set(groups[:n_tr]); gva = set(groups[n_tr:n_tr+n_val]); gte = set(groups[n_tr+n_val:])
        out = {"train":[], "val":[], "test":[]}
        for g, idxs in idxs_by_group.items():
            if g in gtr: out["train"].extend(idxs)
            elif g in gva: out["val"].extend(idxs)
            else: out["test"].extend(idxs)
        return out

    def add_entries(name, ds, conf=None, group_keys: Tuple[str,...]=()):
        n = len(ds)
        groups = {}
        for i in range(n):
            s = ds[i]
            # image -> hash
            img = None
            for k in ("Image","image","img"):
                if k in s: img = s[k]; break
            if img is None:
                for k,v in s.items():
                    if isinstance(v, Image.Image): img=v; break
            if img is None: continue
            img = pil_from_any(img).convert("RGB")
            h = hash_image(img)
            if h in seen: continue
            seen.add(h)
            # group key
            g = None
            for k in group_keys:
                if k in s:
                    g = f"{k}:{s[k]}"; break
            if g is None: g = "nogroup"
            groups.setdefault(g, []).append(i)
        splits = patient_split(groups)
        for sp, idxs in splits.items():
            for i in idxs:
                e = Entry(src=name, idx=i, conf=conf)
                if sp=="train": man_train.append(e)
                elif sp=="val": man_val.append(e)
                else: man_test.append(e)

    add_entries("aptos", aptos, conf=None, group_keys=())
    add_entries("eyepacs", eyepacs, conf=None, group_keys=())
    add_entries("smdg", smdg, conf=None, group_keys=())
    add_entries("retinal_oct", octset, conf=None, group_keys=())
    add_entries("olives", olives_dc, conf="disease_classification", group_keys=("Patient_ID","Eye_ID"))
    add_entries("olives", olives_bio, conf="biomarker_detection", group_keys=("Patient_ID","Eye_ID"))

    for lst in (man_train, man_val, man_test): random.shuffle(lst)
    return ds_cache, man_train, man_val, man_test

if is_main(): print("[Data] Loading and building manifests (cache=/content)...")
ds_cache, man_train, man_val, man_test = build_datasets_and_splits()
if is_main(): print(f"[Data] Split sizes: train={len(man_train)} val={len(man_val)} test={len(man_test)}")

# ================================= Dataloaders ================================
class Collate:
    def __call__(self, batch):
        xs, ys, ms = zip(*batch)
        x = torch.stack(xs, 0)
        y = torch.stack(ys, 0).to(torch.int8)
        m = torch.stack(ms, 0).to(torch.bool)
        return x, y, m

train_ds = UnifiedHFDataset("train", man_train, ds_cache, tfm=train_tfms, ontology=CLASSES)
val_ds   = UnifiedHFDataset("val",   man_val,   ds_cache, tfm=eval_tfms,   ontology=CLASSES)
test_ds  = UnifiedHFDataset("test",  man_test,  ds_cache, tfm=eval_tfms,   ontology=CLASSES)

train_sampler = DistributedSampler(train_ds, num_replicas=WORLD, rank=RANK, shuffle=True, drop_last=False) if IS_DDP else None
val_sampler   = DistributedSampler(val_ds, num_replicas=WORLD, rank=RANK, shuffle=False, drop_last=False) if IS_DDP else None

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
                          sampler=train_sampler, num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=True, prefetch_factor=4, collate_fn=Collate())
val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                          sampler=val_sampler, num_workers=max(2, cfg.num_workers//2), pin_memory=True,
                          persistent_workers=True, prefetch_factor=4, collate_fn=Collate())

# ============================ Loss & metrics ==================================
def masked_bce_with_logits(logits, y_int8, mask_bool, pos_weight=None):
    target = (y_int8 == 1).float()
    logits = logits.float()
    loss = F.binary_cross_entropy_with_logits(logits[mask_bool], target[mask_bool])
    return loss

def compute_batch_metrics(logits, y_int8, mask_bool, thr=0.5):
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        pred = (probs >= thr).to(torch.int8)
        tgt  = (y_int8 == 1).to(torch.int8)
        known = mask_bool
        tp = ((pred==1) & (tgt==1) & known).sum().item()
        fp = ((pred==1) & (tgt==0) & known).sum().item()
        fn = ((pred==0) & (tgt==1) & known).sum().item()
        precision = tp / max(1, (tp+fp))
        recall = tp / max(1, (tp+fn))
        f1 = 2*precision*recall / max(1e-8, (precision+recall))
        acc = ((pred==tgt) & known).sum().item() / max(1, known.sum().item())
        return precision, recall, f1, acc

def estimate_pos_weight(loader, max_batches=50):
    pos = torch.zeros(len(CLASSES), dtype=torch.float64, device=DEVICE)
    neg = torch.zeros(len(CLASSES), dtype=torch.float64, device=DEVICE)
    seen = torch.zeros(len(CLASSES), dtype=torch.float64, device=DEVICE)
    bcount = 0
    for x, y, m in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE); m = m.to(DEVICE)
        tgt = (y==1).to(torch.float32)
        pos += (tgt*m).sum(dim=0)
        neg += ((y==0).to(torch.float32)*m).sum(dim=0)
        seen += m.sum(dim=0)
        bcount += 1
        if bcount >= max_batches: break
    pw = torch.where(pos>0, neg.clamp(min=1)/pos.clamp(min=1), torch.ones_like(pos))
    return pw.to(torch.float32)

# =========================== Model / Opt / Sched ===============================
model = B_Eye_O_Marker_CNN_V2(num_classes=len(CLASSES)).to(DEVICE)
if IS_DDP:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, broadcast_buffers=False, find_unused_parameters=False)

opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9,0.999))
class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, opt, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps; self.total_steps = total_steps
        super().__init__(opt, last_epoch)
    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            return [base * step / max(1, self.warmup_steps) for base in self.base_lrs]
        t = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))
        return [base * factor for base in self.base_lrs]

steps_per_epoch = math.ceil(len(train_ds) / (cfg.batch_size * (WORLD if IS_DDP else 1)))
max_steps = steps_per_epoch * cfg.epochs
warmup_steps = max(1, int(cfg.warmup_epochs * steps_per_epoch))
sched = WarmupCosine(opt, warmup_steps, max_steps)

amp_dtype = torch.bfloat16 if (cfg.amp_dtype.lower()=="bf16" and torch.cuda.is_bf16_supported()) else torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype==torch.float16))

if is_main(): print("[Train] Estimating class pos_weight...")
pos_weight = estimate_pos_weight(train_loader).to(DEVICE)

# FLOPs -> MFU
def compute_flops_per_img():
    m = B_Eye_O_Marker_CNN_V2(num_classes=len(CLASSES)).to(DEVICE)
    m.eval()
    dummy = torch.randn(1,3,cfg.image_size,cfg.image_size, device=DEVICE)
    flops, params = thop.profile(m, inputs=(dummy, ), verbose=False)
    del m, dummy
    torch.cuda.empty_cache()
    return float(flops)
flops_per_img = compute_flops_per_img()
A100_BF16_TFLOPS = 312.0
if is_main(): print(f"[Model] Estimated FLOPs/img: {flops_per_img/1e9:.2f} GFLOPs")

# ================================ Logging =====================================
best_val_acc = 0.0
best_path = os.path.join(cfg.ckpt_dir, f"{cfg.model_name}.safetensors")
metrics_log = []

# =========================== Train / Val loops ================================
def run_epoch(epoch:int, model, loader, train=True):
    if train: model.train()
    else: model.eval()
    t0 = time.time()
    n_seen, loss_sum, prec_sum, rec_sum, f1_sum, acc_sum, steps = 0,0.0,0.0,0.0,0.0,0.0,0
    max_mem = 0
    for x, y, m in loader:
        x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True); m = m.to(DEVICE, non_blocking=True)
        bs = x.size(0)
        if train:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits = model(x)
                loss = masked_bce_with_logits(logits, y, m)
            if amp_dtype==torch.float16:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            sched.step()
        else:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits = model(x)
                loss = masked_bce_with_logits(logits, y, m)
        p, r, f1, acc = compute_batch_metrics(logits.detach(), y, m)
        n_seen += bs; loss_sum += float(loss.item()); prec_sum += p; rec_sum += r; f1_sum += f1; acc_sum += acc; steps += 1
        if torch.cuda.is_available():
            max_mem = max(max_mem, torch.cuda.max_memory_allocated(device=DEVICE))
    # aggregate across DDP
    device = DEVICE
    vt = torch.tensor([loss_sum, prec_sum, rec_sum, f1_sum, acc_sum, float(n_seen), float(steps)],
                      device=device, dtype=torch.float64)
    if IS_DDP: dist.all_reduce(vt, op=dist.ReduceOp.SUM)
    loss_mean = (vt[0] / vt[6]).item()
    precision = (vt[1] / vt[6]).item()
    recall = (vt[2] / vt[6]).item()
    f1 = (vt[3] / vt[6]).item()
    acc = (vt[4] / vt[6]).item()
    n_total = int(vt[5].item())

    dt = time.time() - t0
    imgs_per_sec = n_total / max(1e-6, dt)
    achieved_tflops = (flops_per_img * n_total * 3.0) / dt / 1e12  # fwd+bwd+update ~3x
    denom = (A100_BF16_TFLOPS * (WORLD if IS_DDP else 1))
    mfu = float(achieved_tflops / max(1e-6, denom))
    mem_gb = (max_mem / (1024**3)) if torch.cuda.is_available() else 0.0

    return {"loss": loss_mean, "precision": precision, "recall": recall, "f1": f1, "acc": acc,
            "imgs_per_sec": imgs_per_sec, "mem_gb": mem_gb, "mfu": mfu, "dt_s": dt, "steps": int(steps), "n_seen": int(n_total)}

if is_main(): print("[Train] Starting training...")
for epoch in range(1, cfg.epochs+1):
    if IS_DDP:
        train_sampler.set_epoch(epoch); val_sampler.set_epoch(epoch)
    train_stats = run_epoch(epoch, model, train_loader, train=True)
    val_stats   = run_epoch(epoch, model, val_loader,   train=False)

    if is_main():
        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"train: loss={train_stats['loss']:.4f} F1={train_stats['f1']:.3f} P={train_stats['precision']:.3f} R={train_stats['recall']:.3f} acc={train_stats['acc']:.3f} "
              f"| val: loss={val_stats['loss']:.4f} acc={val_stats['acc']:.3f} "
              f"| thrpt={train_stats['imgs_per_sec']:.1f} img/s mem={train_stats['mem_gb']:.2f}GB MFU={train_stats['mfu']*100:.1f}%")
        metrics_log.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "val_accuracy": val_stats["acc"],
            "train_precision": train_stats["precision"],
            "train_recall": train_stats["recall"],
            "train_f1": train_stats["f1"],
            "throughput_img_per_s": train_stats["imgs_per_sec"],
            "memory_gb": train_stats["mem_gb"],
            "mfu": train_stats["mfu"],
            "train_acc": train_stats["acc"],
        })
        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            tensors = {k: v.detach().cpu() for k, v in (model.module.state_dict() if IS_DDP else model.state_dict()).items()}
            meta = {"cfg": json.dumps(cfg.__dict__), "classes": json.dumps(CLASSES), "best_val_acc": f"{best_val_acc:.4f}"}
            safetensors_save(tensors, best_path, metadata=meta)
            torch.save({"state_dict": (model.module.state_dict() if IS_DDP else model.state_dict()),
                        "cfg": cfg.__dict__, "classes": CLASSES, "epoch": epoch, "best_val_acc": best_val_acc},
                       os.path.join(cfg.ckpt_dir, f"{cfg.model_name}.pt"))
            print(f"  ↳ Saved best to {best_path} (val_acc={best_val_acc:.3f})")

barrier()
if is_main():
    print("[Train] Done. Saving final artifacts...")

    if not os.path.exists(best_path):
        tensors = {k: v.detach().cpu() for k, v in (model.module.state_dict() if IS_DDP else model.state_dict()).items()}
        meta = {"cfg": json.dumps(cfg.__dict__), "classes": json.dumps(CLASSES), "best_val_acc": f"{best_val_acc:.4f}"}
        safetensors_save(tensors, best_path, metadata=meta)
        torch.save({"state_dict": (model.module.state_dict() if IS_DDP else model.state_dict()),
                    "cfg": cfg.__dict__, "classes": CLASSES, "epoch": cfg.epochs, "best_val_acc": best_val_acc},
                   os.path.join(cfg.ckpt_dir, f"{cfg.model_name}.pt"))
        print(f"Saved final to {best_path}")

    # Metrics CSV (exact 9 metrics)
    import csv
    with open(cfg.metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch","f1","val_loss","loss","val_accuracy","precision","recall","throughput_img_per_s","memory_gb","mfu"])
        for row in metrics_log:
            w.writerow([row["epoch"], row["train_f1"], row["val_loss"], row["train_loss"], row["val_accuracy"],
                        row["train_precision"], row["train_recall"], row["throughput_img_per_s"], row["memory_gb"], row["mfu"]])
    print(f"[Metrics] CSV saved to {cfg.metrics_csv}")

    # 9 plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def plot_metric(name, key):
        xs = [r["epoch"] for r in metrics_log]
        ys = [r[key] for r in metrics_log]
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Epoch"); plt.ylabel(name); plt.title(name)
        out = os.path.join(cfg.plots_dir, f"{name.replace(' ','_')}.png")
        plt.savefig(out, bbox_inches="tight"); plt.close()
        return out
    plot_metric("F1 Score", "train_f1")
    plot_metric("Val Loss", "val_loss")
    plot_metric("Loss", "train_loss")
    plot_metric("Val Accuracy", "val_accuracy")
    plot_metric("Precision", "train_precision")
    plot_metric("Recall", "train_recall")
    plot_metric("Throughput img/s", "throughput_img_per_s")
    plot_metric("Memory GB", "memory_gb")
    plot_metric("MFU", "mfu")
    print(f"[Plots] Saved 9 plots to {cfg.plots_dir}")

    # ======================= FINAL CLEANUP / CACHE CLEAR =======================
    # 1) Flush GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    # 2) Remove any stray caches outside /content to prevent space leaks
    def safe_rm(path):
        try:
            if os.path.exists(path) and not os.path.realpath(path).startswith(BASE_DIR):
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

    for p in ["~/.cache", "/root/.cache", "~/.triton", "/root/.triton"]:
        safe_rm(os.path.expanduser(p))

    print("[Cleanup] Non-/content caches cleared; /content caches/artifacts preserved.")
    print("[Done] All artifacts are in /content.")
