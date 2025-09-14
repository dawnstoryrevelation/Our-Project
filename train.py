# our script for training B-Eye-O-Marker V1 

import os
os.environ.setdefault("PJRT_DEVICE", "TPU") 
os.environ.setdefault("XLA_USE_BF16", "1")
# stuff
import math
import time
import random 

from dataclasses import dataclass
# imports.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

# TPU setup for Colab.
USE_XLA = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    USE_XLA = True
except Exception:
    USE_XLA = False

# Triton for BiNLOP-3
HAS_TRITON = False
if not USE_XLA and torch.cuda.is_available():
    try:
        import triton
        import triton.language as tl
        HAS_TRITON = True
    except Exception:
        HAS_TRITON = False
# more imports 
from datasets import load_dataset
from torchvision import transforms
from safetensors.torch import save_file as safetensors_save

# BiNLOP-3!!!!
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

    def extra_repr(self):
        with torch.no_grad():
            g1, g2, k1, k2 = self._params()
            return f"channels={self.channels}, gamma_min={self.gamma_min}, g1~{g1.mean().item():.3f}, g2~{g2.mean().item():.3f}, k1~{k1.mean().item():.3f}, k2~{k2.mean().item():.3f}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g1, g2, k1, k2 = self._params()
        if k1.dim() == 1:
            g1 = g1.view(1, -1, 1, 1)
            g2 = g2.view(1, -1, 1, 1)
            k1 = k1.view(1, -1, 1, 1)
            k2 = k2.view(1, -1, 1, 1)
        return g2 * x + (1.0 - g1) * torch.clamp(x, min=-k1, max=k1) \
                       + (g1 - g2) * torch.clamp(x, min=-k2, max=k2)

# Triton implement
if HAS_TRITON:
    @triton.jit
    def _binlop3_kernel(X, G1, G2, K1, K2, Y, n_elements, H, W, C,
                        BLOCK_SIZE: tl.constexpr):
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
                x_c = x.contiguous()
                N, C, H, W = x_c.shape
                y = torch.empty_like(x_c)
                BLOCK = 1024
                n = x_c.numel()
                grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
                _binlop3_kernel[grid](
                    x_c, g1.contiguous(), g2.contiguous(), k1.contiguous(), k2.contiguous(),
                    y, n, H, W, C, BLOCK_SIZE=BLOCK
                )
                return y
            return super().forward(x)
else:
    BiNLOP3_Triton = BiNLOP3

# define the model

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act_layer=BiNLOP3_Triton):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1)
        self.act = act_layer(out_ch)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
# build model more
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act_layer=BiNLOP3_Triton):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, stride=stride, act_layer=act_layer)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1)
        self.act = act_layer(out_ch)
        self.short = (
            nn.Identity() if (in_ch == out_ch and stride == 1)
            else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                               nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1))
        )
    def forward(self, x):
        s = self.short(x)
        x = self.bn2(self.conv1(x))
        x = self.act(x + s)
        return x
# more...
class OLIVESCNN(nn.Module):
    def __init__(self, num_classes=2, chs=(64,160,288,416), in_ch=1):
        super().__init__()
        self.stem = ConvBNAct(in_ch, chs[0], stride=2)
        self.stage1 = nn.Sequential(BasicBlock(chs[0], chs[0]), BasicBlock(chs[0], chs[0]))
        self.stage2 = nn.Sequential(BasicBlock(chs[0], chs[1], stride=2), BasicBlock(chs[1], chs[1]))
        self.stage3 = nn.Sequential(BasicBlock(chs[1], chs[2], stride=2), BasicBlock(chs[2], chs[2]))
        self.stage4 = nn.Sequential(BasicBlock(chs[2], chs[3], stride=2), BasicBlock(chs[3], chs[3]))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(chs[3], num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x
# Helper to auto-pick widths so param count fits.
@torch.no_grad()
def choose_channels():
    candidates = [
        (48,112,192,256),
        (48,128,224,320),
        (56,128,224,320),
        (64,128,224,320),
        (64,144,240,320),
        (64,144,256,352),
    ]
    best = None
    for chs in candidates:
        m = OLIVESCNN(chs=chs)
        n = sum(p.numel() for p in m.parameters())
        if 6_000_000 <= n <= 7_000_000:
            best = (chs, n)
            break
        if best is None or abs(n - 6_500_000) < abs(best[1] - 6_500_000):
            best = (chs, n)
    chs, n = best
    print(f"Selected channels {chs} -> {n/1e6:.2f}M params")
    return chs

# data pipelines

@dataclass
class CFG:
    seed: int = 42
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 0  # TPU-friendly (avoid multiprocess workers on XLA)
    epochs: int = 4 
    lr: float = 2e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    label_key: str = 'Disease Label'  # per HF page

cfg = CFG()

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)

train_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((cfg.image_size, cfg.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
])

eval_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((cfg.image_size, cfg.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])
# stream windows
class HFStreamWindow(IterableDataset):
    def __init__(self, split: str, start: int, k: int, transform, label_key: str):
        super().__init__()
        self.dataset_stream = load_dataset('gOLIVES/OLIVES_Dataset', 'disease_classification', split=split, streaming=True)
        self.start = start
        self.k = k
        self.transform = transform
        self.label_key = label_key
    def __iter__(self):
        n = 0
        yielded = 0
        for ex in self.dataset_stream:
            if n < self.start:
                n += 1
                continue
            img = self.transform(ex['Image'])
            label = torch.tensor(ex[self.label_key], dtype=torch.long)
            yield img, label
            n += 1
            yielded += 1
            if yielded >= self.k:
                break

# Windows within the first 5,000 examples
TRAIN_START, TRAIN_K = 0, 20000
VAL_START,   VAL_K   = 20000, 3000
train_stream = HFStreamWindow('train', start=TRAIN_START, k=TRAIN_K, transform=train_tfms, label_key=cfg.label_key)
val_stream   = HFStreamWindow('train', start=VAL_START,   k=VAL_K,   transform=eval_tfms,   label_key=cfg.label_key)

train_loader = DataLoader(train_stream, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False, persistent_workers=False)
val_loader   = DataLoader(val_stream,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False, persistent_workers=False)

# utils

def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, opt, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(opt, last_epoch)
    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            return [base * step / max(1, self.warmup_steps) for base in self.base_lrs]
        t = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))
        return [base * factor for base in self.base_lrs]

# Device.
if USE_XLA:
    try:
        device = torch_xla.device()  # modern API. avoids deprecation warning
    except Exception:
        device = xm.xla_device()
    print("Using XLA device:", device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
auto_chs = choose_channels()
model = OLIVESCNN(num_classes=2, chs=auto_chs, in_ch=1).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
steps_per_epoch = math.ceil(TRAIN_K / cfg.batch_size)
val_steps = math.ceil(VAL_K / cfg.batch_size)
max_steps = steps_per_epoch * cfg.epochs
warmup_steps = max(1, int(cfg.warmup_epochs * steps_per_epoch))
sched = WarmupCosine(opt, warmup_steps, max_steps)

best_val_acc = 0.0
save_path = 'olives_cnn_binlop3.safetensors'

print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

for epoch in range(1, cfg.epochs + 1):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    t0 = time.time()

    for step, (x, y) in enumerate(train_loader, start=1):
        if step % 10 == 0:
            print(f"  step {step}/{steps_per_epoch}", flush=True)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        if USE_XLA:
            with torch.autocast("xla", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            loss.backward()
            xm.optimizer_step(opt, barrier=True)
            # Prefer new sync API when available
            try:
                import torch_xla as _tx
                _tx.sync()
            except Exception:
                xm.mark_step()
        else:
            autocast_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

        epoch_loss += loss.item()
        epoch_acc  += accuracy(logits, y)
        sched.step()

    train_loss = epoch_loss / max(1, steps_per_epoch)
    train_acc  = epoch_acc  / max(1, steps_per_epoch)

    model.eval()
    val_loss = 0.0
    val_acc  = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if USE_XLA:
                with torch.autocast("xla", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                # Sync step for XLA (prefer torch_xla.sync if available)
                try:
                    import torch_xla as _tx
                    if hasattr(_tx, "sync"):
                        _tx.sync()
                    else:
                        xm.mark_step()
                except Exception:
                    xm.mark_step()
            else:
                autocast_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
            val_loss += loss.item()
            val_acc  += accuracy(logits, y)

    val_loss /= max(1, val_steps)
    val_acc  /= max(1, val_steps)

    dt = time.time() - t0
    print(f"Epoch {epoch:02d}/{cfg.epochs} | {dt:.1f}s | train_loss={train_loss:.4f} train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        import json
        tensors = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        metadata = {"cfg": json.dumps(cfg.__dict__), "chs": json.dumps(list(auto_chs))}
        safetensors_save(tensors, save_path, metadata=metadata)
        print(f"  ↳ Saved best to {save_path} (val_acc={best_val_acc:.3f})")

print("Training complete.")
if not os.path.exists(save_path):
    import json
    tensors = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    metadata = {"cfg": json.dumps(cfg.__dict__), "chs": json.dumps(list(auto_chs))}
    safetensors_save(tensors, save_path, metadata=metadata)
print(f"Model state_dict saved to {os.path.abspath(save_path)}")
