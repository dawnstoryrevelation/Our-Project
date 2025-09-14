import os, json, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load

SAFETENSORS_PATH = "b_eye_o_marker_v1.safetensors"
ONNX_PATH        = "b_eye_o_marker_v1.onnx"
OPSET            = 18  # 17+ recommended; 18 is widely supported as of 2025
IMAGE_SIZE       = 224 # will be ignored at runtime due to dynamic axes

# -------------------------
# 1) Model code (matches training)
# -------------------------

class BiNLOP3(nn.Module):
    """
    φ(x) = γ2 x + (1 − γ1) clamp(x, −k1, k1) + (γ1 − γ2) clamp(x, −k2, k2)
    Safe parameterization with per-channel parameters (g1_hat, g2_hat, s1_hat, d_hat).
    All ops are basic torch ops → exportable to ONNX.
    """
    def __init__(self, channels:int, gamma_min:float=0.5, per_channel:bool=True):
        super().__init__()
        self.channels = channels
        self.gamma_min = float(gamma_min)
        shape = (channels,) if per_channel else (1,)

        # Parameters (will be loaded from state_dict)
        self.g1_hat = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        self.g2_hat = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        self.s1_hat = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        self.d_hat  = nn.Parameter(torch.zeros(shape, dtype=torch.float32))

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
            g1 = g1.view(1, -1, 1, 1)
            g2 = g2.view(1, -1, 1, 1)
            k1 = k1.view(1, -1, 1, 1)
            k2 = k2.view(1, -1, 1, 1)
        # clamp + fmadds are all ONNX-friendly
        return g2 * x + (1.0 - g1) * torch.clamp(x, min=-k1, max=k1) \
                       + (g1 - g2) * torch.clamp(x, min=-k2, max=k2)

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1)
        self.act  = BiNLOP3(out_ch)  # per-channel
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1)
        self.act   = BiNLOP3(out_ch)
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

class OLIVESCNN(nn.Module):
    def __init__(self, num_classes=2, chs=(64,160,288,416), in_ch=1):
        super().__init__()
        self.stem   = ConvBNAct(in_ch, chs[0], stride=2)
        self.stage1 = nn.Sequential(BasicBlock(chs[0], chs[0]), BasicBlock(chs[0], chs[0]))
        self.stage2 = nn.Sequential(BasicBlock(chs[0], chs[1], stride=2), BasicBlock(chs[1], chs[1]))
        self.stage3 = nn.Sequential(BasicBlock(chs[1], chs[2], stride=2), BasicBlock(chs[2], chs[2]))
        self.stage4 = nn.Sequential(BasicBlock(chs[2], chs[3], stride=2), BasicBlock(chs[3], chs[3]))
        self.head   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(chs[3], num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x

# -------------------------
# 2) Load state dict and infer shapes
# -------------------------
sd = safetensors_load(SAFETENSORS_PATH)  # dict[str, Tensor]
# Try to read metadata saved by our training pipeline (optional)
meta = getattr(sd, "metadata", {}) if hasattr(sd, "metadata") else {}
chs = None
num_classes = None

# A) If metadata exists, prefer it
if meta:
    try:
        if "chs" in meta:
            chs = tuple(json.loads(meta["chs"]))
        if "cfg" in meta:
            cfg = json.loads(meta["cfg"])
            if "num_classes" in cfg:
                num_classes = int(cfg["num_classes"])
    except Exception:
        pass

# B) Infer from tensor shapes if needed
#   - final linear weight: "head.2.weight" (or any head.*.weight with shape [num_classes, C])
linear_keys = [k for k,v in sd.items() if "head" in k and k.endswith("weight") and v.ndim == 2]
if linear_keys:
    lk = linear_keys[0]
    num_classes = sd[lk].shape[0] if num_classes is None else num_classes
    c_last = sd[lk].shape[1]
else:
    raise RuntimeError("Could not locate final linear layer in state_dict to infer classes/channels.")

#   - stem conv out_channels gives chs[0]
stem_keys = [k for k,v in sd.items() if k.endswith("stem.conv.weight")]
if stem_keys:
    ch0 = sd[stem_keys[0]].shape[0]
else:
    raise RuntimeError("Could not infer stem channels from state_dict (stem.conv.weight not found).")

#   - heuristics: read first conv weight at each stage to infer the rest
def find_first_conv_out(prefix):
    # e.g., "stage2.0.conv1.conv.weight"
    for k, v in sd.items():
        if k.endswith("conv.weight") and k.startswith(prefix):
            return v.shape[0]
    return None

ch1 = find_first_conv_out("stage2.0.conv1")
ch2 = find_first_conv_out("stage3.0.conv1")
ch3 = find_first_conv_out("stage4.0.conv1")
# Fallback: if any missing, try to infer from batchnorm or the head input
if ch1 is None:
    for k, v in sd.items():
        if k.startswith("stage2.0.bn2.weight"):
            ch1 = v.shape[0]
            break
if ch2 is None:
    for k, v in sd.items():
        if k.startswith("stage3.0.bn2.weight"):
            ch2 = v.shape[0]
            break
if ch3 is None:
    ch3 = c_last  # head in_features

chs = (ch0, ch1, ch2, ch3)

print(f"Inferred -> num_classes={num_classes}, chs={chs}")

# -------------------------
# 3) Rebuild model, load weights
# -------------------------
model = OLIVESCNN(num_classes=num_classes, chs=chs, in_ch=1)
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing:
    print("[warn] Missing keys not loaded:", missing)
if unexpected:
    print("[warn] Unexpected keys in state_dict:", unexpected)
model.eval()

# -------------------------
# 4) Export to ONNX
# -------------------------
# Use a dummy input; we’ll make N,H,W dynamic in ONNX.
dummy = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)

# Dynamic axes so you can change batch and spatial dims at inference:
dynamic_axes = {
    "input":  {0: "N", 2: "H", 3: "W"},
    "logits": {0: "N"}
}
input_names  = ["input"]
output_names = ["logits"]

torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    export_params=True,
    opset_version=OPSET,
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes
)
print(f"[ok] Exported ONNX -> {ONNX_PATH}")

# -------------------------
# 5) (Optional) Simplify + sanity check
# -------------------------
try:
    import convert_to_onnx
    from onnxsim import simplify
    m = convert_to_onnx.load(ONNX_PATH)
    m_simplified, check = simplify(m)
    if check:
        convert_to_onnx.save(m_simplified, ONNX_PATH)
        print("[ok] Simplified ONNX model saved.")
except Exception as e:
    print("[info] Skipped ONNX simplification:", repr(e))

# Quick runtime check (CPU)
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy.numpy()})
    print("[ok] ONNXRuntime forward pass worked. logits shape:", out[0].shape)
except Exception as e:
    print("[warn] ONNXRuntime check failed:", repr(e))
