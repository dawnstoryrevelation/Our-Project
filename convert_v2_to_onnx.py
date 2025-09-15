# export_to_onnx.py
import os, sys, json, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- BiNLOP-3 ---------------------------------------
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
        if x.dim() == 2:  # [N, C]
            if g1.dim()==1:
                g1,g2,k1,k2 = g1[None,:],g2[None,:],k1[None,:],k2[None,:]
            return g2 * x + (1.0 - g1) * x.clamp(-k1, k1) + (g1 - g2) * x.clamp(-k2, k2)
        if k1.dim() == 1:
            # broadcast over spatial dims if present
            view = (1, -1) + (1,)*(x.dim()-2)
            g1 = g1.view(view); g2 = g2.view(view); k1 = k1.view(view); k2 = k2.view(view)
        return g2 * x + (1.0 - g1) * x.clamp(-k1, k1) + (g1 - g2) * x.clamp(-k2, k2)

# --------------------- CNN-V2 (BiNLOP-3 activation) ---------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act_layer=BiNLOP3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1)
        self.act = act_layer(out_ch)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act_layer=BiNLOP3):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, stride=stride, act_layer=act_layer)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1)
        self.act = act_layer(out_ch)
        self.short = (
            nn.Identity() if (in_ch == out_ch and stride == 1) else
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1))
        )
    def forward(self, x):
        s = self.short(x)
        x = self.bn2(self.conv1(x))
        x = self.act(x + s)
        return x

class B_Eye_O_Marker_CNN_V2(nn.Module):
    """
    The V2 BiNLOP-3 CNN used in the later runs:
      - Input: [B, 3, 384, 384] (defaults; configurable)
      - Stages: widths roughly (96, 192, 384, 640) -> ~20M params
      - Head: GAP + Linear(num_classes)
    """
    def __init__(self, num_classes=9, chs=(96, 192, 384, 640), in_ch=3):
        super().__init__()
        self.stem = ConvBNAct(in_ch, chs[0], stride=2, act_layer=BiNLOP3)
        self.stage1 = nn.Sequential(BasicBlock(chs[0], chs[0], act_layer=BiNLOP3),
                                    BasicBlock(chs[0], chs[0], act_layer=BiNLOP3))
        self.stage2 = nn.Sequential(BasicBlock(chs[0], chs[1], stride=2, act_layer=BiNLOP3),
                                    BasicBlock(chs[1], chs[1], act_layer=BiNLOP3))
        self.stage3 = nn.Sequential(BasicBlock(chs[1], chs[2], stride=2, act_layer=BiNLOP3),
                                    BasicBlock(chs[2], chs[2], act_layer=BiNLOP3))
        self.stage4 = nn.Sequential(BasicBlock(chs[2], chs[3], stride=2, act_layer=BiNLOP3),
                                    BasicBlock(chs[3], chs[3], act_layer=BiNLOP3))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(chs[3], num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x

# (Optional) Older tiny CNN fallback (if your weights happen to be from the very first OLIVESCNN)
class OLIVESCNN(nn.Module):
    def __init__(self, num_classes=9, chs=(64,160,288,416), in_ch=1):
        super().__init__()
        self.stem = ConvBNAct(in_ch, chs[0], stride=2, act_layer=BiNLOP3)
        self.stage1 = nn.Sequential(BasicBlock(chs[0], chs[0], act_layer=BiNLOP3),
                                    BasicBlock(chs[0], chs[0], act_layer=BiNLOP3))
        self.stage2 = nn.Sequential(BasicBlock(chs[0], chs[1], stride=2, act_layer=BiNLOP3),
                                    BasicBlock(chs[1], chs[1], act_layer=BiNLOP3))
        self.stage3 = nn.Sequential(BasicBlock(chs[1], chs[2], stride=2, act_layer=BiNLOP3),
                                    BasicBlock(chs[2], chs[2], act_layer=BiNLOP3))
        self.stage4 = nn.Sequential(BasicBlock(chs[2], chs[3], stride=2, act_layer=BiNLOP3),
                                    BasicBlock(chs[3], chs[3], act_layer=BiNLOP3))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(chs[3], num_classes))
    def forward(self, x):
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        return self.head(x)

# ------------------------------ Utils -----------------------------------------
def read_metadata_safetensors(path):
    try:
        from safetensors import safe_open
        md = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            md = f.metadata() or {}
        # unpack nested json strings if present
        out = dict(md)
        for k in ("cfg", "classes"):
            if k in md:
                try:
                    out[k] = json.loads(md[k])
                except Exception:
                    pass
        return out
    except Exception as e:
        print(f"[warn] Could not read safetensors metadata: {e}")
        return {}

def load_state_safetensors(path):
    from safetensors.torch import load_file
    return load_file(path, device="cpu")

def build_model_from_metadata(meta, arch_hint="v2"):
    # defaults
    num_classes = 9
    in_ch = 3
    chs_v2 = (96, 192, 384, 640)

    # try infer from metadata
    if "classes" in meta and isinstance(meta["classes"], (list, tuple)):
        num_classes = len(meta["classes"])
    if "cfg" in meta and isinstance(meta["cfg"], dict):
        if "image_size" in meta["cfg"]:
            pass  # we only need it for dummy input
        if "in_ch" in meta["cfg"]:
            in_ch = int(meta["cfg"]["in_ch"])
    if "chs" in meta:
        try:
            chs_v2 = tuple(int(x) for x in json.loads(meta["chs"]))
        except Exception:
            pass

    if arch_hint.lower() == "olives":
        return OLIVESCNN(num_classes=num_classes, in_ch=in_ch)
    else:
        return B_Eye_O_Marker_CNN_V2(num_classes=num_classes, chs=chs_v2, in_ch=in_ch)

def main():
    p = argparse.ArgumentParser(description="Export B-Eye-O-Marker CNN-V2 (BiNLOP-3) to ONNX")
    p.add_argument("--weights", type=str, required=True, help="Path to .safetensors weights")
    p.add_argument("--out", type=str, required=True, help="Output ONNX path")
    p.add_argument("--image-size", type=int, default=384, help="Input H=W (default: 384)")
    p.add_argument("--channels", type=int, default=3, help="Input channels (default: 3)")
    p.add_argument("--arch", type=str, default="v2", choices=["v2","olives"], help="Architecture hint")
    p.add_argument("--opset", type=int, default=18, help="ONNX opset (>=17 recommended)")
    p.add_argument("--dynamic", action="store_true", help="Export with dynamic batch/shape axes")
    p.add_argument("--simplify", action="store_true", help="Run onnx-simplifier if available")
    args = p.parse_args()

    if not os.path.exists(args.weights):
        print(f"[error] Weights not found: {args.weights}")
        sys.exit(1)

    print(f"[info] Reading metadata from: {args.weights}")
    meta = read_metadata_safetensors(args.weights)
    print(f"[info] Metadata keys: {list(meta.keys())}")

    # Build model + load state
    model = build_model_from_metadata(meta, arch_hint=args.arch)
    state = load_state_safetensors(args.weights)

    # Try strict=True first; fall back to strict=False
    try:
        model.load_state_dict(state, strict=True)
        print("[info] Loaded weights (strict=True).")
    except Exception as e:
        print(f"[warn] Strict load failed: {e}")
        model.load_state_dict(state, strict=False)
        print("[info] Loaded weights (strict=False).")

    model.eval()
    model.cpu()

    # Input shape
    C = args.channels
    H = W = args.image_size
    dummy = torch.randn(1, C, H, W, dtype=torch.float32)

    # Dynamic axes
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch"}
        }

    # Export
    print(f"[info] Exporting to ONNX: {args.out} (opset={args.opset})")
    torch.onnx.export(
        model,
        dummy,
        args.out,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes
    )

    # Validate ONNX
    try:
        import onnx
        onnx_model = onnx.load(args.out)
        onnx.checker.check_model(onnx_model)
        print("[info] ONNX model check passed.")
    except Exception as e:
        print(f"[warn] Could not validate ONNX with onnx.checker: {e}")

    # Optional simplify
    if args.simplify:
        try:
            import onnxsim
            print("[info] Running onnx-simplifier...")
            simplified, ok = onnxsim.simplify(args.out, dynamic_input_shape=args.dynamic)
            if ok:
                onnx.save(simplified, args.out)
                print("[info] Simplified ONNX saved.")
            else:
                print("[warn] onnx-simplifier returned ok=False (kept original).")
        except Exception as e:
            print(f"[warn] onnx-simplifier not run: {e}")

    # Quick inference test (CPU)
    try:
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        sess = ort.InferenceSession(args.out, providers=providers)
        outs = sess.run(None, {"input": dummy.numpy()})
        print(f"[info] ONNX runtime test OK. Output shape: {outs[0].shape}")
    except Exception as e:
        print(f"[warn] Skipping ORT test: {e}")

    print("[done] Export complete.")

if __name__ == "__main__":
    main()
