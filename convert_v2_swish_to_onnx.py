# export_swish_to_onnx.py
import os
import sys
import json
import math
import argparse
import torch
import torch.nn as nn

# ---------------- Swish (SiLU) model definition -----------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act_layer=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1)
        self.act = act_layer()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act_layer=nn.SiLU):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, stride=stride, act_layer=act_layer)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1)
        self.act = act_layer()
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

class B_Eye_O_Marker_CNN_V2_Swish(nn.Module):
    """
    Swish / SiLU variant of B-Eye-O-Marker CNN-V2.
    Default widths chosen to approximate ~20M params; overridden by metadata if present.
    """
    def __init__(self, num_classes=9, chs=(96,192,384,640), in_ch=3):
        super().__init__()
        self.stem = ConvBNAct(in_ch, chs[0], stride=2, act_layer=nn.SiLU)
        self.stage1 = nn.Sequential(BasicBlock(chs[0], chs[0], act_layer=nn.SiLU),
                                    BasicBlock(chs[0], chs[0], act_layer=nn.SiLU))
        self.stage2 = nn.Sequential(BasicBlock(chs[0], chs[1], stride=2, act_layer=nn.SiLU),
                                    BasicBlock(chs[1], chs[1], act_layer=nn.SiLU))
        self.stage3 = nn.Sequential(BasicBlock(chs[1], chs[2], stride=2, act_layer=nn.SiLU),
                                    BasicBlock(chs[2], chs[2], act_layer=nn.SiLU))
        self.stage4 = nn.Sequential(BasicBlock(chs[2], chs[3], stride=2, act_layer=nn.SiLU),
                                    BasicBlock(chs[3], chs[3], act_layer=nn.SiLU))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(chs[3], num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x

# ---------------- utils to read safetensors metadata + weights -----------------
def read_metadata_safetensors(path):
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            md = f.metadata() or {}
        out = dict(md)
        # try to parse common nested json strings
        for k in ("cfg", "classes", "chs"):
            if k in out:
                try:
                    out[k] = json.loads(out[k])
                except Exception:
                    pass
        return out
    except Exception as e:
        print(f"[warn] Could not read safetensors metadata: {e}")
        return {}

def load_state_safetensors(path):
    from safetensors.torch import load_file
    return load_file(path, device="cpu")

def build_model_from_metadata(meta, arch_hint="swish"):
    # sensible defaults consistent with earlier scripts
    num_classes = 9
    in_ch = 3
    chs_default = (96,192,384,640)
    if "classes" in meta and isinstance(meta["classes"], (list,tuple)):
        num_classes = len(meta["classes"])
    if "cfg" in meta and isinstance(meta["cfg"], dict):
        if "in_ch" in meta["cfg"]:
            try: in_ch = int(meta["cfg"]["in_ch"])
            except: pass
    if "chs" in meta:
        try:
            if isinstance(meta["chs"], str):
                chs_try = json.loads(meta["chs"])
            else:
                chs_try = meta["chs"]
            chs_default = tuple(int(x) for x in chs_try)
        except Exception:
            pass
    # build Swish model
    return B_Eye_O_Marker_CNN_V2_Swish(num_classes=num_classes, chs=chs_default, in_ch=in_ch)

# ---------------------- main export routine -----------------------
def main():
    p = argparse.ArgumentParser(description="Export Swish B-Eye-O-Marker (.safetensors) to ONNX")
    p.add_argument("--weights", type=str, required=True, help="Path to .safetensors weights file")
    p.add_argument("--out", type=str, required=True, help="Output .onnx path")
    p.add_argument("--image-size", type=int, default=384, help="Input H/W (default 384)")
    p.add_argument("--channels", type=int, default=3, help="Input channels (default 3)")
    p.add_argument("--opset", type=int, default=18, help="ONNX opset version (default 18)")
    p.add_argument("--dynamic", action="store_true", help="Export with dynamic axes for batch/height/width")
    p.add_argument("--simplify", action="store_true", help="Run onnx-simplifier if available")
    p.add_argument("--arch", choices=["swish","olives"], default="swish", help="Architecture hint (swish/olives)")
    args = p.parse_args()

    if not os.path.exists(args.weights):
        print(f"[error] Weights file not found: {args.weights}")
        sys.exit(1)

    print(f"[info] Reading metadata from: {args.weights}")
    meta = read_metadata_safetensors(args.weights)
    if meta:
        print(f"[info] Metadata keys: {list(meta.keys())}")
    else:
        print("[info] No metadata found or could not parse.")

    # Build model and load state
    model = build_model_from_metadata(meta, arch_hint=args.arch)
    print(f"[info] Constructed model: {model.__class__.__name__} | params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    state = load_state_safetensors(args.weights)

    # load into model (try strict then fallback)
    try:
        model.load_state_dict(state, strict=True)
        print("[info] Loaded weights with strict=True")
    except Exception as e:
        print(f"[warn] strict=True load failed: {e}. Attempting strict=False load.")
        try:
            model.load_state_dict(state, strict=False)
            print("[info] Loaded weights with strict=False")
        except Exception as e2:
            print(f"[error] Failed to load state dict: {e2}")
            sys.exit(1)

    model.eval().cpu()

    # build dummy input
    C = args.channels
    H = W = args.image_size
    dummy = torch.randn(1, C, H, W, dtype=torch.float32)

    # dynamic axes config
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch"}
        }

    # export
    print(f"[info] Exporting ONNX -> {args.out} (opset={args.opset})")
    try:
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
    except Exception as e:
        print(f"[error] torch.onnx.export failed: {e}")
        sys.exit(1)

    # validate ONNX model
    try:
        import onnx
        onnx_model = onnx.load(args.out)
        onnx.checker.check_model(onnx_model)
        print("[info] ONNX model check passed.")
    except Exception as e:
        print(f"[warn] ONNX checker warning/error: {e}")

    # optional simplify
    if args.simplify:
        try:
            import onnxsim
            print("[info] Running onnx-simplifier...")
            simplified, ok = onnxsim.simplify(args.out, dynamic_input_shape=args.dynamic)
            if ok:
                onnx.save(simplified, args.out)
                print("[info] Simplified ONNX written.")
            else:
                print("[warn] onnx-simplifier returned ok=False; original ONNX kept.")
        except Exception as e:
            print(f"[warn] onnx-simplifier not available or failed: {e}")

    # quick runtime test (CPU)
    try:
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        sess = ort.InferenceSession(args.out, providers=providers)
        inp_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]
        print(f"[info] ORT session created. Inputs: {sess.get_inputs()[0].name} {sess.get_inputs()[0].shape}, Outputs: {out_names}")
        sample = dummy.numpy()
        res = sess.run(None, {inp_name: sample})
        print(f"[info] ONNX runtime test OK. Output shape: {res[0].shape}")
    except Exception as e:
        print(f"[warn] ONNX Runtime test skipped or failed: {e}")

    print("[done] Export complete. ONNX saved to:", os.path.abspath(args.out))

if __name__ == "__main__":
    main()
