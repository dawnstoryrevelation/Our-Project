# compare_preproc_pt_onnx.py
import sys, os, numpy as np, torch, onnxruntime as ort
from PIL import Image
from safetensors.torch import load_file
import torch.nn as nn, math

# --------- Minimal BiNLOP3 + Model class (match your training) ----------
class BiNLOP3(nn.Module):
    def __init__(self, channels:int, gamma_min:float=0.5, per_channel=True,
                 init_g1=0.75, init_g2=0.6, init_k1=1.0, init_k2=6.0):
        super().__init__()
        self.channels=channels; self.gamma_min=float(gamma_min)
        shape=(channels,) if per_channel else (1,)
        def inv_sig(y):
            y = max(min(y, 1-1e-6), 1e-6)
            return math.log(y/(1-y))
        g1_target=(init_g1-self.gamma_min)/(1-self.gamma_min)
        g2_target=(init_g2-self.gamma_min)/(init_g1-self.gamma_min)
        s1_hat_init=math.log(max(init_k1,1e-4))
        d_hat_init=math.log(max(init_k2-init_k1,1e-4))
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
        return g1,g2,k1,k2
    def forward(self, x):
        g1,g2,k1,k2 = self._params()
        if k1.dim()==1:
            if x.dim()==4:
                g1=g1.view(1,-1,1,1); g2=g2.view(1,-1,1,1); k1=k1.view(1,-1,1,1); k2=k2.view(1,-1,1,1)
        return g2 * x + (1.0 - g1) * x.clamp(-k1, k1) + (g1 - g2) * x.clamp(-k2, k2)

class ConvBNAct(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1,act_layer=BiNLOP3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,3,stride=stride,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1)
        self.act = act_layer(out_ch)
    def forward(self,x): return self.act(self.bn(self.conv(x)))

class BasicBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1,act_layer=BiNLOP3):
        super().__init__()
        self.conv1=ConvBNAct(in_ch,out_ch,stride=stride,act_layer=act_layer)
        self.conv2=nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1)
        self.act=act_layer(out_ch)
        self.short = nn.Identity() if (in_ch==out_ch and stride==1) else nn.Sequential(
            nn.Conv2d(in_ch,out_ch,1,stride=stride,bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1)
        )
    def forward(self,x):
        s = self.short(x)
        x = self.bn2(self.conv1(x))
        x = self.act(x + s)
        return x

class B_Eye_O_Marker_CNN_V2(nn.Module):
    def __init__(self, num_classes=9, chs=(96,192,384,640), in_ch=3):
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
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(chs[3], num_classes))
    def forward(self,x):
        x=self.stem(x); x=self.stage1(x); x=self.stage2(x); x=self.stage3(x); x=self.stage4(x); return self.head(x)

# --------- helpers ----------
def preprocess_resize(img, size=384):
    img = img.convert("RGB").resize((size,size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)/255.0
    arr = arr.transpose(2,0,1)  # CHW
    return arr[None,:,:,:].astype(np.float32)

def preprocess_letterbox(img, size=384):
    # center letterbox with black pad (mimic old UI)
    w,h = img.size
    scale = min(size/w, size/h)
    nw, nh = int(w*scale), int(h*scale)
    img_r = img.convert("RGB").resize((nw,nh), Image.BILINEAR)
    new = Image.new("RGB", (size,size), (0,0,0))
    new.paste(img_r, ((size-nw)//2, (size-nh)//2))
    arr = np.array(new).astype(np.float32)/255.0
    arr = arr.transpose(2,0,1)
    return arr[None,:,:,:].astype(np.float32)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python compare_preproc_pt_onnx.py <image.jpg>")
        sys.exit(1)
    img_path = sys.argv[1]
    img = Image.open(img_path)

    # build model and load safetensors
    model = B_Eye_O_Marker_CNN_V2(num_classes=9, chs=(96,192,384,640), in_ch=3)
    sd = load_file("B-Eye-O-Marker_CNN-V2.safetensors", device="cpu")
    # try strict load
    model.load_state_dict(sd, strict=False)
    model.eval()

    # ONNX session
    sess = ort.InferenceSession("B-Eye-O-Marker_CNN-V2.onnx", providers=["CPUExecutionProvider"])

    for name, preproc in [("resize", preprocess_resize), ("letterbox", preprocess_letterbox)]:
        x = preproc(img)
        print("=== Preproc:", name, "input shape:", x.shape)
        # PyTorch
        xt = torch.from_numpy(x)
        with torch.no_grad():
            pt_logits = model(xt).cpu().numpy().flatten()
        pt_probs = 1/(1+np.exp(-pt_logits))  # sigmoid (multi-label) — adjust if CE trained
        # ONNX
        onnx_inp = {sess.get_inputs()[0].name: x}
        outs = sess.run(None, onnx_inp)
        onnx_logits = np.array(outs[0]).flatten()
        onnx_probs = 1/(1+np.exp(-onnx_logits))
        print("PyTorch logits:", np.round(pt_logits,4))
        print("ONNX   logits:", np.round(onnx_logits,4))
        print("Diff logits (onnx - pt):", np.round(onnx_logits - pt_logits,6))
        print("PyTorch probs:", np.round(pt_probs,4))
        print("ONNX probs   :", np.round(onnx_probs,4))
        print()
