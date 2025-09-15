from safetensors import safe_open
with safe_open("B-Eye-O-Marker_CNN-V2.safetensors", framework="pt", device="cpu") as f:
    md = f.metadata()
print("metadata keys:", md.keys())
print(md.get("classes", md.get("cfg")))
