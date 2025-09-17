# diagnose_safetensors.py
import sys, os, hashlib, traceback

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def try_safe_open(path):
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            md = f.metadata() or {}
            print("Success: read metadata keys:", list(md.keys()))
            return True
    except Exception as e:
        print("safetensors safe_open failed:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_safetensors.py <file.safetensors>")
        sys.exit(1)
    p = sys.argv[1]
    if not os.path.exists(p):
        print("File not found:", p); sys.exit(2)
    st = os.stat(p)
    print("File:", p)
    print("Size (bytes):", st.st_size)
    print("SHA256:", sha256_of(p))
    ok = try_safe_open(p)
    if not ok:
        print("\nLikely causes: truncated file, incomplete download, OneDrive placeholder, or file corruption.")
        print("If this came from Hugging Face or another remote store, re-download the file.")
