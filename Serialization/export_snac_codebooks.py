"""
export_snac_codebooks.py — extract RVQ codebook weights from SNAC ONNX models.

SNAC codebooks are embedded inside quantize_encodings.onnx (as Embedding weights).
This script extracts them and saves in the same format as EnCodec:
  codebooks.json  — {"n_q": N, "vocab_size": S, "dim": D}
  codebooks.bin   — flat float32 array [level][code][dim]

Usage:
  python serialization/export_snac_codebooks.py [--models 24khz 32khz 44khz]
"""
import argparse
import json
import os
import onnx
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "snac_onnx_exports")

# Expected codebook shape per model (from model.json)
META = {
    "24khz": {"n_q": 3, "vocab_size": 4096, "dim": 8},
    "32khz": {"n_q": 4, "vocab_size": 4096, "dim": 8},
    "44khz": {"n_q": 4, "vocab_size": 4096, "dim": 8},
}


def extract_codebook_weights(onnx_path, vocab_size, dim, n_q):
    model = onnx.load(onnx_path)
    found = []
    for init in model.graph.initializer:
        # Look for Embedding weight tensors matching [vocab_size, dim]
        if list(init.dims) == [vocab_size, dim]:
            arr = onnx.numpy_helper.to_array(init)
            found.append(arr)
    if len(found) != n_q:
        print(f"  WARNING: found {len(found)} codebook tensors (expected {n_q})")
    return found


def export_model(name):
    meta = META[name]
    n_q = meta["n_q"]
    vocab_size = meta["vocab_size"]
    dim = meta["dim"]

    onnx_path = os.path.join(_ROOT, name, "quantize_encodings.onnx")
    if not os.path.exists(onnx_path):
        print(f"  SKIP: {onnx_path} not found")
        return

    weights = extract_codebook_weights(onnx_path, vocab_size, dim, n_q)

    if len(weights) != n_q:
        print(f"  SKIP ({name}): expected {n_q} codebooks, found {len(weights)}")
        return

    # Save as flat float32 array: [level][code][dim]
    flat = np.concatenate([w.flatten() for w in weights]).astype(np.float32)
    assert flat.shape[0] == n_q * vocab_size * dim, f"size mismatch {flat.shape[0]}"

    out_dir = os.path.join(_ROOT, name)
    bin_path = os.path.join(out_dir, "codebooks.bin")
    json_path = os.path.join(out_dir, "codebooks.json")

    flat.tofile(bin_path)
    with open(json_path, "w") as f:
        json.dump({"n_q": n_q, "vocab_size": vocab_size, "dim": dim}, f)

    sizes = [f"{w.shape}" for w in weights]
    print(f"  {name}: {len(weights)} codebooks, {sizes}, {len(flat)} floats → codebooks.bin/json")


def main():
    parser = argparse.ArgumentParser(description="Extract SNAC codebooks from ONNX")
    parser.add_argument("--models", nargs="+", choices=list(META),
                        default=list(META), metavar="NAME")
    args = parser.parse_args()
    for name in args.models:
        export_model(name)
    print("Done.")


if __name__ == "__main__":
    main()
