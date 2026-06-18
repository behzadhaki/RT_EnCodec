"""
export_snac.py — export rt_snac to ONNX as four independent graphs per model.

Each graph corresponds to one stage in the pipeline (mirrors export_onnx.py):

  encode_audio_segment.onnx   audio        → z (continuous latent)
  quantize_encodings.onnx     z            → codes_0 … codes_{N-1}   (int64)
  decode_codes.onnx           codes_0 …    → zq_0 … zq_{N-1}         (per level)
  decode_audio.onnx           z_q (summed) → audio

Unlike EnCodec, SNAC is fully convolutional (no LSTM state) and has no
bandwidth axis — instead it emits N multi-scale codebook levels at different
rates. The codes stay a LIST of per-level tensors end to end (never flattened),
and decode_codes returns the per-level embedding contributions *unsummed* so a
caller can audition any subset of levels:  z_q = sum(zq_i for i in keep).

N = number of codebooks (read from each model's config):
  snac_24khz: 3   (vq_strides [4, 2, 1])
  snac_32khz: 4   (vq_strides [8, 4, 2, 1])
  snac_44khz: 4   (vq_strides [8, 4, 2, 1])

The decoder's stochastic NoiseBlock is disabled (set_noise(False)) so the
graphs are deterministic; the fidelity cost is ~3e-5 L1 (measured).

Outputs written to serialization/snac_onnx_exports/<model>/

Run from repo root:
  python serialization/export_snac.py [--models 24khz 32khz 44khz]
"""

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import onnx

warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*")
warnings.filterwarnings("ignore", message=".*Constant folding.*")
warnings.filterwarnings("ignore", message=".*torchaudio.*torchcodec.*")

from rt_snac import SNAC

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPORTS_ROOT = os.path.join(_HERE, "snac_onnx_exports")
_OPSET = 17

MODELS = {
    "24khz": "hubertsiuzdak/snac_24khz",
    "32khz": "hubertsiuzdak/snac_32khz",
    "44khz": "hubertsiuzdak/snac_44khz",
}


# ---------------------------------------------------------------------------
# Stage wrappers — generic over N codebooks (no LSTM state, no scale)
# ---------------------------------------------------------------------------

class EncodeAudioSegment(nn.Module):
    """Stage 1.  audio [B, 1, T] → z [B, latent_dim, T_b].

    NOTE: preprocess (right-pad to a multiple of preprocess_pad) is the
    caller's responsibility — kept OUT of the graph so T stays dynamic.
    Baking it would constant-fold the pad amount for one specific length.
    """
    def __init__(self, model: SNAC):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, audio: torch.Tensor):
        return self.encoder(audio)


class QuantizeEncodings(nn.Module):
    """Stage 2.  z [B, latent_dim, T_b] → N int64 code tensors,
    one per level, codes_i shape [B, T_b // stride_i]."""
    def __init__(self, model: SNAC):
        super().__init__()
        self.model = model

    def forward(self, z: torch.Tensor):
        return tuple(self.model.quantize_encodings(z))


class DecodeCodes(nn.Module):
    """Stage 3.  N code tensors → N per-level zq_i [B, latent_dim, T_b]
    (each already upsampled to the base grid, NOT summed)."""
    def __init__(self, model: SNAC):
        super().__init__()
        self.model = model

    def forward(self, *codes: torch.Tensor):
        return tuple(self.model.decode_codes(list(codes)))


class DecodeAudio(nn.Module):
    """Stage 4.  summed z_q [B, latent_dim, T_b] → audio [B, 1, T']."""
    def __init__(self, model: SNAC):
        super().__init__()
        self.model = model

    def forward(self, z_q: torch.Tensor):
        return self.model.decode_audio(z_q)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _export(wrapper, dummy_inputs, path, input_names, output_names, dynamic_axes):
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        path,
        opset_version=_OPSET,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    onnx.checker.check_model(onnx.load(path))
    print(f"  saved & verified → {os.path.relpath(path, _HERE)}")


def _write_meta(model: SNAC, name: str, out_dir: str) -> None:
    """Sidecar describing the multi-scale layout (for the web/runtime side)."""
    strides = list(model.vq_strides)
    meta = {
        "model": name,
        "sampling_rate": model.sampling_rate,
        "hop_length": int(model.hop_length),
        "frame_rate": model.sampling_rate / float(model.hop_length),
        "latent_dim": model.latent_dim,
        "n_codebooks": model.n_codebooks,
        "vq_strides": strides,
        "codebook_size": model.codebook_size,
        "codebook_dim": model.codebook_dim,
        "attn_window_size": model.attn_window_size,
        "preprocess_pad": int(model.hop_length * (strides[0] if model.attn_window_size is None
                                                  else _lcm(strides[0], model.attn_window_size))),
        "noise": False,
    }
    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  layout → {os.path.relpath(os.path.join(out_dir, 'model.json'), _HERE)}  "
          f"(N={meta['n_codebooks']}, strides={strides}, {meta['frame_rate']:.3f} Hz)")


def _lcm(a, b):
    import math
    return math.lcm(a, b)


# ---------------------------------------------------------------------------
# Per-model export
# ---------------------------------------------------------------------------

def export_model(name: str) -> None:
    repo_id = MODELS[name]
    out_dir = os.path.join(_EXPORTS_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}\n{name}  ({repo_id})\n{'='*60}")
    model = SNAC.from_pretrained(repo_id).eval()
    model.set_noise(False)   # deterministic, ONNX-exportable decoder

    N = model.n_codebooks
    code_names = [f"codes_{i}" for i in range(N)]
    zq_names = [f"zq_{i}" for i in range(N)]

    # Dummy input: already preprocessed (length a multiple of preprocess_pad),
    # since the host pads before the encode graph. T_b stays small for fast export.
    B = 1
    pad_to = int(model.hop_length) * (model.vq_strides[0] if model.attn_window_size is None
                                      else _lcm(model.vq_strides[0], model.attn_window_size))
    T = pad_to * 3
    audio = torch.randn(B, 1, T)

    with torch.no_grad():
        z = model.encode_audio_segment(audio)
        codes = model.quantize_encodings(z)
        zqs = model.decode_codes(codes)
        z_q = sum(zqs)
    print(f"  shapes: z {tuple(z.shape)}  "
          f"codes {[tuple(c.shape) for c in codes]}  "
          f"zq {tuple(zqs[0].shape)}")

    # 1. encode_audio_segment
    _export(EncodeAudioSegment(model), (audio,),
            os.path.join(out_dir, "encode_audio_segment.onnx"),
            input_names=["audio"], output_names=["z"],
            dynamic_axes={"audio": {0: "B", 2: "T"}, "z": {0: "B", 2: "T_b"}})

    # 2. quantize_encodings  (N code outputs at different rates)
    _export(QuantizeEncodings(model), (z,),
            os.path.join(out_dir, "quantize_encodings.onnx"),
            input_names=["z"], output_names=code_names,
            dynamic_axes={"z": {0: "B", 2: "T_b"},
                          **{code_names[i]: {0: "B", 1: f"Tf{i}"} for i in range(N)}})

    # 3. decode_codes  (N code inputs → N per-level zq outputs)
    _export(DecodeCodes(model), tuple(codes),
            os.path.join(out_dir, "decode_codes.onnx"),
            input_names=code_names, output_names=zq_names,
            dynamic_axes={**{code_names[i]: {0: "B", 1: f"Tf{i}"} for i in range(N)},
                          **{zq_names[i]: {0: "B", 2: "T_b"} for i in range(N)}})

    # 4. decode_audio  (summed embedding → audio)
    _export(DecodeAudio(model), (z_q,),
            os.path.join(out_dir, "decode_audio.onnx"),
            input_names=["z_q"], output_names=["audio"],
            dynamic_axes={"z_q": {0: "B", 2: "T_b"}, "audio": {0: "B", 2: "T"}})

    _write_meta(model, name, out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export rt_snac pipeline stages to ONNX. "
                    "Default: 24khz only (32khz/44khz add local attention).")
    parser.add_argument("--models", nargs="+", choices=list(MODELS),
                        default=["24khz"], metavar="NAME",
                        help=f"which model(s) to export. choices: {list(MODELS)}")
    args = parser.parse_args()

    print(f"Output root: {_EXPORTS_ROOT}")
    print(f"Models: {args.models}")
    for name in args.models:
        export_model(name)
    print(f"\nDone. Exports written to {_EXPORTS_ROOT}/")


if __name__ == "__main__":
    main()
