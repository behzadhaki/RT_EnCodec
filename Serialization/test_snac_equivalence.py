"""
test_snac_equivalence.py — verify each exported SNAC ONNX graph matches the
rt_snac PyTorch stage it was exported from.

Requires export_snac.py to have been run first:
    python serialization/export_snac.py --models 24khz 32khz 44khz
    pytest serialization/test_snac_equivalence.py -v

Tolerance: conv-only 24k matches to ~1e-5; the attention models (32k/44k)
drift to ~2e-4 from SDPA/rotary/LayerNorm float reordering, so floats are
checked at atol/rtol 1e-3. Codes stay bit-exact at all three rates.

Per stage:
  encode_audio_segment : z          allclose
  quantize_encodings   : codes_i    bit-exact (int64), all N levels
  decode_codes         : zq_i       allclose, all N levels
  decode_audio         : audio      allclose

Each stage is checked at TWO input lengths to confirm the dynamic axes hold
(preprocess is host-side, so the encode graph must generalise across T).
"""

import math
import os
import sys

import numpy as np
import onnxruntime as ort
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rt_snac import SNAC

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPORTS = os.path.join(_HERE, "snac_onnx_exports")
# Conv-only (24k) is ~1e-5. The attention models (32k/44k) drift to ~2e-4 from
# SDPA / rotary / LayerNorm float reordering — still numerically equivalent.
ATOL = 1e-3
RTOL = 1e-3


def _close(a, b):
    return np.allclose(a, b, atol=ATOL, rtol=RTOL)

MODELS = ["24khz", "32khz", "44khz"]
REPO = {"24khz": "hubertsiuzdak/snac_24khz",
        "32khz": "hubertsiuzdak/snac_32khz",
        "44khz": "hubertsiuzdak/snac_44khz"}


def _pad_to(model, audio):
    pad_to = int(model.hop_length) * (model.vq_strides[0] if model.attn_window_size is None
                                      else math.lcm(model.vq_strides[0], model.attn_window_size))
    T = audio.shape[-1]
    right = math.ceil(T / pad_to) * pad_to - T
    return torch.nn.functional.pad(audio, (0, right))


def _sess(name, stage):
    return ort.InferenceSession(os.path.join(_EXPORTS, name, f"{stage}.onnx"),
                                providers=["CPUExecutionProvider"])


def _np(t):
    return t.detach().cpu().numpy()


@pytest.fixture(scope="module", params=MODELS)
def ctx(request):
    name = request.param
    out_dir = os.path.join(_EXPORTS, name)
    if not os.path.isdir(out_dir):
        pytest.skip(f"{out_dir} missing — run export_snac.py --models {name}")
    model = SNAC.from_pretrained(REPO[name]).eval()
    model.set_noise(False)
    return name, model


@pytest.mark.parametrize("T", [12000, 30000])   # two lengths → distinct T_b for all
                                                # models (and distinct attention
                                                # window counts for 32k/44k)
def test_pipeline_equivalence(ctx, T):
    name, model = ctx
    N = model.n_codebooks
    torch.manual_seed(0)
    audio = _pad_to(model, torch.randn(1, 1, T))

    with torch.no_grad():
        z_pt = model.encoder(audio)
        codes_pt = model.quantize_encodings(z_pt)
        zqs_pt = model.decode_codes(codes_pt)
        audio_pt = model.decode_audio(sum(zqs_pt))

    # 1. encode_audio_segment
    z_ort = _sess(name, "encode_audio_segment").run(None, {"audio": _np(audio)})[0]
    assert _close(z_ort, _np(z_pt)), \
        f"encode z max err {np.abs(z_ort - _np(z_pt)).max()}"

    # 2. quantize_encodings — N int64 outputs, bit-exact
    code_out = _sess(name, "quantize_encodings").run(None, {"z": z_ort})
    assert len(code_out) == N
    for i in range(N):
        assert np.array_equal(code_out[i], _np(codes_pt[i])), f"codes_{i} mismatch"

    # 3. decode_codes — N code inputs → N per-level zq outputs
    feed = {f"codes_{i}": _np(codes_pt[i]).astype(np.int64) for i in range(N)}
    zq_out = _sess(name, "decode_codes").run(None, feed)
    assert len(zq_out) == N
    for i in range(N):
        assert _close(zq_out[i], _np(zqs_pt[i])), \
            f"zq_{i} max err {np.abs(zq_out[i] - _np(zqs_pt[i])).max()}"

    # 4. decode_audio — sum of zq → audio
    z_q = np.sum(zq_out, axis=0).astype(np.float32)
    audio_out = _sess(name, "decode_audio").run(None, {"z_q": z_q})[0]
    assert _close(audio_out, _np(audio_pt)), \
        f"decode audio max err {np.abs(audio_out - _np(audio_pt)).max()}"


def test_level_subset_decode(ctx):
    """The per-level use case: decoding a subset of levels through the ONNX
    graphs must equal doing it in PyTorch."""
    name, model = ctx
    N = model.n_codebooks
    torch.manual_seed(1)
    audio = _pad_to(model, torch.randn(1, 1, 9000))

    with torch.no_grad():
        z_pt = model.encoder(audio)
        codes_pt = model.quantize_encodings(z_pt)

    feed = {f"codes_{i}": _np(codes_pt[i]).astype(np.int64) for i in range(N)}
    zq_out = _sess(name, "decode_codes").run(None, feed)
    dec = _sess(name, "decode_audio")

    for keep in ([0], list(range(N))):
        z_q = np.sum([zq_out[i] for i in keep], axis=0).astype(np.float32)
        a_ort = dec.run(None, {"z_q": z_q})[0]
        with torch.no_grad():
            zqs_pt = model.decode_codes(codes_pt)
            a_pt = model.decode_audio(sum(zqs_pt[i] for i in keep))
        assert _close(a_ort, _np(a_pt)), \
            f"levels {keep} max err {np.abs(a_ort - _np(a_pt)).max()}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
