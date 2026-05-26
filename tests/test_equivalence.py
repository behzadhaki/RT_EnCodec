"""
Numerical equivalence tests: encodec (original) vs rt_encodec (ONNX-compatible fork).

For every model / bandwidth / input-length combination the tests verify that:
  - encode()  produces identical codebook indices and scale factors
  - decode()  produces identical reconstructed audio tensors
  - forward() produces identical full encode-decode roundtrip tensors
  - the rt_encodec exporting_to_onnx flag does not alter numerical results
  - batch_size > 1 is handled identically by both packages

Input lengths are chosen to cover:
  - clean multiples of the overall encoder stride (8×5×4×2 = 320)
  - lengths where (length % per-layer-stride) != 0  →  exercises the
    fixed ceiling arithmetic in get_extra_padding_for_conv1d
  - inputs that cross the 48 kHz model's segment boundary (segment_length=48000)
    →  exercises the overlap-add path in decode()
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from encodec import EncodecModel as OrigModel      # unmodified upstream
from rt_encodec import EncodecModel as RTModel     # ONNX-compatible fork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pair(factory: str, bandwidth: float):
    """Return (orig, rt) with identical random weights and the same bandwidth."""
    orig = getattr(OrigModel, factory)(pretrained=False)
    rt   = getattr(RTModel,  factory)(pretrained=False)
    rt.load_state_dict(orig.state_dict())
    orig.set_target_bandwidth(bandwidth)
    rt.set_target_bandwidth(bandwidth)
    orig.eval()
    rt.eval()
    return orig, rt


def _audio(channels: int, n_samples: int, seed: int = 0) -> torch.Tensor:
    """Fixed-seed random tensor of shape [1, channels, n_samples]."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, channels, n_samples, generator=g)


def _assert_frames_equal(orig_frames, rt_frames, label: str = ""):
    prefix = f"[{label}] " if label else ""
    assert len(orig_frames) == len(rt_frames), (
        f"{prefix}frame count: orig={len(orig_frames)} rt={len(rt_frames)}"
    )
    for i, ((o_codes, o_scale), (r_codes, r_scale)) in enumerate(
        zip(orig_frames, rt_frames)
    ):
        assert o_codes.shape == r_codes.shape, (
            f"{prefix}frame {i} codes shape: {o_codes.shape} vs {r_codes.shape}"
        )
        assert torch.equal(o_codes, r_codes), (
            f"{prefix}frame {i} codes differ"
        )
        if o_scale is None:
            assert r_scale is None, f"{prefix}frame {i} scale: orig=None rt={r_scale}"
        else:
            assert r_scale is not None, (
                f"{prefix}frame {i} scale: orig={o_scale.shape} rt=None"
            )
            assert torch.equal(o_scale, r_scale), (
                f"{prefix}frame {i} scale differs"
            )


def _assert_audio_equal(orig_out: torch.Tensor, rt_out: torch.Tensor, label: str = ""):
    prefix = f"[{label}] " if label else ""
    assert orig_out.shape == rt_out.shape, (
        f"{prefix}shape: {orig_out.shape} vs {rt_out.shape}"
    )
    assert torch.equal(orig_out, rt_out), (
        f"{prefix}tensors differ — "
        f"max |Δ| = {(orig_out - rt_out).abs().max():.6e}"
    )


# ---------------------------------------------------------------------------
# Parameter sets
# ---------------------------------------------------------------------------

BW_24 = [1.5, 3.0, 6.0, 12.0, 24.0]
BW_48 = [3.0, 6.0, 12.0, 24.0]

# 24 kHz — overall encoder stride = 8×5×4×2 = 320
# Mix of: clean multiples of 320, lengths where length % 8 != 0,
# lengths where length % 5 != 0 after the first strided conv, etc.
LENGTHS_24 = [
    pytest.param(24000, id="24000-1s-clean"),
    pytest.param(48000, id="48000-2s-clean"),
    pytest.param(8000,  id="8000-short"),
    pytest.param(320,   id="320-one-frame"),
    pytest.param(24001, id="24001-not-div-8"),       # length % 8 = 1
    pytest.param(24005, id="24005-not-div-5"),       # length % 5 = 0 but tests mix
    pytest.param(24007, id="24007-not-div-8-or-4"),  # length % 8 = 7
    pytest.param(24319, id="24319-not-div-320"),     # length % 320 = 319
    pytest.param(16384, id="16384-pow2"),
    pytest.param(12345, id="12345-arbitrary"),
    pytest.param(7777,  id="7777-prime-ish"),
]

# 48 kHz — segment_length = 48000, segment_stride ≈ 47520 (1 % overlap)
# Include lengths that cross the segment boundary to test overlap-add.
LENGTHS_48 = [
    pytest.param(48000,  id="48000-1s-clean"),
    pytest.param(96000,  id="96000-2s-2-segments"),
    pytest.param(24000,  id="24000-half-s"),
    pytest.param(47999,  id="47999-just-under-seg"),
    pytest.param(48001,  id="48001-crosses-seg"),
    pytest.param(96001,  id="96001-2s+1-crosses-seg"),
    pytest.param(48319,  id="48319-not-div-320"),
    pytest.param(12345,  id="12345-arbitrary"),
    pytest.param(144000, id="144000-3s-3-segments"),
]


# ---------------------------------------------------------------------------
# encode() equivalence
# ---------------------------------------------------------------------------

class TestEncode:
    """Codebook indices and scale factors must be bit-exact."""

    @pytest.mark.parametrize("n_samples", LENGTHS_24)
    @pytest.mark.parametrize("bandwidth", BW_24)
    def test_24khz(self, bandwidth, n_samples):
        orig, rt = _make_pair("encodec_model_24khz", bandwidth)
        x = _audio(1, n_samples)
        with torch.no_grad():
            rt_frames, _ = rt.encode(x)
            _assert_frames_equal(orig.encode(x), rt_frames)

    @pytest.mark.parametrize("n_samples", LENGTHS_48)
    @pytest.mark.parametrize("bandwidth", BW_48)
    def test_48khz(self, bandwidth, n_samples):
        orig, rt = _make_pair("encodec_model_48khz", bandwidth)
        x = _audio(2, n_samples)
        with torch.no_grad():
            rt_frames, _ = rt.encode(x)
            _assert_frames_equal(orig.encode(x), rt_frames)


# ---------------------------------------------------------------------------
# decode() equivalence
# ---------------------------------------------------------------------------

class TestDecode:
    """Decoded audio must be bit-exact given identical encoded frames."""

    @pytest.mark.parametrize("n_samples", LENGTHS_24)
    @pytest.mark.parametrize("bandwidth", BW_24)
    def test_24khz(self, bandwidth, n_samples):
        orig, rt = _make_pair("encodec_model_24khz", bandwidth)
        x = _audio(1, n_samples)
        with torch.no_grad():
            frames = orig.encode(x)
            rt_audio, _ = rt.decode(frames)
            _assert_audio_equal(orig.decode(frames), rt_audio)

    @pytest.mark.parametrize("n_samples", LENGTHS_48)
    @pytest.mark.parametrize("bandwidth", BW_48)
    def test_48khz(self, bandwidth, n_samples):
        orig, rt = _make_pair("encodec_model_48khz", bandwidth)
        x = _audio(2, n_samples)
        with torch.no_grad():
            frames = orig.encode(x)
            rt_audio, _ = rt.decode(frames)
            _assert_audio_equal(orig.decode(frames), rt_audio)


# ---------------------------------------------------------------------------
# forward() equivalence  (full encode + decode roundtrip)
# ---------------------------------------------------------------------------

class TestForward:
    """End-to-end forward pass must be bit-exact."""

    @pytest.mark.parametrize("n_samples", LENGTHS_24)
    @pytest.mark.parametrize("bandwidth", BW_24)
    def test_24khz(self, bandwidth, n_samples):
        orig, rt = _make_pair("encodec_model_24khz", bandwidth)
        x = _audio(1, n_samples)
        with torch.no_grad():
            _assert_audio_equal(orig(x), rt(x))

    @pytest.mark.parametrize("n_samples", LENGTHS_48)
    @pytest.mark.parametrize("bandwidth", BW_48)
    def test_48khz(self, bandwidth, n_samples):
        orig, rt = _make_pair("encodec_model_48khz", bandwidth)
        x = _audio(2, n_samples)
        with torch.no_grad():
            _assert_audio_equal(orig(x), rt(x))


# ---------------------------------------------------------------------------
# exporting_to_onnx flag is numerically neutral
# ---------------------------------------------------------------------------

class TestExportingFlagNeutral:
    """
    rt_encodec with exporting_to_onnx=True must produce the same output as
    with the flag False.  The flag only gates Python-level asserts; it must
    not alter any computation.
    """

    @pytest.mark.parametrize("bandwidth", BW_24)
    @pytest.mark.parametrize("n_samples", [24000, 24001, 48000, 24319])
    def test_24khz(self, bandwidth, n_samples):
        off = RTModel.encodec_model_24khz(pretrained=False)
        on  = RTModel.encodec_model_24khz(pretrained=False)
        on.load_state_dict(off.state_dict())
        off.set_target_bandwidth(bandwidth); off.eval()
        on.set_target_bandwidth(bandwidth);  on.eval()
        on.exporting_to_onnx = True
        x = _audio(1, n_samples)
        with torch.no_grad():
            _assert_audio_equal(off(x), on(x), label="flag-neutral-24khz")

    @pytest.mark.parametrize("bandwidth", BW_48)
    @pytest.mark.parametrize("n_samples", [48000, 48001, 96000, 48319])
    def test_48khz(self, bandwidth, n_samples):
        off = RTModel.encodec_model_48khz(pretrained=False)
        on  = RTModel.encodec_model_48khz(pretrained=False)
        on.load_state_dict(off.state_dict())
        off.set_target_bandwidth(bandwidth); off.eval()
        on.set_target_bandwidth(bandwidth);  on.eval()
        on.exporting_to_onnx = True
        x = _audio(2, n_samples)
        with torch.no_grad():
            _assert_audio_equal(off(x), on(x), label="flag-neutral-48khz")


# ---------------------------------------------------------------------------
# Batch size > 1
# ---------------------------------------------------------------------------

class TestBatchEquivalence:
    """
    With batch_size=2, orig and rt must still agree exactly.
    This exercises all batch-dependent paths (normalization, padding, LSTM).
    """

    @pytest.mark.parametrize("bandwidth", BW_24)
    @pytest.mark.parametrize("n_samples", [24000, 24001, 24319])
    def test_24khz(self, bandwidth, n_samples):
        orig, rt = _make_pair("encodec_model_24khz", bandwidth)
        x = torch.cat([_audio(1, n_samples, seed=0), _audio(1, n_samples, seed=1)], dim=0)
        with torch.no_grad():
            _assert_audio_equal(orig(x), rt(x), label="batch2-24khz")

    @pytest.mark.parametrize("bandwidth", BW_48)
    @pytest.mark.parametrize("n_samples", [48000, 48001, 96000])
    def test_48khz(self, bandwidth, n_samples):
        orig, rt = _make_pair("encodec_model_48khz", bandwidth)
        x = torch.cat([_audio(2, n_samples, seed=0), _audio(2, n_samples, seed=1)], dim=0)
        with torch.no_grad():
            _assert_audio_equal(orig(x), rt(x), label="batch2-48khz")


# ---------------------------------------------------------------------------
# Padding correctness — targeted regression for the ceiling-arithmetic fix
# ---------------------------------------------------------------------------

class TestPaddingRegression:
    """
    Directly verify that inputs whose length is not divisible by the encoder
    stride produce the same output from both packages.  These are the exact
    cases where the wrong floor+1 formula would have silently given a shorter
    padded tensor, potentially dropping the last frame or producing a shape
    mismatch.
    """

    # For 24 kHz the per-layer strides are 8, 5, 4, 2.
    # lengths chosen so (length % stride) != 0 at the first encoder layer.
    @pytest.mark.parametrize("n_samples", [
        8001,   # % 8 = 1
        8003,   # % 8 = 3
        8007,   # % 8 = 7
        24001,
        24007,
        24319,  # % 320 = 319  (non-div at every stride level)
        40001,
    ])
    def test_24khz_non_divisible_lengths(self, n_samples):
        orig, rt = _make_pair("encodec_model_24khz", 6.0)
        x = _audio(1, n_samples)
        with torch.no_grad():
            o_frames = orig.encode(x)
            r_frames, _ = rt.encode(x)
            rt_audio, _ = rt.decode(r_frames)
        _assert_frames_equal(o_frames, r_frames, label=f"padding-regression-{n_samples}")
        _assert_audio_equal(orig.decode(o_frames), rt_audio,
                            label=f"padding-regression-decode-{n_samples}")

    @pytest.mark.parametrize("n_samples", [
        48001,
        48007,
        48319,
        96001,
        96319,
    ])
    def test_48khz_non_divisible_lengths(self, n_samples):
        orig, rt = _make_pair("encodec_model_48khz", 6.0)
        x = _audio(2, n_samples)
        with torch.no_grad():
            o_frames = orig.encode(x)
            r_frames, _ = rt.encode(x)
            rt_audio, _ = rt.decode(r_frames)
        _assert_frames_equal(o_frames, r_frames, label=f"padding-regression-{n_samples}")
        _assert_audio_equal(orig.decode(o_frames), rt_audio,
                            label=f"padding-regression-decode-{n_samples}")
