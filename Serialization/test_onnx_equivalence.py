"""
test_onnx_equivalence.py

Verify that each of the four exported ONNX graphs produces outputs numerically
identical to the corresponding rt_encodec PyTorch stage.

What is checked
---------------
  encode_audio_segment : embeddings allclose, LSTM state allclose
  quantize_encodings   : codes bit-exact (int64)
  decode_codes         : embeddings allclose
  decode_audio         : audio allclose, LSTM state allclose
  Stateful calls       : state returned from call 1 fed into call 2 matches PT

ONNX files are expected at serialization/onnx_exports/{24k,48k}/{bw}kbps/.
Run export_onnx.py first.

Run:
  pytest serialization/test_onnx_equivalence.py -v
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import onnxruntime as ort
from rt_encodec import EncodecModel
from export_onnx import (
    EncodeAudioSegment24, QuantizeEncodings24, DecodeCodes24, DecodeAudio24,
    EncodeAudioSegment48, QuantizeEncodings48, DecodeCodes48, DecodeAudio48,
    _zero_state, LSTM_LAYERS, LSTM_HIDDEN,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
BW = 6.0
_EXPORTS_ROOT = os.path.join(_HERE, "encodec_onnx_exports")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onnx(model_hz: str, stage: str):
    path = os.path.join(_EXPORTS_ROOT, model_hz, f"{BW:g}kbps", f"{stage}.onnx")
    if not os.path.exists(path):
        pytest.skip(f"{path} not found — run export_onnx.py first")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().numpy()


def _assert_exact(pt: torch.Tensor, ort_out: np.ndarray, name: str):
    assert np.array_equal(_np(pt), ort_out), (
        f"{name}: outputs differ — max |Δ| = {np.abs(_np(pt) - ort_out).max():.3e}"
    )


def _assert_close(pt: torch.Tensor, ort_out: np.ndarray, name: str, atol=5e-4):
    assert np.allclose(_np(pt), ort_out, atol=atol, rtol=1e-4), (
        f"{name}: outputs differ — max |Δ| = {np.abs(_np(pt) - ort_out).max():.3e}"
    )


def _assert_state_close(pt: torch.Tensor, ort_out: np.ndarray, name: str):
    assert np.allclose(_np(pt), ort_out, atol=5e-4, rtol=1e-2), (
        f"{name}: state differs — max |Δ| = {np.abs(_np(pt) - ort_out).max():.3e}, "
        f"max rel = {(np.abs(_np(pt) - ort_out) / (np.abs(ort_out) + 1e-8)).max():.3e}"
    )


def _rand_state(batch: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(LSTM_LAYERS, batch, LSTM_HIDDEN, generator=g)
    c = torch.randn(LSTM_LAYERS, batch, LSTM_HIDDEN, generator=g)
    return h, c


def _audio(channels: int, n_samples: int, batch: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, channels, n_samples, generator=g)


def _model_24(pretrained=True):
    m = EncodecModel.encodec_model_24khz(pretrained=pretrained)
    m.set_target_bandwidth(BW)
    m.exporting_to_onnx = True
    m.eval()
    return m


def _model_48(pretrained=True):
    m = EncodecModel.encodec_model_48khz(pretrained=pretrained)
    m.set_target_bandwidth(BW)
    m.exporting_to_onnx = True
    m.eval()
    return m


# ---------------------------------------------------------------------------
# 24 kHz — encode_audio_segment
# ---------------------------------------------------------------------------

class TestEncodeAudioSegment24:

    @pytest.fixture(scope="class")
    def wrapper(self):
        return EncodeAudioSegment24(_model_24())

    @pytest.fixture(scope="class")
    def session(self):
        return _onnx("24k", "encode_audio_segment")

    @pytest.mark.parametrize("n_samples", [320, 3200, 24000, 24001])
    def test_emb_close_mono(self, wrapper, session, n_samples):
        audio = _audio(1, n_samples)
        h, c = _zero_state(1)
        with torch.no_grad():
            emb_pt, h_pt, c_pt = wrapper(audio, h, c)
        outs = session.run(None, {"audio": _np(audio), "h_in": _np(h), "c_in": _np(c)})
        _assert_close(emb_pt, outs[0], "emb")
        _assert_state_close(h_pt, outs[1], "h_out")
        _assert_state_close(c_pt, outs[2], "c_out")

    @pytest.mark.parametrize("n_samples", [320, 24000])
    def test_emb_close_stereo_batch(self, wrapper, session, n_samples):
        audio = _audio(1, n_samples, batch=2)
        h, c = _zero_state(2)
        with torch.no_grad():
            emb_pt, h_pt, c_pt = wrapper(audio, h, c)
        outs = session.run(None, {"audio": _np(audio), "h_in": _np(h), "c_in": _np(c)})
        _assert_close(emb_pt, outs[0], "emb")
        _assert_state_close(h_pt, outs[1], "h_out")
        _assert_state_close(c_pt, outs[2], "c_out")

    def test_stateful_second_call(self, wrapper, session):
        audio1 = _audio(1, 320, seed=0)
        audio2 = _audio(1, 320, seed=1)
        h0, c0 = _zero_state(1)
        with torch.no_grad():
            _, h1_pt, c1_pt = wrapper(audio1, h0, c0)
            emb2_pt, h2_pt, c2_pt = wrapper(audio2, h1_pt, c1_pt)
        outs1 = session.run(None, {"audio": _np(audio1), "h_in": _np(h0), "c_in": _np(c0)})
        outs2 = session.run(None, {"audio": _np(audio2), "h_in": outs1[1], "c_in": outs1[2]})
        _assert_close(emb2_pt, outs2[0], "emb (stateful)")
        _assert_state_close(h2_pt, outs2[1], "h_out (stateful)")
        _assert_state_close(c2_pt, outs2[2], "c_out (stateful)")


# ---------------------------------------------------------------------------
# 24 kHz — quantize_encodings
# ---------------------------------------------------------------------------

class TestQuantizeEncodings24:

    @pytest.fixture(scope="class")
    def wrapper(self):
        return QuantizeEncodings24(_model_24())

    @pytest.fixture(scope="class")
    def session(self):
        return _onnx("24k", "quantize_encodings")

    @pytest.mark.parametrize("t_frames", [1, 10, 75])
    def test_codes_exact(self, wrapper, session, t_frames):
        g = torch.Generator().manual_seed(t_frames)
        emb = torch.randn(1, 128, t_frames, generator=g)
        with torch.no_grad():
            codes_pt = wrapper(emb)
        outs = session.run(None, {"emb": _np(emb)})
        _assert_exact(codes_pt, outs[0], "codes")


# ---------------------------------------------------------------------------
# 24 kHz — decode_codes
# ---------------------------------------------------------------------------

class TestDecodeCodes24:

    @pytest.fixture(scope="class")
    def wrapper(self):
        return DecodeCodes24(_model_24())

    @pytest.fixture(scope="class")
    def session(self):
        return _onnx("24k", "decode_codes")

    @pytest.fixture(scope="class")
    def K(self):
        return int(1000 * BW // (_model_24(pretrained=False).frame_rate * 10))

    @pytest.mark.parametrize("t_frames", [1, 10, 75])
    def test_emb_close(self, wrapper, session, K, t_frames):
        codes = torch.randint(0, 1024, (1, K, t_frames))
        with torch.no_grad():
            emb_pt = wrapper(codes)
        outs = session.run(None, {"codes": _np(codes)})
        _assert_close(emb_pt, outs[0], "emb")


# ---------------------------------------------------------------------------
# 24 kHz — decode_audio
# ---------------------------------------------------------------------------

class TestDecodeAudio24:

    @pytest.fixture(scope="class")
    def wrapper(self):
        return DecodeAudio24(_model_24())

    @pytest.fixture(scope="class")
    def session(self):
        return _onnx("24k", "decode_audio")

    @pytest.mark.parametrize("t_frames", [1, 10, 75])
    def test_audio_close(self, wrapper, session, t_frames):
        g = torch.Generator().manual_seed(t_frames)
        emb = torch.randn(1, 128, t_frames, generator=g)
        h, c = _zero_state(1)
        with torch.no_grad():
            audio_pt, h_pt, c_pt = wrapper(emb, h, c)
        outs = session.run(None, {"emb": _np(emb), "h_in": _np(h), "c_in": _np(c)})
        _assert_close(audio_pt, outs[0], "audio")
        _assert_state_close(h_pt, outs[1], "h_out")
        _assert_state_close(c_pt, outs[2], "c_out")

    def test_stateful_second_call(self, wrapper, session):
        g0 = torch.Generator().manual_seed(0)
        g1 = torch.Generator().manual_seed(1)
        emb1 = torch.randn(1, 128, 1, generator=g0)
        emb2 = torch.randn(1, 128, 1, generator=g1)
        h0, c0 = _zero_state(1)
        with torch.no_grad():
            _, h1_pt, c1_pt = wrapper(emb1, h0, c0)
            audio2_pt, _, _ = wrapper(emb2, h1_pt, c1_pt)
        outs1 = session.run(None, {"emb": _np(emb1), "h_in": _np(h0), "c_in": _np(c0)})
        outs2 = session.run(None, {"emb": _np(emb2), "h_in": outs1[1], "c_in": outs1[2]})
        _assert_close(audio2_pt, outs2[0], "audio (stateful)")


# ---------------------------------------------------------------------------
# 48 kHz — encode_audio_segment
# ---------------------------------------------------------------------------

class TestEncodeAudioSegment48:
    """Minimum valid input: 1280 samples (= 4 × 320)."""

    @pytest.fixture(scope="class")
    def wrapper(self):
        return EncodeAudioSegment48(_model_48())

    @pytest.fixture(scope="class")
    def session(self):
        return _onnx("48k", "encode_audio_segment")

    @pytest.mark.parametrize("n_samples", [1280, 1281, 4800, 48000, 48001])
    def test_emb_close(self, wrapper, session, n_samples):
        audio = _audio(2, n_samples)
        h, c = _zero_state(1)
        with torch.no_grad():
            emb_pt, scale_pt, h_pt, c_pt = wrapper(audio, h, c)
        outs = session.run(None, {"audio": _np(audio), "h_in": _np(h), "c_in": _np(c)})
        _assert_close(emb_pt, outs[0], "emb")
        _assert_close(scale_pt, outs[1], "scale")
        _assert_state_close(h_pt, outs[2], "h_out")
        _assert_state_close(c_pt, outs[3], "c_out")

    def test_stateful_second_call(self, wrapper, session):
        audio1 = _audio(2, 1280, seed=0)
        audio2 = _audio(2, 1280, seed=1)
        h0, c0 = _zero_state(1)
        with torch.no_grad():
            _, _, h1_pt, c1_pt = wrapper(audio1, h0, c0)
            emb2_pt, _, _, _ = wrapper(audio2, h1_pt, c1_pt)
        outs1 = session.run(None, {"audio": _np(audio1), "h_in": _np(h0), "c_in": _np(c0)})
        outs2 = session.run(None, {"audio": _np(audio2), "h_in": outs1[2], "c_in": outs1[3]})
        _assert_close(emb2_pt, outs2[0], "emb (stateful)")


# ---------------------------------------------------------------------------
# 48 kHz — quantize_encodings
# ---------------------------------------------------------------------------

class TestQuantizeEncodings48:

    @pytest.fixture(scope="class")
    def wrapper(self):
        return QuantizeEncodings48(_model_48())

    @pytest.fixture(scope="class")
    def session(self):
        return _onnx("48k", "quantize_encodings")

    @pytest.mark.parametrize("t_frames", [1, 10, 150])
    def test_codes_exact(self, wrapper, session, t_frames):
        g = torch.Generator().manual_seed(t_frames)
        emb = torch.randn(1, 128, t_frames, generator=g)
        with torch.no_grad():
            codes_pt = wrapper(emb)
        outs = session.run(None, {"emb": _np(emb)})
        _assert_exact(codes_pt, outs[0], "codes")


# ---------------------------------------------------------------------------
# 48 kHz — decode_codes
# ---------------------------------------------------------------------------

class TestDecodeCodes48:

    @pytest.fixture(scope="class")
    def wrapper(self):
        return DecodeCodes48(_model_48())

    @pytest.fixture(scope="class")
    def session(self):
        return _onnx("48k", "decode_codes")

    @pytest.fixture(scope="class")
    def K(self):
        return int(1000 * BW // (_model_48(pretrained=False).frame_rate * 10))

    @pytest.mark.parametrize("t_frames", [1, 10, 150])
    def test_emb_close(self, wrapper, session, K, t_frames):
        codes = torch.randint(0, 1024, (1, K, t_frames))
        with torch.no_grad():
            emb_pt = wrapper(codes)
        outs = session.run(None, {"codes": _np(codes)})
        _assert_close(emb_pt, outs[0], "emb")


# ---------------------------------------------------------------------------
# 48 kHz — decode_audio
# ---------------------------------------------------------------------------

class TestDecodeAudio48:

    @pytest.fixture(scope="class")
    def wrapper(self):
        return DecodeAudio48(_model_48())

    @pytest.fixture(scope="class")
    def session(self):
        return _onnx("48k", "decode_audio")

    @pytest.mark.parametrize("t_frames", [1, 10, 150])
    def test_audio_close(self, wrapper, session, t_frames):
        g = torch.Generator().manual_seed(t_frames)
        emb = torch.randn(1, 128, t_frames, generator=g)
        scale = torch.ones(1, 1)
        h, c = _zero_state(1)
        with torch.no_grad():
            audio_pt, h_pt, c_pt = wrapper(emb, scale, h, c)
        outs = session.run(
            None, {"emb": _np(emb), "scale": _np(scale), "h_in": _np(h), "c_in": _np(c)}
        )
        _assert_close(audio_pt, outs[0], "audio")
        _assert_state_close(h_pt, outs[1], "h_out")
        _assert_state_close(c_pt, outs[2], "c_out")

    def test_stateful_second_call(self, wrapper, session):
        g0 = torch.Generator().manual_seed(0)
        g1 = torch.Generator().manual_seed(1)
        emb1 = torch.randn(1, 128, 1, generator=g0)
        emb2 = torch.randn(1, 128, 1, generator=g1)
        scale = torch.ones(1, 1)
        h0, c0 = _zero_state(1)
        with torch.no_grad():
            _, h1_pt, c1_pt = wrapper(emb1, scale, h0, c0)
            audio2_pt, _, _ = wrapper(emb2, scale, h1_pt, c1_pt)
        outs1 = session.run(
            None, {"emb": _np(emb1), "scale": _np(scale), "h_in": _np(h0), "c_in": _np(c0)}
        )
        outs2 = session.run(
            None, {"emb": _np(emb2), "scale": _np(scale), "h_in": outs1[1], "c_in": outs1[2]}
        )
        _assert_close(audio2_pt, outs2[0], "audio (stateful)")
