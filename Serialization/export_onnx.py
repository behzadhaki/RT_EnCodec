"""
export_onnx.py

Exports the rt_encodec encoder and decoder to ONNX.

Sections
--------
1. Sanity-check  — load the 24 kHz model, run encode / decode / forward on a
                   real audio clip and print shapes.
2. Encoder ONNX  — wrap the 48 kHz encoder, test it, export to
                   encodec_encoder.onnx, inspect the saved graph.
3. Decoder ONNX  — wrap the 48 kHz decoder, export to encodec_decoder.onnx.

Run from repo root:
    python serialization/export_onnx.py
or from inside the serialization/ folder:
    python export_onnx.py
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torchaudio
import onnx

# Suppress non-actionable export warnings:
# - legacy TorchScript ONNX exporter deprecation (switching to dynamo is a
#   separate effort; the legacy path works fine for this use case)
# - LSTM variable-batch warning (h0/c0 are initialised dynamically in SLSTM,
#   so the warning does not apply)
warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX.*")
warnings.filterwarnings("ignore", message=".*Exporting a model to ONNX with a batch_size other than 1.*")
warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm.*")
warnings.filterwarnings("ignore", message=".*Constant folding.*Only steps=1.*")
warnings.filterwarnings("ignore", message=".*torchaudio.*torchcodec.*")

from rt_encodec import EncodecModel


# ---------------------------------------------------------------------------
# 1. Sanity-check with the 24 kHz model
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Sanity-check (24 kHz model)")
print("=" * 60)

model_24 = EncodecModel.encodec_model_24khz()
model_24.set_target_bandwidth(12.0)
model_24.exporting_to_onnx = True
model_24.eval()

wav, sr = torchaudio.load(os.path.join(os.path.dirname(__file__), "..", "test_24k.wav"))
wav = wav[:, : model_24.sample_rate * 2]   # keep 2 seconds
wav_in = wav.unsqueeze(0)                  # [1, 1, T]
print(f"Audio input shape : {wav_in.shape}")

with torch.no_grad():
    wav_rec = model_24.forward(wav_in)
    print(f"Forward output    : {wav_rec.shape}")

    encoded_frames_24 = model_24.encode(wav_in)
    encodings = [ef[0] for ef in encoded_frames_24]
    print(f"Is normalised     : {model_24.normalize}")
    print(f"Segment frames    : {len(encoded_frames_24)}")
    print(f"Codes shapes      : {[e.shape for e in encodings]}")
    print(f"Scaling factors   : {[ef[1] for ef in encoded_frames_24]}")

    decoded = model_24.decode(encoded_frames_24)[:, :, : wav_in.shape[-1]]
    print(f"Decoded shape     : {decoded.shape}")


# ---------------------------------------------------------------------------
# 2. Encoder — export to ONNX (48 kHz model)
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("2. Encoder ONNX export (48 kHz model)")
print("=" * 60)


class EncodecEncoderWrapper(torch.nn.Module):
    def __init__(self, encodec_model):
        super().__init__()
        self.encodec_model = encodec_model

    def forward(self, x):
        # x: [n_channels, n_samples]  (no batch dimension)
        x_batched = x.unsqueeze(0)                      # [1, n_channels, n_samples]
        encoded_frames = self.encodec_model.encode(x_batched)
        codes = encoded_frames[0][0]                    # [1, n_codebooks, n_frames]
        return codes.squeeze(0)                         # [n_codebooks, n_frames]


encoder_model = EncodecModel.encodec_model_48khz()
encoder_model.set_target_bandwidth(3.0)
encoder_model.exporting_to_onnx = True
encoder_model.eval()

encoder_wrapper = EncodecEncoderWrapper(encoder_model)
# 48 kHz model is always stereo (2 channels); only n_samples is dynamic.
dummy_audio = torch.randn(2, 48000)   # [n_channels=2, n_samples]

with torch.no_grad():
    codes_out = encoder_wrapper(dummy_audio)
    print(f"Encoder input shape  : {dummy_audio.shape}")
    print(f"Encoder output shape : {codes_out.shape}")

encoder_onnx_path = os.path.join(os.path.dirname(__file__), "encodec_encoder.onnx")
torch.onnx.export(
    encoder_wrapper,
    dummy_audio,
    encoder_onnx_path,
    opset_version=13,
    input_names=["audio_input"],
    output_names=["encoded_codes"],
    # dim 0 (channels) is fixed at 2; only n_samples varies
    dynamic_axes={
        "audio_input":   {1: "n_samples"},
        "encoded_codes": {1: "n_frames"},
    },
)
print(f"Exported encoder → {encoder_onnx_path}")

onnx_encoder = onnx.load(encoder_onnx_path)
print("ONNX encoder input shape:", onnx_encoder.graph.input[0].type.tensor_type.shape)
print(f"segment_length={encoder_model.segment_length}  "
      f"segment_stride={encoder_model.segment_stride}")


# ---------------------------------------------------------------------------
# 3. Decoder — export to ONNX (48 kHz model, same weights)
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("3. Decoder ONNX export (48 kHz model)")
print("=" * 60)


class EncodecDecoderWrapper(torch.nn.Module):
    def __init__(self, encodec_model):
        super().__init__()
        self.encodec_model = encodec_model

    def forward(self, encoded_frames):
        # encoded_frames: List[EncodedFrame] where EncodedFrame = (codes, scale)
        decoded_audio = self.encodec_model.decode(encoded_frames)[0]
        return decoded_audio


decoder_model = EncodecModel.encodec_model_48khz()
decoder_model.set_target_bandwidth(3.0)
decoder_model.exporting_to_onnx = True
decoder_model.eval()

decoder_wrapper = EncodecDecoderWrapper(decoder_model)

# Generate example encoded frames from the 48 kHz encoder for tracing
with torch.no_grad():
    encoded_frames_48 = encoder_model.encode(dummy_audio.unsqueeze(0))

decoder_onnx_path = os.path.join(os.path.dirname(__file__), "encodec_decoder.onnx")
torch.onnx.export(
    decoder_wrapper,
    encoded_frames_48,
    decoder_onnx_path,
    opset_version=13,
    input_names=["encoded_frames"],
    output_names=["decoded_audio"],
    dynamic_axes={"encoded_frames": {2: "n_frames"}},
)
print(f"Exported decoder → {decoder_onnx_path}")
