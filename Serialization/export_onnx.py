"""
export_onnx.py — export rt_encodec to ONNX as four independent graphs per model.

Each graph corresponds to one stage in the pipeline:

  encode_audio_segment.onnx   audio + LSTM state → embeddings + LSTM state
  quantize_encodings.onnx     embeddings → codes   (stateless)
  decode_codes.onnx           codes → embeddings   (stateless)
  decode_audio.onnx           embeddings + LSTM state → audio + LSTM state

For the 48 kHz model, encode_audio_segment also returns an RMS scale and
decode_audio takes it back to undo the normalisation.

LSTM state shape: [num_layers=2, B, hidden_size=512] for all stateful graphs.
  24 kHz: B=1 (mono) or B=2 (stereo-as-batch)
  48 kHz: B always 1

Outputs written to serialization/onnx_exports/{24k,48k}/{bw}kbps/
K (active codebooks) is fixed at export time:
  24 kHz: K = floor(bw * 1000 / (75  * 10))
  48 kHz: K = floor(bw * 1000 / (150 * 10))

Run from repo root:
  python serialization/export_onnx.py [--bw24 BW [BW ...]] [--bw48 BW [BW ...]]
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
warnings.filterwarnings("ignore", message=".*batch_size other than 1.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*")
warnings.filterwarnings("ignore", message=".*Constant folding.*Only steps=1.*")
warnings.filterwarnings("ignore", message=".*torchaudio.*torchcodec.*")

from rt_encodec import EncodecModel

_HERE = os.path.dirname(os.path.abspath(__file__))
LSTM_LAYERS = 2
LSTM_HIDDEN = 512   # mult * n_filters = 16 * 32, shared by encoder and decoder


# ---------------------------------------------------------------------------
# Stage wrappers — 24 kHz
# ---------------------------------------------------------------------------

class EncodeAudioSegment24(nn.Module):
    """SEANet encoder for the 24 kHz model.

    Inputs : audio [B, 1, T],        h [2, B, 512],  c [2, B, 512]
    Outputs: emb   [B, D, T_frames], h [2, B, 512],  c [2, B, 512]
    """
    def __init__(self, model: EncodecModel):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, audio: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        emb, (h_out, c_out) = self.encoder(audio, (h, c))
        return emb, h_out, c_out


class QuantizeEncodings24(nn.Module):
    """RVQ encoder for the 24 kHz model (stateless).

    Inputs : emb   [B, D, T_frames]
    Outputs: codes [B, K, T_frames]  (int64)
    """
    def __init__(self, model: EncodecModel):
        super().__init__()
        self.quantizer = model.quantizer
        self.frame_rate = model.frame_rate
        self.bandwidth = model.bandwidth

    def forward(self, emb: torch.Tensor):
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        return codes.transpose(0, 1)   # [K, B, Tf] → [B, K, Tf]


class DecodeCodes24(nn.Module):
    """RVQ decoder for the 24 kHz model (stateless).

    Inputs : codes [B, K, T_frames]
    Outputs: emb   [B, D, T_frames]
    """
    def __init__(self, model: EncodecModel):
        super().__init__()
        self.quantizer = model.quantizer

    def forward(self, codes: torch.Tensor):
        return self.quantizer.decode(codes.transpose(0, 1))   # [B,K,Tf]→[K,B,Tf]


class DecodeAudio24(nn.Module):
    """SEANet decoder for the 24 kHz model.

    Inputs : emb   [B, D, T_frames], h [2, B, 512],  c [2, B, 512]
    Outputs: audio [B, 1, T],        h [2, B, 512],  c [2, B, 512]
    """
    def __init__(self, model: EncodecModel):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, emb: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        audio, (h_out, c_out) = self.decoder(emb, (h, c))
        return audio, h_out, c_out


# ---------------------------------------------------------------------------
# Stage wrappers — 48 kHz
# ---------------------------------------------------------------------------

class EncodeAudioSegment48(nn.Module):
    """SEANet encoder for the 48 kHz model, with RMS normalisation.

    Inputs : audio [1, 2, T],                  h [2, 1, 512],  c [2, 1, 512]
    Outputs: emb   [1, D, T_frames], scale [1, 1], h [2, 1, 512], c [2, 1, 512]
    """
    def __init__(self, model: EncodecModel):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, audio: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        mono = audio.mean(dim=1, keepdim=True)
        volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
        scale = 1e-8 + volume
        emb, (h_out, c_out) = self.encoder(audio / scale, (h, c))
        return emb, scale.view(-1, 1), h_out, c_out


class QuantizeEncodings48(nn.Module):
    """RVQ encoder for the 48 kHz model (stateless).

    Inputs : emb   [1, D, T_frames]
    Outputs: codes [1, K, T_frames]  (int64)
    """
    def __init__(self, model: EncodecModel):
        super().__init__()
        self.quantizer = model.quantizer
        self.frame_rate = model.frame_rate
        self.bandwidth = model.bandwidth

    def forward(self, emb: torch.Tensor):
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        return codes.transpose(0, 1)   # [K, 1, Tf] → [1, K, Tf]


class DecodeCodes48(nn.Module):
    """RVQ decoder for the 48 kHz model (stateless).

    Inputs : codes [1, K, T_frames]
    Outputs: emb   [1, D, T_frames]
    """
    def __init__(self, model: EncodecModel):
        super().__init__()
        self.quantizer = model.quantizer

    def forward(self, codes: torch.Tensor):
        return self.quantizer.decode(codes.transpose(0, 1))


class DecodeAudio48(nn.Module):
    """SEANet decoder for the 48 kHz model, with RMS denormalisation.

    Inputs : emb   [1, D, T_frames], scale [1, 1], h [2, 1, 512], c [2, 1, 512]
    Outputs: audio [1, 2, T],                      h [2, 1, 512], c [2, 1, 512]
    """
    def __init__(self, model: EncodecModel):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, emb: torch.Tensor, scale: torch.Tensor,
                h: torch.Tensor, c: torch.Tensor):
        audio, (h_out, c_out) = self.decoder(emb, (h, c))
        return audio * scale.view(-1, 1, 1), h_out, c_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_state(batch: int, device='cpu') -> tuple:
    h = torch.zeros(LSTM_LAYERS, batch, LSTM_HIDDEN, device=device)
    c = torch.zeros(LSTM_LAYERS, batch, LSTM_HIDDEN, device=device)
    return h, c


def _export(wrapper, dummy_inputs, path, input_names, output_names, dynamic_axes):
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        path,
        opset_version=13,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    model_onnx = onnx.load(path)
    onnx.checker.check_model(model_onnx)
    print(f"  saved & verified → {path}")


# ---------------------------------------------------------------------------
# Codebook export
# ---------------------------------------------------------------------------

def _export_codebooks(model: EncodecModel, model_root_dir: str) -> None:
    """Export all RVQ codebook embedding matrices to a flat binary file.

    Writes two files to model_root_dir:
      codebooks.bin  — float32 LE, shape [n_q, vocab_size, dim], row-major
      codebooks.json — {"n_q": int, "vocab_size": int, "dim": int}

    All bandwidth variants share the same codebooks; bandwidth only controls
    how many levels are active, so this is exported once per model.
    """
    layers = model.quantizer.vq.layers
    embeds = [layer._codebook.embed.detach().cpu() for layer in layers]
    codebooks = torch.stack(embeds, dim=0)           # [n_q, vocab_size, dim]
    n_q, vocab_size, dim = codebooks.shape

    os.makedirs(model_root_dir, exist_ok=True)

    bin_path = os.path.join(model_root_dir, "codebooks.bin")
    with open(bin_path, "wb") as f:
        f.write(codebooks.numpy().astype("float32").tobytes())

    meta_path = os.path.join(model_root_dir, "codebooks.json")
    with open(meta_path, "w") as f:
        json.dump({"n_q": n_q, "vocab_size": vocab_size, "dim": dim}, f)

    print(f"  codebooks [{n_q}, {vocab_size}, {dim}] → {bin_path}")


# ---------------------------------------------------------------------------
# Per-bandwidth export helpers
# ---------------------------------------------------------------------------

BW_24 = [1.5, 3.0, 6.0, 12.0, 24.0]
BW_48 = [3.0, 6.0, 12.0, 24.0]
_EXPORTS_ROOT = os.path.join(_HERE, "encodec_onnx_exports")


def _bw_dirname(bw: float) -> str:
    return f"{bw:g}kbps"


def _export_24khz(model: EncodecModel, bw: float) -> None:
    out_dir = os.path.join(_EXPORTS_ROOT, "24k", _bw_dirname(bw))
    os.makedirs(out_dir, exist_ok=True)

    model.set_target_bandwidth(bw)
    K = int(1000 * bw // (model.frame_rate * 10))
    print(f"\n  24 kHz  {bw:g} kbps  (K={K})")

    # T=321: odd → non-zero extra_padding exercised (avoids constant-folding)
    B, T = 1, 321
    h, c = _zero_state(B)
    audio = torch.randn(B, 1, T)

    # 1. encode_audio_segment
    enc_seg = EncodeAudioSegment24(model)
    with torch.no_grad():
        emb_out, _, _ = enc_seg(audio, h, c)
    _export(enc_seg, (audio, h, c),
            os.path.join(out_dir, "encode_audio_segment.onnx"),
            input_names=["audio", "h_in", "c_in"],
            output_names=["emb", "h_out", "c_out"],
            dynamic_axes={
                "audio": {0: "B", 2: "T"},
                "h_in":  {1: "B"}, "c_in":  {1: "B"},
                "emb":   {0: "B", 2: "T_frames"},
                "h_out": {1: "B"}, "c_out": {1: "B"},
            })

    # 2. quantize_encodings
    quant = QuantizeEncodings24(model)
    with torch.no_grad():
        codes_out = quant(emb_out)
    _export(quant, (emb_out,),
            os.path.join(out_dir, "quantize_encodings.onnx"),
            input_names=["emb"],
            output_names=["codes"],
            dynamic_axes={"emb": {0: "B", 2: "T_frames"}, "codes": {0: "B", 2: "T_frames"}})

    # 3. decode_codes
    dec_codes = DecodeCodes24(model)
    codes = torch.randint(0, 1024, (B, K, codes_out.shape[-1]))
    with torch.no_grad():
        emb_dec = dec_codes(codes)
    _export(dec_codes, (codes,),
            os.path.join(out_dir, "decode_codes.onnx"),
            input_names=["codes"],
            output_names=["emb"],
            dynamic_axes={"codes": {0: "B", 2: "T_frames"}, "emb": {0: "B", 2: "T_frames"}})

    # 4. decode_audio
    dec_audio = DecodeAudio24(model)
    _export(dec_audio, (emb_dec, h, c),
            os.path.join(out_dir, "decode_audio.onnx"),
            input_names=["emb", "h_in", "c_in"],
            output_names=["audio", "h_out", "c_out"],
            dynamic_axes={
                "emb":   {0: "B", 2: "T_frames"},
                "h_in":  {1: "B"}, "c_in":  {1: "B"},
                "audio": {0: "B", 2: "T"},
                "h_out": {1: "B"}, "c_out": {1: "B"},
            })


def _export_48khz(model: EncodecModel, bw: float) -> None:
    out_dir = os.path.join(_EXPORTS_ROOT, "48k", _bw_dirname(bw))
    os.makedirs(out_dir, exist_ok=True)

    model.set_target_bandwidth(bw)
    K = int(1000 * bw // (model.frame_rate * 10))
    print(f"\n  48 kHz  {bw:g} kbps  (K={K})")

    # T=1281: odd + ≥ 1280 (=4×320). Min valid input: 1280 samples.
    h, c = _zero_state(1)
    audio = torch.randn(1, 2, 1281)

    # 1. encode_audio_segment
    enc_seg = EncodeAudioSegment48(model)
    with torch.no_grad():
        emb_out, scale_out, _, _ = enc_seg(audio, h, c)
    _export(enc_seg, (audio, h, c),
            os.path.join(out_dir, "encode_audio_segment.onnx"),
            input_names=["audio", "h_in", "c_in"],
            output_names=["emb", "scale", "h_out", "c_out"],
            dynamic_axes={"audio": {2: "T"}, "emb": {2: "T_frames"}})

    # 2. quantize_encodings
    quant = QuantizeEncodings48(model)
    with torch.no_grad():
        codes_out = quant(emb_out)
    _export(quant, (emb_out,),
            os.path.join(out_dir, "quantize_encodings.onnx"),
            input_names=["emb"],
            output_names=["codes"],
            dynamic_axes={"emb": {2: "T_frames"}, "codes": {2: "T_frames"}})

    # 3. decode_codes
    dec_codes = DecodeCodes48(model)
    codes = torch.randint(0, 1024, (1, K, codes_out.shape[-1]))
    with torch.no_grad():
        emb_dec = dec_codes(codes)
    _export(dec_codes, (codes,),
            os.path.join(out_dir, "decode_codes.onnx"),
            input_names=["codes"],
            output_names=["emb"],
            dynamic_axes={"codes": {2: "T_frames"}, "emb": {2: "T_frames"}})

    # 4. decode_audio
    dec_audio = DecodeAudio48(model)
    scale = torch.ones(1, 1)
    _export(dec_audio, (emb_dec, scale, h, c),
            os.path.join(out_dir, "decode_audio.onnx"),
            input_names=["emb", "scale", "h_in", "c_in"],
            output_names=["audio", "h_out", "c_out"],
            dynamic_axes={"emb": {2: "T_frames"}, "audio": {2: "T"}})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export rt_encodec pipeline stages to ONNX. "
                    "Without flags, all bandwidths for both models are exported."
    )
    parser.add_argument("--bw24", type=float, nargs="+", metavar="BW",
                        help="24 kHz bandwidth(s) to export (kbps). "
                             f"Default: all {BW_24}")
    parser.add_argument("--bw48", type=float, nargs="+", metavar="BW",
                        help="48 kHz bandwidth(s) to export (kbps). "
                             f"Default: all {BW_48}")
    args = parser.parse_args()

    bws_24 = args.bw24 if args.bw24 else BW_24
    bws_48 = args.bw48 if args.bw48 else BW_48

    print(f"Output root: {_EXPORTS_ROOT}")
    print(f"24 kHz bandwidths : {bws_24}")
    print(f"48 kHz bandwidths : {bws_48}")

    if bws_24:
        print(f"\n{'='*60}")
        print("24 kHz model")
        print(f"{'='*60}")
        model_24 = EncodecModel.encodec_model_24khz()
        model_24.exporting_to_onnx = True
        model_24.eval()
        for bw in bws_24:
            _export_24khz(model_24, bw)
        _export_codebooks(model_24, os.path.join(_EXPORTS_ROOT, "24k"))

    if bws_48:
        print(f"\n{'='*60}")
        print("48 kHz model")
        print(f"{'='*60}")
        model_48 = EncodecModel.encodec_model_48khz()
        model_48.exporting_to_onnx = True
        model_48.eval()
        for bw in bws_48:
            _export_48khz(model_48, bw)
        _export_codebooks(model_48, os.path.join(_EXPORTS_ROOT, "48k"))

    print(f"\nDone. All exports written to {_EXPORTS_ROOT}/")


if __name__ == "__main__":
    main()
