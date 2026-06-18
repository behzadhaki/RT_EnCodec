import json
import math
import os
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from .layers import Encoder, Decoder, NoiseBlock
from .vq import ResidualVectorQuantize


class SNAC(nn.Module):
    def __init__(
        self,
        sampling_rate=44100,
        encoder_dim=64,
        encoder_rates=[3, 3, 7, 7],
        latent_dim=None,
        decoder_dim=1536,
        decoder_rates=[7, 7, 3, 3],
        attn_window_size=32,
        codebook_size=4096,
        codebook_dim=8,
        vq_strides=[8, 4, 2, 1],
        noise=True,
        depthwise=True,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(
            encoder_dim,
            encoder_rates,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )
        self.n_codebooks = len(vq_strides)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.vq_strides = vq_strides
        self.attn_window_size = attn_window_size
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            vq_strides=vq_strides,
        )
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            noise,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )

    def preprocess(self, audio_data):
        length = audio_data.shape[-1]
        lcm = math.lcm(self.vq_strides[0], self.attn_window_size or 1)
        pad_to = self.hop_length * lcm
        right_pad = math.ceil(length / pad_to) * pad_to - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data

    def forward(self, audio_data: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data)
        z = self.encoder(audio_data)
        z_q, codes = self.quantizer(z)
        audio_hat = self.decoder(z_q)
        return audio_hat[..., :length], codes

    def encode(self, audio_data: torch.Tensor) -> List[torch.Tensor]:
        audio_data = self.preprocess(audio_data)
        z = self.encoder(audio_data)
        _, codes = self.quantizer(z)
        return codes

    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        z_q = self.quantizer.from_codes(codes)
        audio_hat = self.decoder(z_q)
        return audio_hat

    # ------------------------------------------------------------------ #
    # rt_snac: four-stage pipeline for ONNX export (mirrors rt_encodec).  #
    # Stages are pure functions of their inputs (no LSTM / recurrent      #
    # state — SNAC is fully convolutional). The multi-scale codes stay a  #
    # LIST of per-level tensors end to end (never flattened), and         #
    # decode_codes returns the per-level embedding contributions          #
    # *unsummed* so the caller can audition any subset of levels.         #
    # ------------------------------------------------------------------ #

    def set_noise(self, enabled: bool) -> "SNAC":
        """Enable/disable the decoder's stochastic NoiseBlock term in place.

        Default (True) reproduces upstream exactly. False makes the decoder
        deterministic and ONNX-exportable (the noise term is dropped)."""
        for m in self.modules():
            if isinstance(m, NoiseBlock):
                m.noise_enabled = enabled
        return self

    def encode_audio_segment(self, audio_data: torch.Tensor) -> torch.Tensor:
        """Stage 1. audio [B,1,T] -> continuous latent z [B, latent_dim, T_b]."""
        audio_data = self.preprocess(audio_data)
        return self.encoder(audio_data)

    def quantize_encodings(self, z: torch.Tensor) -> List[torch.Tensor]:
        """Stage 2. z [B, latent_dim, T_b] -> list of n_codebooks int64 code
        tensors, one per level, each at its own rate (T_b // stride_i)."""
        _, codes = self.quantizer(z)
        return codes

    def decode_codes(self, codes: List[torch.Tensor]) -> List[torch.Tensor]:
        """Stage 3. per-level codes -> per-level quantized embeddings zq_i,
        each [B, latent_dim, T_b] (already upsampled to the base grid).
        NOT summed: caller picks which levels to keep, then sums."""
        zqs = []
        for i in range(self.quantizer.n_codebooks):
            q = self.quantizer.quantizers[i]
            z_p_i = q.decode_code(codes[i])
            z_q_i = q.out_proj(z_p_i)
            z_q_i = z_q_i.repeat_interleave(q.stride, dim=-1)
            zqs.append(z_q_i)
        return zqs

    def decode_audio(self, z_q: torch.Tensor) -> torch.Tensor:
        """Stage 4. summed embedding z_q [B, latent_dim, T_b] -> audio [B,1,T'].
        Output is the padded length; caller trims to the original sample count."""
        return self.decoder(z_q)

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(**config)
        return model

    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        from huggingface_hub import hf_hub_download

        if not os.path.isdir(repo_id):
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json", **kwargs)
            model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", **kwargs)
            model = cls.from_config(config_path)
            state_dict = torch.load(model_path, map_location="cpu")
        else:
            model = cls.from_config(os.path.join(repo_id, "config.json"))
            state_dict = torch.load(os.path.join(repo_id, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
