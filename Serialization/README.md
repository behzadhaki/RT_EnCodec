# Serialization

ONNX export scripts and equivalence tests for `rt_encodec`.

---

## Scripts

### `export_onnx.py`

Exports each model as four independent ONNX graphs — one per pipeline stage.

```bash
# Export all bandwidths for both models (default)
python serialization/export_onnx.py

# Export specific bandwidths only
python serialization/export_onnx.py --bw24 6.0 12.0
python serialization/export_onnx.py --bw48 6.0
python serialization/export_onnx.py --bw24 6.0 --bw48 6.0
```

Outputs are written to `serialization/encodec_onnx_exports/`. Each model/bandwidth
combination gets its own subdirectory with four files.

---

### `test_onnx_equivalence.py`

Verifies that each exported ONNX graph produces outputs numerically identical
to the corresponding PyTorch stage. Requires `export_onnx.py` to have been run
first at `BW = 6.0 kbps` (the default test bandwidth).

```bash
pytest serialization/test_onnx_equivalence.py -v
```

What is checked per stage:

| Stage | Check |
|---|---|
| `encode_audio_segment` | embeddings allclose, LSTM state allclose |
| `quantize_encodings` | codes bit-exact (int64) |
| `decode_codes` | embeddings allclose |
| `decode_audio` | audio allclose, LSTM state allclose |

Stateful chaining (state from call N fed into call N+1) is also verified for
the two LSTM-bearing stages.

---

## Output folder structure

```
encodec_onnx_exports/
├── 24k/
│   ├── 1.5kbps/
│   │   ├── encode_audio_segment.onnx
│   │   ├── quantize_encodings.onnx
│   │   ├── decode_codes.onnx
│   │   └── decode_audio.onnx
│   ├── 3kbps/
│   ├── 6kbps/
│   ├── 12kbps/
│   └── 24kbps/
└── 48k/
    ├── 3kbps/
    ├── 6kbps/
    ├── 12kbps/
    └── 24kbps/
```

Each subdirectory contains the same four files; the model version and active
codebook count `K` are determined by the folder path.

---

## Stage I/O

### 24 kHz model

| Stage | Inputs | Outputs |
|---|---|---|
| `encode_audio_segment` | `audio [B, 1, T]`, `h_in [2, B, 512]`, `c_in [2, B, 512]` | `emb [B, 128, T_frames]`, `h_out`, `c_out` |
| `quantize_encodings` | `emb [B, 128, T_frames]` | `codes [B, K, T_frames]` (int64) |
| `decode_codes` | `codes [B, K, T_frames]` | `emb [B, 128, T_frames]` |
| `decode_audio` | `emb [B, 128, T_frames]`, `h_in [2, B, 512]`, `c_in [2, B, 512]` | `audio [B, 1, T]`, `h_out`, `c_out` |

`B = 1` for mono, `B = 2` for stereo-as-batch. `T_frames = T // 320`.

### 48 kHz model

| Stage | Inputs | Outputs |
|---|---|---|
| `encode_audio_segment` | `audio [1, 2, T]`, `h_in [2, 1, 512]`, `c_in [2, 1, 512]` | `emb [1, 128, T_frames]`, `scale [1, 1]`, `h_out`, `c_out` |
| `quantize_encodings` | `emb [1, 128, T_frames]` | `codes [1, K, T_frames]` (int64) |
| `decode_codes` | `codes [1, K, T_frames]` | `emb [1, 128, T_frames]` |
| `decode_audio` | `emb [1, 128, T_frames]`, `scale [1, 1]`, `h_in [2, 1, 512]`, `c_in [2, 1, 512]` | `audio [1, 2, T]`, `h_out`, `c_out` |

The 48 kHz encoder applies RMS normalisation and returns `scale`; the decoder
takes `scale` back to undo it. Minimum valid input: **1280 samples** (= 4 × 320).

---

## LSTM state

`h` and `c` have shape `[2, B, 512]` (2 layers, hidden size 512).

Pass zero tensors for stateless / one-shot inference:

```python
import torch
h = torch.zeros(2, 1, 512)
c = torch.zeros(2, 1, 512)
```

For streaming, feed the returned `h_out` / `c_out` back as inputs on the next
call to carry temporal context across chunk boundaries.

---

## K — active codebooks

`K = floor(bandwidth × 1000 / (frame_rate × 10))`

| Model | Bandwidth | K |
|---|---|---|
| 24 kHz (75 Hz) | 1.5 kbps | 2 |
| | 3 kbps | 4 |
| | 6 kbps | 8 |
| | 12 kbps | 16 |
| | 24 kbps | 32 |
| 48 kHz (150 Hz) | 3 kbps | 2 |
| | 6 kbps | 4 |
| | 12 kbps | 8 |
| | 24 kbps | 16 |
