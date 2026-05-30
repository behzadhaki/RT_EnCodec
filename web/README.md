# EnCodec Web Experiments

Browser-based demos using the exported ONNX models via `onnxruntime-web`.

---

## Running locally

Serve from the **repo root** (so the ONNX model paths resolve correctly):

```bash
python -m http.server 8000
# then open http://localhost:8000/web/experiment1.html
```

The models are loaded from `serialization/encodec_onnx_exports/`. Run
`python serialization/export_onnx.py` first if they don't exist yet.

---

## Experiments

### `experiment1.html` — Encode / Decode Explorer

Encode audio with EnCodec and compare the reconstruction spectrogram against
the original.

**Source**: sine, triangle, sawtooth, square, sweep (all with duration), or file upload (WAV / MP3 / etc., max 10 s).

**Model**: 24 kHz or 48 kHz, all exported bandwidths.

**Mode**:
- *Non-streaming* — encode the full audio as one pass, LSTM state reset to zero.
- *Streaming* — process in chunks with LSTM state carried across boundaries.
  For 24 kHz, chunk size is user-selectable (320 – 9600 samples).
  For 48 kHz, chunks are always 1 s (48 000 samples).

**Spectrogram**:
- *Source* — original audio in plasma colormap.
- *Decoded* — reconstructed audio in hot colormap.
- *Both* — plasma layer at full opacity, hot layer overlaid with screen blend.

---

### `experiment2.html` — Latent Space Interpolation

Configure two audio sources (A and B), encode both, then morph between them
by dragging an interpolation slider. The interpolation is performed at one of
three points in the EnCodec pipeline:

| Point | Location | What happens |
|---|---|---|
| **Encoder Latents** | After `encode_audio_segment`, before the VQ | Continuous encoder embeddings are linearly interpolated, then **re-quantized** (snapped to nearest codebook entries) before decoding. The decoder stays in-distribution while the interpolation explores the pre-VQ manifold. |
| **VQ Codes** | After `quantize_encodings` | Discrete codebook indices are mixed element-wise: each position uses B's code with probability α, determined by a deterministic golden-ratio hash. Mixed codes are re-embedded via `decode_codes` before audio decoding. |
| **Quantized Embeddings** | After `decode_codes`, before `decode_audio` | The continuous embeddings output by the code decoder are linearly interpolated, then decoded by the audio LSTM. |

Releasing the slider triggers a fresh inference pass through the pipeline from
the chosen interpolation point onwards. Works with both non-streaming and
streaming (OLA) modes. Results are shown in an **A → B** spectrogram tab
alongside the individual decoded spectrograms for A and B.

---

---

### `experiment3.html` — Dynamics Transfer

Configure two audio sources (A and B), encode both, then apply B's **embedding-space dynamics** onto A. The idea: extract the frame-to-frame motion of B's latent trajectory and inject it into A's latent space, so A takes on B's rhythmic / timbral variation while keeping its own spectral identity.

#### Transfer modes

| Mode | Formula | Notes |
|---|---|---|
| **Per-frame** | `out[t] = A[t] + str × δB[t]` | Adds each frame's delta independently. A runs naturally; B's micro-dynamics layer on top. |
| **Cumulative** | `out[t] = A[t] + str × Σ δB` | Accumulates B's deltas. A runs naturally; B's long-term drift integrates into A over time. |
| **Anchored** | `out[t] = A[onset] + str × (B[t] − B[onset])` | Freezes A at its onset embedding; follows B's trajectory from there. A's own motion disappears; you hear B's dynamics starting from A's tonal colour. |

`str` = Strength slider (0 – 1). `onset` = Onset slider (where in A the transfer begins).

#### Apply point

| Point | Location |
|---|---|
| **Encoder latents** | Pre-quantisation encoder output — widest modification range, re-quantised before decoding |
| **Quantized embeddings** | Post-RVQ decoder input — constrained to codebook space, decoded directly |

#### Delta method

**Raw** — deltas computed directly from B's embedding frames: `δB[t] = emb_B[t] − emb_B[t−1]`.

**SVD** — decomposes B's full delta sequence via eigendecomposition of the 128×128 delta covariance matrix. Lets you isolate the most (or least) significant axes of B's motion.

- **Comps slider** — how many leading eigenvectors to use (top-N mode).
- **Scree bar** — click individual bars to toggle components on/off; overrides the slider.
- **Pre-proc** flags (SVD only):
  - *Center δ* — subtract per-channel mean of deltas before SVD. Removes DC drift so components capture oscillatory motion only.
  - *Norm σ* — divide each channel by its std dev before SVD, rescale after reconstruction. Gives all 128 dims equal weight in the covariance instead of letting high-variance channels dominate. The DC mean (if centering is also on) is **not** re-added after reconstruction — this intentionally strips any long-term drift from the transfer.

#### Dim filter (Raw mode only)

Sliders for `|Δ| range` and `Var range` select which of the 128 embedding dimensions are active, based on each dimension's mean absolute delta and variance in B. Inactive dimensions pass A through unchanged.

#### 48 kHz notes

The 48 kHz model encodes stereo (mono duplicated to L/R). Encoder and decoder LSTM states are carried across OLA segments (1 s segments, 0.9 s stride) for continuous latent trajectories. Cross-segment deltas are zeroed out in the SVD covariance to prevent boundary jumps from corrupting the components.

---

## Files

```
web/
├── experiment1.html    Encode / decode explorer
├── experiment2.html    Latent space interpolation
├── experiment3.html    Dynamics transfer
├── encodec-worker.js   Web Worker: onnxruntime-web inference (shared by all)
└── README.md
```
