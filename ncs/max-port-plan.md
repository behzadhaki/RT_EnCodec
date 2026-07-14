# EnCodec & SNAC to Max Port

## Overview

Port of encodec and snac neural audio codecs to Max as a collection of
single-purpose externals. Each model variant is a separate module. The pipeline
is decomposed into four operations: **encode**, **vq**, **embedcodes**, **decode**.

Two implementations exist:
- **Buffer** modules (`ncs.*`) -- read/write from `buffer~`, process entire signals
- **Real-time** modules (`ncs.rt.*`) -- stream audio sample-by-sample via `~` inlets/outlets

`vq` and `embedcodes` are stateless and shared between both implementations.
Only `encode` and `decode` have separate buffer and streaming variants.

---

## Model Variants

### EnCodec

| Variant | Sample Rate | Channels | Frame Rate | Codebooks | Bandwidths | Real-time |
|---------|------------|----------|------------|-----------|------------|-----------|
| 24 kHz | 24000 Hz | 1 (mono) | 75 Hz | 32 max | 1.5, 3, 6, 12, 24 kbps | Yes |
| 48 kHz | 48000 Hz | 2 (stereo) | 150 Hz | 16 max | 3, 6, 12, 24 kbps | No |

Active codebooks per bandwidth (K = floor(bw * 1000 / (frame_rate * 10))):

| Bandwidth | 24 kHz K | 48 kHz K |
|-----------|----------|----------|
| 1.5 kbps | 2 | -- |
| 3 kbps | 4 | 2 |
| 6 kbps | 8 | 4 |
| 12 kbps | 16 | 8 |
| 24 kbps | 32 | 16 |

Key architectural notes:
- 24 kHz: causal convolutions, stateful LSTM, no overlap-add
- 48 kHz: non-causal convolutions, stereo, RMS normalisation (encode computes
  scale, decode denormalises), segment-based overlap-add -- cannot run real-time

### SNAC

| Variant | Sample Rate | Frame Rate | Codebooks | VQ Strides | Attn Window | Real-time |
|---------|------------|------------|-----------|------------|-------------|-----------|
| 24 kHz | 24000 Hz | ~54.42 Hz | 3 | [4, 2, 1] | None | Yes |
| 32 kHz | 32000 Hz | ~72.56 Hz | 4 | [8, 4, 2, 1] | 32 | Yes (chunked) |
| 44 kHz | 44100 Hz | ~100.0 Hz | 4 | [8, 4, 2, 1] | 32 | Yes (chunked) |

Key architectural notes:
- Fully convolutional -- no LSTM state
- Multi-scale RVQ: different codebook levels operate at different temporal
  resolutions (finest level at base frame rate, coarser levels downsampled
  by their stride factor)
- 32 kHz and 44 kHz have local attention (non-overlapping windows of 32
  frames) -- requires chunked processing for streaming
- Decoder has stochastic NoiseBlock (disabled for deterministic output)
- Codebook: 4096 entries, factorised to 8-dim with L2-normalised cosine distance

---

## The Four Operations

| Operation | Input | Output | Stateful |
|-----------|-------|--------|----------|
| **encode** | audio samples | latent embeddings | Yes (24k EnCodec: LSTM state) |
| **vq** | latent embeddings | code indices | No |
| **embedcodes** | code indices | quantized embeddings | No |
| **decode** | quantized embeddings | audio samples | Yes (24k EnCodec: LSTM state) |

---

## Module Manifest

### Naming Convention

```
ncs.rt.<model>_<variant>.<operation>     # real-time
ncs.<model>_<variant>.<operation>        # buffer
```

### SNAC -- 24 modules (4 buffer + 4 RT per variant)

```
ncs.snac_24k.encode              ncs.rt.snac_24k.encode~
ncs.snac_24k.vq                  ncs.rt.snac_24k.vq
ncs.snac_24k.embedcodes          ncs.rt.snac_24k.embedcodes
ncs.snac_24k.decode              ncs.rt.snac_24k.decode~

ncs.snac_32k.encode              ncs.rt.snac_32k.encode~
ncs.snac_32k.vq                  ncs.rt.snac_32k.vq
ncs.snac_32k.embedcodes          ncs.rt.snac_32k.embedcodes
ncs.snac_32k.decode              ncs.rt.snac_32k.decode~

ncs.snac_44k.encode              ncs.rt.snac_44k.encode~
ncs.snac_44k.vq                  ncs.rt.snac_44k.vq
ncs.snac_44k.embedcodes          ncs.rt.snac_44k.embedcodes
ncs.snac_44k.decode              ncs.rt.snac_44k.decode~
```

### EnCodec 24 kHz -- 40 modules (4 buffer + 4 RT per bandwidth)

```
ncs.encodec_24k_1.5kbps.encode       ncs.rt.encodec_24k_1.5kbps.encode~
ncs.encodec_24k_1.5kbps.vq           ncs.rt.encodec_24k_1.5kbps.vq
ncs.encodec_24k_1.5kbps.embedcodes   ncs.rt.encodec_24k_1.5kbps.embedcodes
ncs.encodec_24k_1.5kbps.decode       ncs.rt.encodec_24k_1.5kbps.decode~

ncs.encodec_24k_3kbps.encode         ncs.rt.encodec_24k_3kbps.encode~
ncs.encodec_24k_3kbps.vq             ncs.rt.encodec_24k_3kbps.vq
ncs.encodec_24k_3kbps.embedcodes     ncs.rt.encodec_24k_3kbps.embedcodes
ncs.encodec_24k_3kbps.decode         ncs.rt.encodec_24k_3kbps.decode~

ncs.encodec_24k_6kbps.encode         ncs.rt.encodec_24k_6kbps.encode~
ncs.encodec_24k_6kbps.vq             ncs.rt.encodec_24k_6kbps.vq
ncs.encodec_24k_6kbps.embedcodes     ncs.rt.encodec_24k_6kbps.embedcodes
ncs.encodec_24k_6kbps.decode         ncs.rt.encodec_24k_6kbps.decode~

ncs.encodec_24k_12kbps.encode        ncs.rt.encodec_24k_12kbps.encode~
ncs.encodec_24k_12kbps.vq            ncs.rt.encodec_24k_12kbps.vq
ncs.encodec_24k_12kbps.embedcodes    ncs.rt.encodec_24k_12kbps.embedcodes
ncs.encodec_24k_12kbps.decode        ncs.rt.encodec_24k_12kbps.decode~

ncs.encodec_24k_24kbps.encode        ncs.rt.encodec_24k_24kbps.encode~
ncs.encodec_24k_24kbps.vq            ncs.rt.encodec_24k_24kbps.vq
ncs.encodec_24k_24kbps.embedcodes    ncs.rt.encodec_24k_24kbps.embedcodes
ncs.encodec_24k_24kbps.decode        ncs.rt.encodec_24k_24kbps.decode~
```

### EnCodec 48 kHz -- 16 modules (4 per bandwidth, no ~ variants)

```
ncs.encodec_48k_3kbps.encode         ncs.encodec_48k_3kbps.vq
ncs.encodec_48k_3kbps.embedcodes     ncs.encodec_48k_3kbps.decode

ncs.encodec_48k_6kbps.encode         ncs.encodec_48k_6kbps.vq
ncs.encodec_48k_6kbps.embedcodes     ncs.encodec_48k_6kbps.decode

ncs.encodec_48k_12kbps.encode        ncs.encodec_48k_12kbps.vq
ncs.encodec_48k_12kbps.embedcodes    ncs.encodec_48k_12kbps.decode

ncs.encodec_48k_24kbps.encode        ncs.encodec_48k_24kbps.vq
ncs.encodec_48k_24kbps.embedcodes    ncs.encodec_48k_24kbps.decode
```

**Total: 64 modules**

Note: For EnCodec, the `encode` and `decode` modules are identical across all
bandwidths of the same sample rate. The bandwidth-specific ONNX graphs are only
needed for `vq` and `embedcodes`. The per-bandwidth namespacing is kept for
consistency, but encode/decode could potentially be deduplicated.

---

## Interface Specifications

### encode (buffer)

Reads audio from a named `buffer~`, processes the entire signal.

```
Inlet 1 (message): buffer name (symbol), bang to trigger
Outlet 1 (message): embeddings as list
Outlet 2 (message): (48k only) RMS scale as float
```

### encode~ (real-time)

Processes audio signal input frame-by-frame.

```
Inlet 1 (signal): audio input
Inlet 2 (message): bang to reset LSTM state (24k EnCodec only)
Outlet 1 (message): embeddings as list, one message per frame
Outlet 2 (message): (48k only) RMS scale as float
```

### vq

Stateless. Quantises embeddings into code indices.

```
Inlet 1 (message): embeddings as list
Outlet 1 (message): codes as list
```

**EnCodec vq output format:** Flat list of K integers (codebook indices),
one per active codebook. K is fixed at load time based on the bandwidth
variant (e.g., K=8 for 24k 6kbps). Example for K=4: `512 1023 0 42`.

**SNAC vq output format:** List of N integers (one per codebook level),
all transmitted at base frame rate. Coarser-level codes are repeated
(zero-order hold) to align with the finest level. N is fixed at load time.
Example for N=4: `5 1023 42 7`.

### embedcodes

Stateless. Looks up codebook entries and sums quantised embeddings.

```
Inlet 1 (message): codes as list (same format as vq output)
Outlet 1 (message): quantised embeddings as list
```

### decode (buffer)

Writes decoded audio to a named `buffer~`.

```
Inlet 1 (message): buffer name (symbol), embeddings as list
Inlet 2 (message): (48k only) RMS scale as float
```

### decode~ (real-time)

Takes embedding messages and outputs audio signal.

```
Inlet 1 (message): embeddings as list
Inlet 2 (message): (48k only) RMS scale as float
Outlet 1 (signal): decoded audio output
```

---

## Data Flow

### Real-time pipeline

```
[audio in] --> ncs.rt.*.encode~ --> ncs.rt.*.vq --> ncs.rt.*.embedcodes --> ncs.rt.*.decode~ --> [audio out]
                (signal)            (list/msg)       (list/msg)              (list/msg)           (signal)
```

### Buffer pipeline

```
[buffer~] --> ncs.*.encode --> ncs.*.vq --> ncs.*.embedcodes --> ncs.*.decode --> [buffer~]
               (triggered)     (list/msg)    (list/msg)           (triggered)
```

### Direct code transfer (bypass vq/embedcodes)

Codes can be stored, transmitted, or manipulated as integer lists between
the vq and embedcodes steps. This enables:
- Saving codes to disk (compressed storage)
- Network transmission of codes only
- Manual code manipulation before decoding
- Level-by-level code selection (SNAC)

---

## Controls

| Module | Control | Type | Values | Notes |
|--------|---------|------|--------|-------|
| `encodec_24k_*.vq` | bandwidth | int (K) | 2, 4, 8, 16, 32 | Number of active codebooks. Fixed at load time per module variant. |
| `encodec_48k_*.vq` | bandwidth | int (K) | 2, 4, 8, 16 | Same as above. |
| `snac_*.vq` | none | -- | -- | Always uses all codebooks (3 or 4). |
| All encode/decode | none | -- | -- | Fixed neural net weights. |
| All embedcodes | none | -- | -- | Fixed codebook lookup. |

The bandwidth is a **load-time** parameter, not a runtime control. Each
bandwidth variant is a separate module with its own ONNX graph where K
is baked in. This avoids runtime codebook slicing.

---

## Timing and Latency

### Frame Rates

| Model | Frame Rate | Samples per Frame |
|-------|------------|-------------------|
| EnCodec 24k | 75 Hz | 320 |
| EnCodec 48k | 150 Hz | 320 |
| SNAC 24k | ~54.42 Hz | 441 |
| SNAC 32k | ~72.56 Hz | 441 |
| SNAC 44k | ~100.0 Hz | 441 |

### Streaming Latency (encode~ to decode~)

| Model | Latency | Calculation |
|-------|---------|-------------|
| EnCodec 24k | 4.27 ms | 320 samples / 24000 Hz |
| EnCodec 48k | N/A | Non-causal, buffer only |
| SNAC 24k | 18.38 ms | 441 samples / 24000 Hz |
| SNAC 32k | 441 ms | 14112 samples / 32000 Hz (32-frame attention chunk) |
| SNAC 44k | 320 ms | 14112 samples / 44100 Hz (32-frame attention chunk) |

SNAC 32k/44k streaming latency comes from the minimum chunk size required
by the local attention: `hop_length * lcm(vq_strides[0], attn_window_size)`
= `441 * lcm(8, 32)` = `441 * 32` = 14112 samples. The attention operates
on non-overlapping windows of 32 frames, so this is the irreducible minimum.

---

## SNAC Hierarchical Codes

### Multi-scale structure

SNAC codebooks operate at different temporal resolutions. For snac_32k
with strides [8, 4, 2, 1]:

| Level | Stride | Rate | Code per N base frames |
|-------|--------|------|------------------------|
| 0 | 8 | ~9.07 Hz | 1 per 8 |
| 1 | 4 | ~18.14 Hz | 1 per 4 |
| 2 | 2 | ~36.28 Hz | 1 per 2 |
| 3 | 1 | ~72.56 Hz | 1 per 1 |

### Transmission strategy

All levels are transmitted at **base frame rate** (~72.56 Hz for 32k).
Coarser-level codes are repeated via zero-order hold:

```
Frame:  0    1    2    3    4    5    6    7    8    9   ...
Lv 0:   A    A    A    A    A    A    A    A    B    B  ...
Lv 1:   C    C    C    C    D    D    D    D    E    E  ...
Lv 2:   F    F    G    G    H    H    I    I    J    J  ...
Lv 3:   K    L    M    N    O    P    Q    R    S    T  ...
```

### Why this is safe

The VQ module runs the full SNAC quantizer in one forward pass. All N
levels are quantized simultaneously from the same latent `z`. The
repetition is purely a timing alignment -- no codes are ever skipped
or lost. This is zero-order hold interpolation on discrete indices.

### Why not audio rate

Codes are discrete integer indices (0-4095) at frame rate (54-100 Hz).
Upsampling to audio rate (44.1-48 kHz) would be 400-900x redundant with
no benefit. The `decode~` module handles the frame-to-audio conversion.

---

## Implementation Notes

### ONNX model loading

Each module loads a single ONNX graph at instantiation. The existing
ONNX exports already map to the four operations:

| ONNX graph | Max operation |
|------------|---------------|
| `encode_audio_segment.onnx` | encode |
| `quantize_encodings.onnx` | vq |
| `decode_codes.onnx` | embedcodes |
| `decode_audio.onnx` | decode |

For EnCodec, each bandwidth has its own set of 4 ONNX graphs (K is
baked in). For SNAC, each sample rate has one set of 4 ONNX graphs.

### Codebook files

EnCodec codebooks are embedded in the ONNX quantize/decode graphs.

SNAC codebooks are exported separately as sidecar files:
- `codebooks.bin` -- float32, shape [n_q, codebook_size, codebook_dim]
- `codebooks.json` -- metadata (n_q, vocab_size, dim)

These must be loaded alongside the ONNX graphs for the `vq` and
`embedcodes` modules.

### LSTM state (EnCodec 24k only)

The 24k encode/decode modules maintain LSTM state (h, c) across frames:
- Shape: [2, 1, 512] per tensor (2 layers, batch=1, hidden=512)
- `encode~` outputs updated state, feeds back on next frame
- `decode~` consumes state from encode~ or from its own feedback
- A bang to inlet 2 resets state to zeros

The 48k model is non-causal and has no LSTM state.

### SNAC noise gate

The SNAC decoder has a stochastic NoiseBlock. For deterministic output
(matching ONNX exports), noise is disabled. If a runtime noise toggle
is desired, it could be exposed as an attribute on the decode module.

### Preprocessing (SNAC buffer mode)

SNAC encode requires input length to be a multiple of
`hop_length * lcm(vq_strides[0], attn_window_size)`:
- 24k: multiple of 441 (hop_length only, no attention)
- 32k/48k: multiple of 14112

Buffer-mode encode modules handle padding internally.

### Preprocessing (SNAC real-time chunked)

For 32k/44k streaming, the encode~ module accumulates samples until
14112 are available (one 32-frame attention chunk), then processes the
chunk. The user controls effective latency via the signal vector size
in Max's DSP settings.
