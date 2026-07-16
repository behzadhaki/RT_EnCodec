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

**Terminology: "frame"** means one time-step of the model's latent/embedding
sequence (the `T` axis of a `[latent_dim x T]` tensor) -- NOT an audio
sample. One frame = `hop_length` audio samples of the original waveform
(the encoder's convolutional downsampling factor). E.g. SNAC 32k/44k
have `hop_length=384`, so "32 frames" = 32 x 384 = 12288 audio samples;
SNAC 24k has `hop_length=512`, so "4 frames" = 4 x 512 = 2048 samples.
Every "N frames"/"frame rate" figure in this doc uses this meaning.

---

## Model Variants

### EnCodec

| Variant | Sample Rate | Channels | Frame Rate | Codebooks | Bandwidths | Real-time |
|---------|------------|----------|------------|-----------|------------|-----------|
| *model name* | *native audio sample rate* | *mono or stereo* | *encoder output rate (frames/sec)* | *max codebooks the model supports* | *bitrate variants available* | *can this run in `rt~` mode* |
| 24 kHz | 24000 Hz | 1 (mono) | 75 Hz | 32 max | 1.5, 3, 6, 12, 24 kbps | Yes |
| 48 kHz | 48000 Hz | 2 (stereo) | 150 Hz | 16 max | 3, 6, 12, 24 kbps | No |

Active codebooks per bandwidth (K = floor(bw * 1000 / (frame_rate * 10))):

| Bandwidth | 24 kHz K | 48 kHz K |
|-----------|----------|----------|
| *target bitrate* | *active codebook count (K) at 24kHz* | *active codebook count (K) at 48kHz* |
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
| *model name* | *native audio sample rate* | *encoder output rate (frames/sec)* | *number of RVQ levels* | *downsample factor per level, coarsest first* | *local-attention window size in frames, if any* | *can this run in `rt~` mode* |
| 24 kHz | 24000 Hz | 46.875 Hz | 3 | [4, 2, 1] | None | Yes |
| 32 kHz | 32000 Hz | 83.33 Hz | 4 | [8, 4, 2, 1] | 32 | Yes (chunked) |
| 44 kHz | 44100 Hz | 114.84 Hz | 4 | [8, 4, 2, 1] | 32 | Yes (chunked) |

*"Frame" = one time-step of the latent sequence, not an audio sample --
see "Terminology" in Overview.*

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
| *pipeline stage* | *what it consumes* | *what it produces* | *does it carry state across calls* |
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

### SNAC -- 24 modules (4 buffer + 4 RT per variant) -- IMPLEMENTED

Actual built module names (note the `kh` suffix, and rt~ folders end in
`_tilde`/`.vq`/`.embedcodes`/`.decode_tilde` but compile to externals
named with the trailing `~`, e.g. `ncs.rt.snac_24kh.encode~`):

```
ncs.snac_24kh.encode             ncs.rt.snac_24kh.encode~
ncs.snac_24kh.vq                 ncs.rt.snac_24kh.vq
ncs.snac_24kh.embedcodes         ncs.rt.snac_24kh.embedcodes
ncs.snac_24kh.decode             ncs.rt.snac_24kh.decode~

ncs.snac_32kh.encode             ncs.rt.snac_32kh.encode~
ncs.snac_32kh.vq                 ncs.rt.snac_32kh.vq
ncs.snac_32kh.embedcodes         ncs.rt.snac_32kh.embedcodes
ncs.snac_32kh.decode             ncs.rt.snac_32kh.decode~

ncs.snac_44kh.encode             ncs.rt.snac_44kh.encode~
ncs.snac_44kh.vq                 ncs.rt.snac_44kh.vq
ncs.snac_44kh.embedcodes         ncs.rt.snac_44kh.embedcodes
ncs.snac_44kh.decode             ncs.rt.snac_44kh.decode~
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

### encode~ (real-time) -- SNAC as built

Processes audio signal input continuously via windowed context-trim
streaming (see "Real-time streaming design" below) -- there is no
per-frame LSTM-state handshake for SNAC (fully convolutional, no
recurrent state).

```
Inlet 1 (signal): audio input
Outlet 1 (message): messageix/tensor chunked embeddings block,
                     channel-major [latent_dim x T], one block per
                     messageix/tensor sequence
Outlet 2 (message, float): @monitor_rtf value, emitted every block
```

Attribute: `@monitor_rtf` (bool, default **on**) -- see "Real-time
performance controls" below.

### vq -- SNAC as built

Stateless. Quantises embeddings into code indices.

```
Inlet 1 (message): messageix/tensor chunked embeddings (from encode~),
                    or `load <path>` to point at a different model
Outlet 1..N (message): per-level codes as plain "list" (tiny regardless
                        of rate, never chunked)
Outlet N+1 (message, float): @monitor_rtf value, emitted every block
```

**SNAC vq output format:** N separate outlets (one per codebook level,
N=3 for 24k / N=4 for 32k-44k), each a plain list of native-resolution
codes at that level's own temporal resolution (NOT upsampled/repeated
to base rate at this stage -- embedcodes does the per-level lookup at
native resolution too; only the FINAL summed embedding is a single
base-rate stream). Outlets send finest level FIRST and level 0
(coarsest) LAST, so a downstream embedcodes (whose leftmost/level0
inlet is hot) sees its cold level1..N-1 inlets updated before the hot
trigger arrives.

### embedcodes -- SNAC as built

Stateless. Looks up codebook entries and sums quantised embeddings.

```
Inlet 1 (hot, message): level 0 codes as plain list, or `load <path>`
Inlet 2..N (cold, message): level 1..N-1 codes as plain list
Outlet 1 (message): messageix/tensor chunked SUMMED embeddings
                     (level0_scale*L0 + ... ), channel-major [latent x T]
Outlet 2..N+1 (message): messageix/tensor chunked per-level unscaled
                          embeddings (for auditioning a single level)
Outlet N+2 (message, float): @monitor_rtf value, emitted every block
```

Attributes: `@level<N>_scale` (float, default 1.0, per level) -- gain
applied before summing into the sum outlet only. `@monitor_rtf` (bool,
default on).

### decode (buffer)

Writes decoded audio to a named `buffer~`.

```
Inlet 1 (message): buffer name (symbol), embeddings as list
```

### decode~ (real-time) -- SNAC as built

Takes chunked embedding messages and outputs a continuous audio signal
via the same windowed context-trim approach as encode~, plus a causal
overlap-add crossfade as a seam-smoothing safety net.

```
Inlet 1 (message): messageix/tensor/load/reset -- quantized embeddings
                    chunk (normally from embedcodes' sum outlet), or
                    `load <path>`, or `reset` to clear streaming state
Outlet 1 (signal): decoded audio output
Outlet 2 (message, bang): underrun state changed -- bangs on entering
                           underrun (silence) and on recovering from it
Outlet 3 (message, float): @monitor_rtf value, emitted every block
```

Attributes: `@prebuffer_blocks` (int, default **0**) -- accumulate N
decoded blocks before (re)starting playback, trading added latency for
headroom against transient processing slowdowns; re-arms after every
underrun so playback resumes with a fresh reserve instead of racing the
worker sample-by-sample. `@monitor_rtf` (bool, default on).

---

## Data Flow

### Real-time pipeline (SNAC, as built)

```
[audio in] --> ncs.rt.snac_*.encode~ --> ncs.rt.snac_*.vq --> ncs.rt.snac_*.embedcodes --> ncs.rt.snac_*.decode~ --> [audio out]
                (signal)                 (messageix/tensor)   (list in / messageix/tensor out)  (messageix/tensor)   (signal)
```

All three rates (24kh/32kh/44kh) use the identical messageix/tensor
chunked transport for encode~->vq and embedcodes->decode~ (see
"Real-time message protocol" below) -- 24k's blocks never actually need
more than one chunk, but speak the same wire format anyway so any
receiver/tooling built for one rate works unmodified against all three.
vq->embedcodes codes stay a plain "list" on all three rates (always
tiny). **This is a transport-level match only -- 24k's tensors are a
genuinely different shape (768ch/3 levels) from 32k/44k's (1024ch/4
levels), so 24k modules cannot be swapped in for 32k/44k modules or vice
versa; 32k and 44k ARE freely interchangeable (identical architecture,
only sample rate and trained weights differ).**

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
| *which module(s) this applies to* | *attribute or parameter name* | *data type* | *allowed range/values (default noted where relevant)* | *what it does / when to use it* |
| `encodec_24k_*.vq` | bandwidth | int (K) | 2, 4, 8, 16, 32 | Number of active codebooks. Fixed at load time per module variant. |
| `encodec_48k_*.vq` | bandwidth | int (K) | 2, 4, 8, 16 | Same as above. |
| `snac_*.vq` | none | -- | -- | Always uses all codebooks (3 or 4). |
| All encode/decode | none | -- | -- | Fixed neural net weights. |
| All embedcodes | none | -- | -- | Fixed codebook lookup. |
| `snac_*.embedcodes` | `@level<N>_scale` | float | any (default 1.0) | Per-level gain applied only when summing into the sum outlet; raw per-level outlets unaffected. |
| `rt.snac_*.decode~` | `@prebuffer_blocks` | int | >=0 (default 0) | Blocks to accumulate before (re)starting playback; 0 = no change from baseline behavior. |
| all `rt.snac_*.*` | `@monitor_rtf` | bool | 0/1 (default 1/on) | Always emits per-block cost out the rightmost outlet; on = real-time factor (elapsed/block-duration), off = raw ms. See "Real-time performance controls" below. |

The bandwidth is a **load-time** parameter, not a runtime control. Each
bandwidth variant is a separate module with its own ONNX graph where K
is baked in. This avoids runtime codebook slicing.

---

## Timing and Latency

### Frame Rates (verified against exported model.json)

| Model | Frame Rate | hop_length (samples/frame) |
|-------|------------|-----------------------------|
| *model + rate* | *encoder output rate (frames/sec)* | *audio samples represented by one latent frame* |
| EnCodec 24k | 75 Hz | 320 |
| EnCodec 48k | 150 Hz | 320 |
| SNAC 24k | 46.875 Hz | 512 |
| SNAC 32k | 83.33 Hz | 384 |
| SNAC 44k | 114.84 Hz | 384 |

*"Frame" = one time-step of the latent sequence, not an audio sample --
see "Terminology" in Overview. `hop_length` IS the frame-to-sample
conversion factor (1 frame = `hop_length` samples).*

### Streaming Latency (encode~ to decode~, as implemented)

The irreducible rt~ latency floor is one `kBlockSize` (the windowed
context-trim block, NOT a single frame): `hop_length * lcm(vq_strides[0],
attn_window_size or 1)`.

| Model | Latency | Calculation |
|-------|---------|-------------|
| *model + rate* | *irreducible encode~-to-decode~ latency floor* | *how that figure is derived* |
| EnCodec 24k | 4.27 ms | 320 samples / 24000 Hz |
| EnCodec 48k | N/A | Non-causal, buffer only |
| SNAC 24k | ~85.3 ms | 2048 samples / 24000 Hz (512 * lcm(4,1)=4) |
| SNAC 32k | ~384 ms | 12288 samples / 32000 Hz (384 * lcm(8,32)=32) |
| SNAC 44k | ~278.6 ms | 12288 samples / 44100 Hz (384 * lcm(8,32)=32) |

*The `lcm(...)=N` term in each Calculation is a frame count (see
"Terminology" in Overview) -- it's multiplied by `hop_length` to get
the sample figure shown.*

SNAC 32k/44k's much larger block (and latency floor) than 24k's comes
from local attention forcing the block to align to a full 32-frame
window; 24k has no attention (attn_window_size=None) so its floor is set
by the much smaller `vq_strides[0]=4` instead. This is architectural,
not a transport/implementation choice -- it can't be reduced without
changing the exported ONNX graphs.

---

## SNAC Hierarchical Codes

### Multi-scale structure

SNAC codebooks operate at different temporal resolutions. For snac_32kh
(base frame rate 83.33 Hz) with strides [8, 4, 2, 1]:

| Level | Stride | Rate | Code per N base frames |
|-------|--------|------|------------------------|
| *codebook level index (0=coarsest)* | *downsample factor vs. base rate* | *this level's native temporal resolution* | *how sparse this level's native codes are vs. base rate* |
| 0 | 8 | ~10.42 Hz | 1 per 8 |
| 1 | 4 | ~20.83 Hz | 1 per 4 |
| 2 | 2 | ~41.67 Hz | 1 per 2 |
| 3 | 1 | ~83.33 Hz | 1 per 1 |

*"Base frames" here = frames at the model's finest/base rate (see
"Terminology" in Overview) -- level 3's own frames ARE base frames
(stride 1); level 0's frames each span 8 base frames (stride 8).*

### Transmission strategy (as implemented)

`vq`'s per-level outlets send codes at each level's own **native
(shorter, for coarser levels) resolution** -- there is no manual
zero-order-hold/repeat step in our Max/C++ code at this stage. The
upsample back to base rate happens **inside** `decode_codes.onnx` (a
Tile op baked into the exported graph) as part of `embedcodes`'
codebook lookup -- by the time embedcodes' outlets emit, every stream
(sum + per-level) is already at base frame rate. So: `vq` outlets =
native/ragged resolution per level; `embedcodes` outlets = uniform
base-rate embeddings.

### Why not audio rate

Codes are discrete integer indices (0-4095) at frame rate (~47-115 Hz
depending on the SNAC rate). Upsampling to audio rate (24.1-44.1 kHz)
would be ~200-500x redundant with no benefit. The `decode`/`decode~`
module handles the frame-to-audio conversion.

---

## Implementation Notes

### ONNX model loading

Each module loads a single ONNX graph at instantiation. The existing
ONNX exports already map to the four operations:

| ONNX graph | Max operation |
|------------|---------------|
| *exported graph filename* | *which of the 4 Max operations loads it* |
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
`hop_length * lcm(vq_strides[0], attn_window_size or 1)` (= `preprocess_pad`
in each rate's exported `model.json`):
- 24k: multiple of 2048 (512 * lcm(4,1)=4)
- 32k/44k: multiple of 12288 (384 * lcm(8,32)=32)

Buffer-mode encode modules handle padding internally.

### Real-time streaming design (all 3 SNAC rates, as implemented)

There is no incremental/cached conv-state ONNX graph for SNAC (unlike
EnCodec's LSTM h/c) -- the exported graphs always run their whole conv
stack fresh on whatever they're given. So `encode~`/`decode~` use
**windowed context-trim streaming**: each new `kBlockSize`-sample block
(2048 @ 24k, 12288 @ 32k/44k -- one `preprocess_pad` unit) is prepended
with the PREVIOUS block as trailing context before `Run()`, and only the
newly-computed (non-context) frames are kept -- the context-derived
frames are discarded every call. This needs no ONNX graph changes and
reuses the exact buffer-mode weights/graphs unchanged; the cost is
discarding roughly half the compute (the context half) every block, and
the `kBlockSize`-sample latency floor from the table above. `decode~`
additionally applies a short causal overlap-add crossfade (looking only
backward into the context region, so no added latency) purely as a
numerical safety net against residual seam clicks. `vq`/`embedcodes`
reuse the buffer-mode ONNX graphs unchanged too (quantization/codebook
lookup are pointwise per-frame, no temporal receptive field, so running
them continuously on small streamed blocks is exactly as correct as
running them once on a whole buffer). Host-rate <-> model-rate
resampling uses a persistent-state `ncs_resample::StreamingResampler`
(not the buffer modules' one-shot `resample_simple`), since a one-shot
resample re-run every block would click at block boundaries.

### Analysis window sizes (as implemented)

The context window is always exactly ONE block (`kContextSize ==
kBlockSize` for `encode~`; `kContextFrames == kBlockFrames` for
`decode~`), so the actual `Run()` input span every call is TWICE the
new-audio block size:

| Rate | New block | Context | Total `Run()` input (encode~, samples) | Total `Run()` input (decode~, latent frames) |
|------|-----------|---------|------------------------------------------|-----------------------------------------------|
| *SNAC sample rate* | *newly-arrived audio/embeddings processed this call* | *trailing data prepended before `Run()` (always == new block)* | *actual encode~ ONNX input size (context+new)* | *actual decode~ ONNX input size (context+new)* |
| 24k | 2048 samples (4 frames) | 2048 samples (4 frames) | 4096 samples | 8 frames (768 x 8) |
| 32k | 12288 samples (32 frames) | 12288 samples (32 frames) | 24576 samples | 64 frames (1024 x 64) |
| 44k | 12288 samples (32 frames) | 12288 samples (32 frames) | 24576 samples | 64 frames (1024 x 64) |

*"Frames" = latent time-steps, not audio samples (see "Terminology" in
Overview) -- the `(N frames)` shown alongside each sample count is that
many samples divided by the rate's `hop_length`.*

Only the newly-computed (non-context) half of each `Run()`'s output is
kept -- the context half is discarded every call, so roughly half the
compute per block is "wasted" recomputing context that was already
correct from the previous call. This is the accepted cost of having no
incremental/cached conv state to reuse.

`decode~`'s causal OLA crossfade window (`kOlaSamples`) is a fixed 256
samples on all 3 rates, but that's a different DURATION per rate since
the sample rate differs:

| Rate | `kOlaSamples` | Duration |
|------|----------------|----------|
| *SNAC sample rate* | *crossfade window size (samples, fixed across rates)* | *that many samples' duration at this rate's sample rate* |
| 24k | 256 | ~10.7 ms |
| 32k | 256 | ~8.0 ms |
| 44k | 256 | ~5.8 ms |

It looks only backward into the context region (never forward), so it
adds zero algorithmic latency beyond the context window above -- purely
a numerical safety net against residual seam clicks at block boundaries.

`encode~`'s host-rate audio is handed to the streaming resampler in
fixed 256-host-sample chunks (`kHostFlushSamples`, same constant on all
3 rates) -- independent of Max's DSP signal-vector size, so resampling
behavior doesn't change if the user changes their vector size setting.

### Minimum input size to run each operation

Two different floors are in play, and they only coincide for 24k:

| Operation | Minimum input | Why | 24k (frames / samples) | 32k-44k (frames / samples) |
|-----------|----------------|-----|--------------------------|------------------------------|
| *what needs the input* | *smallest valid amount of new audio/frames* | *architectural reason for the floor* | *floor at 24k's model rate* | *floor at 32k/44k's model rate* |
| `encode~`/`decode~` (as built, rt~) | `kBlockFrames` = `lcm(vq_strides[0], attn_window_size or 1)` | Block size must satisfy BOTH the coarsest VQ stride AND the attention window at once -- this is the same figure as the "Streaming Latency" table above, just framed as an input-size floor instead of a latency number | 4 frames / 2048 samples | 32 frames / 12288 samples |
| `vq`/`embedcodes` (standalone architectural floor) | `vq_strides[0]` (coarsest level's stride) | The coarsest quantizer level downsamples by this stride internally (per the exported model's multi-scale code lengths -- level0 length = T/stride); feeding fewer frames than the stride means the coarsest level's code tensor would come out zero-length | 4 frames / 2048 samples (same as above -- 24k has no attention, so both floors coincide) | 8 frames / 3072 samples (smaller than encode~/decode~'s 32-frame block) |

*"Frames" = latent time-steps, not audio samples (see "Terminology" in
Overview) -- the sample figure in each cell is frames x that rate's
`hop_length`.*

In normal rt~ use this never matters -- `encode~` always hands `vq`
blocks already sized to the larger (attention) floor. It's relevant if
you feed `vq`/`embedcodes` directly with your own data (e.g. via the
"Direct code transfer" bypass above, or a custom encoder): for 32k/44k,
a hand-built block can be as small as 8 frames (3072 samples-equivalent)
and still produce valid per-level codes, well below the 32-frame
(12288-sample) block the built-in `encode~` always uses.

### Real-time message protocol (messageix/tensor chunking)

`encode~`'s output and `embedcodes`' output are both a full
channel-major `[latent_dim x T]` tensor per block -- 32768 atoms for
32k/44k (1024 ch x 32 frames), 3072 atoms for 24k (768 ch x 4 frames).
Max's own `outlet_cache`/`real_outlet_list` has a confirmed crash
threshold around 15,000-20,000 atoms in a single list message (a raw
`memmove` overflow) -- 32k/44k's blocks are well past that, so both
hops go out as a chunked `messageix <index>` (index -1 = last chunk)
immediately followed by `tensor <data...>`, both real Max message
selectors, reassembled on the receiving side. `kMaxChunkAtoms=12000`
per chunk (safe margin below the crash zone). 24k's blocks (3072 atoms)
never actually need to split, but speak the identical messageix/tensor
protocol anyway (always exactly one chunk) purely for wire-format
uniformity across rates -- see "Cross-rate module compatibility" below
for why that does NOT make 24k mixable with 32k/44k. `vq`'s codes-in and
`vq`->`embedcodes` codes are always small and stay a plain `"list"` on
all three rates.

Each rt~ object's `flush_output()` (a 10ms main-thread timer) drains its
outgoing message queue and sends to the actual Max outlet -- this is
also where deferred `post()`/error logging happens, since calling
`error()` directly from a worker thread has crashed Max's console in
testing. `encode~`/`embedcodes` (the two chunk-emitting stages) drain up
to `kMaxSendsPerTick` (4) queued messages per tick instead of exactly
one, cutting inter-object relay latency significantly; `vq` drains its
entire (small, never-chunked) queue every tick since there's no
crash-risk message-size concern there.

### Real-time performance controls

- **Adaptive ONNX intra-op threading** (`adaptive_intra_op_threads()`,
  `shared_external_helpers.h`): scales `SetIntraOpNumThreads` to
  `min(4, hardware_concurrency())` instead of a hardcoded value, for the
  32k/44k modules (compute-heavy enough that this matters; 24k stays at
  1 thread, its compute is comfortably inside budget already). Avoids
  oversubscription on low-core machines while still using up to 4
  threads on typical hardware.
- **`@prebuffer_blocks`** (rt `decode~`, default 0): accumulate N
  decoded blocks in the output queue before (re)starting playback.
  Trades `N * kBlockSize`-worth of added latency for headroom against
  transient processing slowdowns (jitter). Only helps with jitter, not
  with a persistent throughput deficit (average per-block processing
  time exceeding the real-time budget) -- in that case it only delays
  the first underrun, and re-arms with a now-longer refill after every
  subsequent one.
- **`@monitor_rtf`** (all 4 rt~ module types, default **on**): every
  module always emits its own per-block cost (compute, plus resampling
  for `encode~`, which happens in `operator()` on the audio thread, not
  in the worker-thread `process()` the rest of the timing wraps) out a
  dedicated rightmost outlet, on every single block -- as a real-time
  factor (`elapsed / block_duration`; >=1.0 means that stage alone can
  no longer keep up) when on, or raw milliseconds when off. Since every
  stage divides by the same per-rate block-duration constant, summing
  all 4 stages' RTF values equals the true aggregate *compute* RTF
  (message-relay time between stages isn't included, though it's now
  small after the `kMaxSendsPerTick`/full-drain changes above).
- **`underrun_out`** (rt `decode~`): bangs when playback enters
  underrun (silence, nothing decoded yet) and when it recovers, in
  place of the console logging this used to do. Throttled to at most
  one bang per second per direction to avoid flooding the scheduler
  queue if underrun/recovery oscillates rapidly.
- **Model-load logging**: all modules (buffer and rt~, all rates) now
  log ONLY on load failure, pointing at the model's expected bundled
  path (`"<module>: failed to load model (<path>) -- <exception>.
  Models are expected at <default_model_path>."`) -- a successful load
  is silent.

### Cross-rate module compatibility

**32k and 44k modules of the same operation are freely interchangeable**
(identical architecture -- 1024 latent channels, 4 codebook levels, 32
frames/block, 384 hop -- only sample rate and trained weights differ).
**24k is NOT compatible with 32k or 44k for `encode~`/`vq`/`embedcodes`**,
regardless of the shared messageix/tensor wire format: 24k is a
genuinely different network (768 channels, 3 levels, 4 frames/block,
512 hop), so a 24k tensor fed into a 32k/44k model (or vice versa) is a
straight-up shape mismatch inside the ONNX graph (confirmed: throws
`Invalid input shape: {0}` from the quantizer's `in_proj` Conv node).
`decode~` may APPEAR to tolerate a cross-rate connection without an
audible crash, but that should not be trusted as real interop -- treat
all 24k <-> 32k/44k module mixing as unsupported.
