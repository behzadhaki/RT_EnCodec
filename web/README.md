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

**Source**: sine, triangle, sawtooth, square, sweep, or file upload (WAV / MP3, max 10 s).

**Model**: 24 kHz or 48 kHz, all exported bandwidths.

**Mode**:
- *Non-streaming* — encode the full audio as one pass, LSTM state reset to zero.
- *Streaming* — process in chunks with LSTM state carried across boundaries.
  For 24 kHz, chunk size is user-selectable (320 – 9600 samples).
  For 48 kHz, chunks are always 1 s (48 000 samples).

**Spectrogram**: Source (plasma) / Decoded (hot) / Both (blended).

---

### `experiment2.html` — Latent Space Interpolation

Configure two audio sources (A and B), encode both, then morph between them
by dragging an interpolation slider. Interpolation is performed at one of
three points in the pipeline:

| Point | Location | What happens |
|---|---|---|
| **Encoder Latents** | After encoder LSTM, before VQ | Continuous embeddings are lerp'd then re-quantised before decoding. |
| **VQ Codes** | After `quantize_encodings` | Discrete codes are mixed element-wise with deterministic probability α. |
| **Quantized Embeddings** | After `decode_codes`, before `decode_audio` | The continuous code-decoder output is lerp'd then decoded by the audio LSTM. |

Works with non-streaming and streaming (OLA) modes.

---

### `experiment3.html` — Dynamics Transfer

Apply B's **embedding-space dynamics** onto A. Extracts the frame-to-frame
motion of B's latent trajectory and injects it into A's latent space.

#### Transfer modes

| Mode | Formula |
|---|---|
| **Per-frame** | `out[t] = A[t] + str × δB[t]` |
| **Cumulative** | `out[t] = A[t] + str × Σ δB` |
| **Anchored** | `out[t] = A[onset] + str × (B[t] − B[onset])` |

#### Delta method

**Raw** — frame deltas `δB[t] = emb_B[t] − emb_B[t−1]`.

**SVD** — eigendecomposition of B's 128×128 delta covariance. Scree bar lets
you toggle individual components; slider selects top-N. Optional pre-processing:
*Center δ* (subtract mean) and *Norm σ* (equalise channel variance).

**Apply point**: encoder latents (pre-VQ, re-quantised) or quantized embeddings (post-VQ).

---

### `experiment4.html` — SVD Component Swap

Compute a separate SVD for each source (A and B), then swap or blend their
principal components before decoding. Lets you transfer the dominant
structural directions of B's embedding space into A's, or vice versa.

Each source's encoder latents (or quantized embeddings) are decomposed as
`E = U S Vᵀ`. The scree bars show singular values for A and B. Clicking or
dragging a bar toggles/weights which components are active in the reconstruction.

**Apply point**: encoder latents or quantized embeddings.

---

### `experiment5.html` — Joint SVD Interpolation

Compute a **joint** SVD over the concatenated embeddings `[E_A | E_B]`, yielding
a shared basis U and split time-coefficients V_A, V_B. Interpolation is:

```
E_new[c,t] = Σ_k  U[c,k] · S_eff[k] · V_eff[k,t]
V_eff[k,t] = (1−α) V_A[k,t] + α V_B[k,t]
```

**Singular-value shaping** controls how much each component contributes:

| Mode | S_eff[k] |
|---|---|
| **None** | S[k] (unchanged) |
| **Scale** | S[k] × scalar |
| **Soft** | S[k] × w[k] (drag scree bars individually) |
| **Tilt** | S[k] × e^(−tilt × k/C) (exponential roll-off) |

The shared basis means interpolation stays on the joint data manifold rather
than linearly blending two independent coordinate systems.

---

### `experiment6.html` — Codebook UMAP Explorer

Visualise **all encoder frames** from both sources jointly in 2D via UMAP,
then navigate the latent space in several ways: click to hear a frame,
draw paths, or follow graph-based routes — all decoded directly from the
captured embeddings.

#### UMAP embedding space

Three independent controls shape what each UMAP point represents.
Decoding is **always** from the original `emb_quant` regardless of these choices —
they only affect how frames are positioned in the 2D projection.

**Space** — which embedding to use as the base:

| Toggle | What is projected |
|---|---|
| **Pre-VQ** *(default)* | Raw encoder LSTM output before quantisation (`emb_enc`). Smoother topology; reveals the continuous manifold before the VQ bottleneck collapses it to codebook attractors. |
| **Post-VQ** | Output of `decode_codes` (`emb_quant`) — the actual decoder input. Shows the discrete codebook structure. |

**Per-level mode** projects the raw codebook vector for a single VQ level
`codebook[level][codes[level, t]]` — fully independent of LSTM context.

**Features** — transform applied to the base embedding before UMAP:

| Toggle | UMAP input (per frame, dimensionality) | Effect |
|---|---|---|
| **Emb** *(default)* | `emb[t]` — D | Standard; clusters by timbral content. |
| **Emb+Δ** | `emb[t] ‖ (emb[t]−emb[t−1])` — 2D | Clusters by both current content and rate of change. Separates frames that sound similar but are evolving differently. |
| **Δ** | `emb[t]−emb[t−1]` — D | Clusters purely by velocity in latent space. Silence/sustain collapses to one region; attacks/transients spread by direction of motion. |

**History** — number of past frames K to include as local context:

| K | UMAP input | Notes |
|---|---|---|
| **1** *(default)* | Features-only vector | No history; same as before. |
| **K > 1** | `emb[t] ‖ feat[t−K+1] ‖ … ‖ feat[t]` — D + K×D_feat | Current raw embedding as anchor, then K steps of the selected feature (oldest → newest). Frames are only neighbours if their local trajectories also match, not just their instantaneous values. |

Edge frames (t < K−1) are clamped: `feat[t<0]` uses `feat[0]`.

The three controls compose freely — e.g. **Δ + History 5** gives each point a
`[D + 5D]` vector of current position plus a 5-step local velocity trajectory.

#### Explore mode

Click anywhere on the UMAP to decode and play the nearest frame. A half-second
context window is decoded around the clicked frame using the actual captured
embeddings and scale.

#### Trajectory modes

Draw a freehand path on the canvas (click and drag). The path can be
**shift-dragged** to translate it in UMAP space. Any change to the path
auto-triggers a new decode.

Three playback modes are available once a path is drawn:

---

##### Draw

Interpolates smoothly along the drawn curve. At each time step the two nearest
UMAP frames are found and their `emb_quant` vectors are blended using
inverse-Euclidean weighting before decoding.

**Duration slider** — total playback time in seconds. Controls how many
interpolated frames are sampled along the curve.

---

##### Code

Treats the drawn path's start and end points as endpoints in a **k-NN graph**
built from all UMAP frame positions (temporal edges within each source always
present; cross-source edges within the UMAP k-neighbour radius).

Dijkstra finds the shortest path between the nearest frame to the start and
the nearest frame to the end. Only actual captured frames are played — no
interpolation. The path is visualised as a teal line through the UMAP dots.

---

##### Snap

Morphs the drawn curve to pass through actual data points. Samples the path
at `duration × fps` positions, snapping each to the nearest UMAP frame, then
deduplicates consecutive identical frames to get a sequence of **waypoints**.

The sequence is always decoded from real `emb_quant` frames; what changes
across sub-modes is how context (pre-roll) is constructed around each waypoint.

**Duration sub-modes**

| Sub-mode | Behaviour |
|---|---|
| **Auto** | Fixed pre-roll of ~0.05 s per context window. Total duration follows naturally from the sequence length. |
| **Fixed** | All waypoints are always visited. The frame budget `N_target = round(duration × fps)` is distributed evenly: each waypoint receives `floor(budget / N_waypoints)` context frames (plus one extra for the first `budget % N_waypoints` waypoints). Hard-trimmed or extended to hit exactly N_target. |

**Context construction — `walkBackward`**

At each backward step the algorithm compares:
1. The temporal predecessor in the same source `(src, idx − 1)`.
2. Any frame from the other source within `Ctx swap` UMAP units of the
   current position.

The closer of the two (in UMAP distance) is chosen. Because the check repeats
at every step, context can hop between sources multiple times — a multi-hop
backward walk through the directed latent graph.

**Ctx swap slider** (0 → 2.0 UMAP units)

- `0` — context is always same-source (temporal backward only).
- `> 0` — cross-source hops are allowed whenever a frame from the other source
  falls within the slider radius of the current context position.

The waypoints themselves are never swapped; only the context source changes.

---

#### Loop

Plays the decoded trajectory repeatedly until stopped.

#### 48 kHz notes

The 48 kHz decoder is called in 150-frame chunks (matching the OLA segment
size used during encoding). Each chunk uses the mean of the per-frame RMS
scale values for that window, computed from the actual captured encoder scales.
LSTM state is threaded across chunks for audio continuity.

---

## Files

```
web/
├── experiment1.html    Encode / decode explorer
├── experiment2.html    Latent space interpolation
├── experiment3.html    Dynamics transfer
├── experiment4.html    SVD component swap
├── experiment5.html    Joint SVD interpolation
├── experiment6.html    Codebook UMAP explorer
├── encodec-worker.js   Web Worker: onnxruntime-web inference (shared by all)
├── umap-worker.js      Web Worker: UMAP dimensionality reduction
├── encodec.css         Shared dark-theme stylesheet
├── lib/                Shared UI components (source panels, model panel, etc.)
└── README.md
```
