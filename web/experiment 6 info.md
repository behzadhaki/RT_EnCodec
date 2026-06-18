# Experiment 6 — Architecture

## Computation levels

Everything derives from a strict cascade. When something changes at level N, only levels N+1 and below need to update.

```
SOURCES ──► EMBEDS ──► COORDS ──► WAYPOINTS ──► AUDIO ──► POST-PROCESS
                                                               │
VIEWPORT ──────────────────────────────────────────────────────┤
PLAYBACK_TICK ─────────────────────────────────────────────────┤
                                                               ▼
                                                            RENDER
```

VIEWPORT (pan/zoom) and PLAYBACK_TICK are orthogonal — they only affect rendering, never upstream computation.

---

## State variables

```js
// Sources
rawAudioA, rawAudioB, modelHz
statefulOla          // 48k: carry LSTM state across OLA segments (false = reset per chunk; default false → parallel encode)
encPool              // persistent pool of encode workers for parallel 48k-stateless segments
durA, durB               // actual loaded duration (s) per source; null until first encode

// Folder sources + code sidecars (see "Folder sources, code sidecars…" below)
folderClipFracsA/B   // [0..1] per-clip start fractions when a source is a folder (null otherwise)
pendingResolvedA/B   // {codes,codesDims,scales} resolved from cache for a source whose real encode
                     //   is skipped — a placeholder is sent instead, merged back in onWorkerMsg
pendingSidecarA/B    // {panel,audio,modelOpts} — per-window sidecars to write post-encode
                     //   (24k / 48k-stateful only; 48k whole-file writes 0-full entries inline)
_fullCodesCache      // Map "name:size:hz:bw:ola" → {codes,codesDims,scales}: in-memory full-clip
                     //   codes so re-sampling / A-vs-B / reloads are free within a session

// EMBEDS (async)
currentEmbeds        // {embEncA/B, embEncDimsA/B, embQuantA/B, embQuantDimsA/B,
                     //  codesA/B, codesDimsA/B, scalesA/B}
                     // populated either from worker (encode path) or from cache (cache hit)
encodePhase          // 'A' | 'B' | null

// COORDS inputs
projectionMode       // 'umap' | 'pca' | 'wheel'
wheelRadius          // 'loud' | 'time' | 'atyp' — wheel-mode radius source
wheelAngle           // 'umap1d' | 'pc1' — wheel-mode angle source
wheelScale           // 'value' | 'rank' — wheel-mode radius scaling
umapMode, umapLevel, embSpace, embMode
nNeighbors, minDist, umapWindow
ctxDirection         // 'past' | 'centered' | 'future' — context window placement + delta type
ctxShape             // 'flat' | 'triangular' | 'gaussian' — context slot weighting
                     // (z-score runs before windowing so slot weights survive)
preReduceEnabled     // bool — PCA pre-reduction of UMAP features (≤32 dims, 95% var)
umapMetric           // 'euclidean' | 'cosine' — UMAP distance function
scaleFeatEnabled     // bool — append log(scale) loudness dim (48k only; no-op without scales)
dedupEnabled         // bool — collapse consecutive near-duplicate frames before the UMAP fit

// COORDS (async)
coordsA, coordsB
lastTransform        // {xMin,yMin,scale,offX,offY,H} — returned by renderer; used for coordinate conversion

// Source visibility
srcAEnabled          // bool — whether source A frames are shown/used
srcBEnabled          // bool — whether source B frames are shown/used
// getEnabledSrcs() → ['A'], ['B'], or ['A','B'] (never empty — falls back to both)

// WAYPOINTS inputs
canvasMode           // 'explore' | 'draw' | 'pins'
userTrajectory       // resampled path — draw mode source of truth
rawDrawSegments      // [{pts:[x,y][], enabled:bool}] — completed strokes
rawDrawPoints        // [x,y][] — in-progress stroke (before mouseUp)
drawSegmentBoundaries // indices in userTrajectory where non-first segments begin
pinPoints            // [{src, idx}, ...] — stores frame identity, not UMAP coords
pathMode             // 'draw' | 'code' | 'snap'
trajDir              // 'forward' | 'backward' | 'pingpong'
snapMode             // 'k_frames' | 'equal' | 'proportional'
snapKVal             // K pre-roll frames per waypoint (k_frames mode)
snapDurVal           // total duration in seconds (equal/prop mode)
codeDurMode          // 'frames' | 'seconds'
codeKVal             // total frames (code/frames mode)
codeSecVal           // total seconds (code/seconds mode)
ctxSwapProb          // 0–1: probability of attempting a source/frame swap at each pre-roll step
ctxSwapDist          // UMAP-unit radius for the nearby-frame candidate search

// WAYPOINTS (sync, derived by computeWaypoints())
snapFrames           // visual waypoint dots for snap mode
snapSegBoundaries    // indices in snapFrames where non-first draw segments begin
codePathFrames       // visual waypoint dots for code mode
snapPlayFrames       // full frame sequence with _wp tags for snap playback
genPlaySeq           // [{src,idx},...] frame sequence consumed by audio decode

// POST-PROCESS
lastRawPcm           // Float32Array — unfaded decoded PCM, kept for live fade re-apply
lastRawPcmSr         // sample rate of lastRawPcm

// PLAYBACK (driven by AudioContext time, render-only)
trajAnchorPos        // 0..1 playback progress, -1 = stopped
trajLooping

// RENDER-ONLY (no downstream computation)
viewZoom, viewPanX, viewPanY
selectedPoint, waveHighlight, showTrail
isDrawing, isPanDragging, isShiftDragging, isPinDragging, ...
```

---

## Dirty flags

```js
const dirty = { COORDS: false, WAYPOINTS: false, AUDIO: false, RENDER: false };
```

---

## Entry points → dirty flags

| Trigger | Dirty flags set |
|---------|-----------------|
| Source audio / model change | `EMBEDS → COORDS WAYPOINTS AUDIO RENDER` (via scheduleEncode) |
| Embeddings arrive | `COORDS WAYPOINTS AUDIO RENDER` |
| Any UMAP/PCA param change | `COORDS WAYPOINTS AUDIO RENDER` |
| UMAP/PCA coords arrive | `WAYPOINTS AUDIO RENDER` |
| userTrajectory / pinPoints change | `WAYPOINTS AUDIO RENDER` (via autoRegenerate) |
| Path params change (pathMode, trajDir, snapMode, etc.) | `WAYPOINTS AUDIO RENDER` (via autoRegenerate) |
| Source visibility toggle (srcAEnabled/B) | `WAYPOINTS AUDIO RENDER` (via autoRegenerate) + `RENDER` |
| Pan / zoom / resize | `RENDER` only |
| Waveform player cursor move | `RENDER` only (via scheduleRender) |
| Playback tick (animateTrajAnchor) | `RENDER` only (via scheduleRender) |
| Explore click (selectedPoint) | `RENDER` only |
| Fade in/out inputs change | re-applies `applyGenFade` to `lastRawPcm` directly, no dirty flags |
| Volume slider | `playerG.setVolume()` directly, no dirty flags |

---

## runEffects() — the single scheduler

```js
function runEffects() {
  if (dirty.COORDS && currentEmbeds) {
    dirty.COORDS = false;
    triggerProjection();        // async → dispatches WAYPOINTS AUDIO RENDER on complete
  }
  if (dirty.WAYPOINTS && coordsA && coordsB && hasWaypointSource()) {
    dirty.WAYPOINTS = false;
    computeWaypoints();         // sync — updates snapFrames/codePathFrames/genPlaySeq
    dirty.AUDIO  = true;
    dirty.RENDER = true;
  }
  if (dirty.AUDIO && currentEmbeds && coordsA && coordsB
      && hasWaypointSource()
      && (genPlaySeq !== null || pathMode === 'draw')) {
    dirty.AUDIO = false;
    playTrajectory();           // async → worker decode → playerG.setAudio
  }
  if (dirty.RENDER) {
    dirty.RENDER = false;
    scheduleRender();           // rAF-deduped
  }
}
```

All precondition checks live here — nowhere else.

---

## hasWaypointSource()

Mode-aware guard. Draw mode needs a drawn path; snap/code mode can start from a single pin.

```js
function hasWaypointSource() {
  if (pathMode === 'draw') return userTrajectory.length >= 2;
  return (canvasMode === 'pins' && pinPoints.length >= 1) || userTrajectory.length >= 2;
}
```

---

## scheduleRender() — rAF dedup

```js
let renderPending = false;

function scheduleRender() {
  if (renderPending) return;
  renderPending = true;
  requestAnimationFrame(() => { renderPending = false; renderScatter(); });
}
```

All former direct `renderScatter()` call sites go through `scheduleRender()`. At most one canvas draw fires per animation frame.

---

## Projection — UMAP and PCA

`projectionMode` (`'umap'` | `'pca'` | `'wheel'`) controls which algorithm runs in the web worker.

- **UMAP**: standard UMAP via `umap-js`. Six slots with distinct seeds (`mulberry32`) for reproducible but visually varied layouts.
- **PCA**: two-component PCA via power iteration (covariance matrix, O(N·D²)). Deterministic, no seeds. Runs in the same worker on `compute_pca` message.
- **Wheel**: polar controller layout — θ = `wheelAngle` source **rank** wrapped to [0, 2π) (rank, not value, for even angular spread), r = `wheelRadius` ∈ {loudness (48k scales or raw-audio frame RMS), time, atypicality (z-scored feature norm, `fd.featNorm`)}, mapped to [0.18, 1] per `wheelScale`: `'value'` = proportional with a 5–95% percentile clip (outlier guard; Time skips the clip — its values are uniform by construction), `'rank'` = rank-normalised (outlier-immune, even fill, but radial distance no longer proportional to the value).
  - **Angle sources**: `'umap1d'` (default) — a 1-component UMAP fit (`compute_umap_1d` worker message); neighbourhood-preserving, so timbre families stay angularly contiguous; stochastic. `'pc1'` — the top PCA component via `compute_pca`; deterministic but orders by one linear axis only.
  - Both angle sources cache on `fd.pc1` / `fd.umap1d`; switching angle or radius reuses the cache with no worker round-trip (a missing source computes once). The `pca_result` / `umap1d_result` handlers route to `applyWheelCoords()` only when the mode AND the matching `wheelAngle` are active.
  - Drawn orbits make seamless loops (no pingpong needed). Caveat by design: screen proximity across the centre does NOT mean latent similarity — it's a controller, not a map. UMAP-only rows (neighbours, min-dist, pre-PCA, metric, dedup) hide in wheel mode; Angle and Radius rows show.

Both produce `Float32Array` of shape `[N*2]` (interleaved x,y). The result is split into `coordsA` (first T_A rows) and `coordsB` (last T_B rows) and stored identically regardless of algorithm.

`triggerProjection()` dispatches to the correct algorithm based on `projectionMode`. On PCA result, a single `coordsA/coordsB` pair is set (no slots). On UMAP result, the current slot is populated; slots rotate on recompute.

### Feature build (`buildUMAPFeatureData`)

Per-frame feature vectors are built in a fixed order; the order matters.

```
1. base frames     — embEnc / embQuant / codebook-lookup per umapMode + embSpace
2. embMode         — 'emb' | 'emb+delta' | 'delta'; deltas are direction-aware:
                     past → f[t]−f[t−1], centered → (f[t+1]−f[t−1])/2, future → f[t+1]−f[t]
2b. appendScaleDim — optional log(scale) loudness dim (scaleFeatEnabled, 48k only);
                     weighted AFTER z-score to carry ~10% of total feature variance
                     (one unit-variance dim among D would be negligible)
3. zscoreFrames    — per-dim z-score, jointly over A and B
4. applyContextWindow — K slots placed by ctxDirection:
                     past     [t−K+1 … t]            (anchor last)
                     centered [t−⌊K/2⌋ … t+⌈K/2⌉−1]  (anchor middle)
                     future   [t … t+K−1]            (anchor first)
                     per-slot weights by ctxShape (flat / triangular / gaussian σ=K/3),
                     anchor weight 1.0; edges clamp (repeat first/last frame)
```

**Why z-score before windowing:** per-dim z-scoring after windowing would normalise each slot dimension to unit variance, exactly cancelling any constant per-slot weight. Normalising the base frames first lets the window weights survive into pairwise distances. The anchor frame appears exactly once in the window (no separate anchor prepend).

### Pre-PCA reduction (`reduce_features`)

When `preReduceEnabled` (default on, UMAP mode only) and `D > PRE_REDUCE_KMAX` (32), the feature matrix is PCA-reduced in the worker before any UMAP slot runs:

- Randomized SVD (range finding + one power iteration, Jacobi eigen on the small Gram matrix). Never materialises the centered matrix or a D×D covariance; deterministic (fixed seed).
- Keeps the smallest m components reaching `PRE_REDUCE_VAR` (95%) cumulative variance, clamped to [2, 32].
- Runs on the UMAP-side matrix (`fd.umapFlat`, i.e. after optional dedup); on `reduce_result`, `fd.umapFlat/umapD` are replaced (`fd.reduced` marks it). `fd.flat` is never touched — `launchPCA` always projects the full matrix. `#preReduceInfo` shows `D_in→m dims · NN% var`.
- Why: per-dim z-scoring amplifies noise dims to equal weight; dropping the variance tail makes distances meaningful and umap-js several times faster at large context windows.

### Dedup (`dedupFeatures`)

When `dedupEnabled` (UMAP mode only, default off), runs of consecutive frames whose cosine distance to the previous KEPT frame is below `#dedupThresh` are collapsed per source (a B frame never merges into an A run) **before** pre-PCA reduction and the UMAP fit. Why: adjacent frames of one source are nearly identical, so the KNN graph fills with trivial temporal neighbours and clusters form by source-and-time rather than content.

Bookkeeping: `fd.flat/T_A/T_B/D` (full matrix) are never touched — PCA mode always projects the full matrix. The UMAP path uses `fd.umapFlat/umapTA/umapTB/umapD` (reps, then reduced). `fd.repMapA/repMapB` map every full frame to its representative's global rep row; the `umap_result` handler expands worker coords back to full length, so `coordsA/coordsB` always have one entry per frame (skipped frames overlap their representative on the map). `nNeighbors` is re-clamped to the rep count in `launchUMAPSlot`. `#dedupInfo` shows `N→M pts`.

### UMAP metric + PCA-seeded init

- `umapMetric` (`'euclidean'` default | `'cosine'`) — cosine passes a `distanceFn` to umap-js; discounts overall energy differences, groups timbre better.
- When pre-reduction ran, columns 0–1 of the reduced matrix (the top-2 PCA scores) seed every slot's UMAP init: `umap.embedding` is mutated **in place** after `initializeFit` (the optimizer holds a reference to the same array — replacing it would break the link), scaled to umap-js's ±10 init range plus tiny per-slot jitter. Layouts become more stable across slots and globally more honest; slot variety comes from per-seed negative sampling. Seeding is best-effort (try/catch) — without reduction, slots use random init as before.

The window is non-causal by design — the map is offline analysis, so causality is irrelevant for layout even on the causal 24k model. Generation context is a separate, model-aware mechanism (see "Grain context — `ctxMode`"): causal pre-roll on 24k, symmetric pre+post on 48k.

---

## Source visibility

Two boolean flags (`srcAEnabled`, `srcBEnabled`) gate whether each source participates in rendering and generation. The UI shows small colored toggle buttons ("A" / "B") in the waveform header row for each source.

```js
function getEnabledSrcs() {
  const s = [];
  if (srcAEnabled) s.push('A');
  if (srcBEnabled) s.push('B');
  return s.length > 0 ? s : ['A', 'B']; // safety: never disable both
}
```

**Rendering**: `scatter-renderer.js` receives `srcAEnabled`/`srcBEnabled` and skips `drawTrail` / `drawCirclesEmpty` / `drawTrianglesEmpty` for disabled sources. The UMAP layout (coordinates) is unchanged — only the visual layer is hidden.

**Snapping**: `findNearestUmapPoint(x, y, coordsA, coordsB, enabledSrcs)` accepts `enabledSrcs` and skips the search loop for disabled sources. All three call sites inside `computeSnapWaypoints` pass `enabledSrcs`, covering: pins branch, multi-segment branch, and single-segment branch.

**Draw mode playback**: `playTrajectory`'s two-nearest-neighbour interpolation loop is guarded by `if (srcAEnabled)` / `if (srcBEnabled)`.

**Pin placement/drag**: `findNearestUmapPoint` and the double-click nearest-point loop both pass `getEnabledSrcs()`, so newly placed/dragged pins snap only to enabled source frames.

**Pin remap on disable**: `pinPoints` stores `{src, idx}` (original frame identity, never modified). When a source is disabled, `computeSnapWaypoints`'s pins branch re-snaps each pin's UMAP coordinate to the nearest enabled source. Re-enabling restores the original snap automatically on next `computeWaypoints()`.

---

## Multi-segment draw paths

The user can draw multiple independent strokes. Each completed stroke is pushed to `rawDrawSegments` as `{pts:[x,y][], enabled:bool}`. The in-progress stroke lives in `rawDrawPoints`.

```
rawDrawSegments = [{pts, enabled}, ...]   ← completed
rawDrawPoints   = [[x,y], ...]            ← live stroke (reset on mousedown)
```

**`buildDrawUserTrajectory()`** filters enabled segments, arc-length resamples each one proportionally to its length, and concatenates. It also populates `drawSegmentBoundaries` — indices in the resulting `userTrajectory` where each non-first segment begins.

**`computeSnapWaypoints`** receives `drawSegmentBoundaries`. When present, it samples each segment independently (proportional frame share) and records `segBoundaries` in the result. The renderer uses `segBoundaries` (stored as `snapSegBoundaries`) to insert `moveTo` (instead of `lineTo`) at each segment gap in the snap overlay, so no visual line connects unrelated strokes.

**Pingpong `_wp` offset**: The backward half of a pingpong sequence has its `_wp` tags offset by M (the number of forward waypoints). This lets the renderer correctly look up `snapFrames[M..2M-1]` (the reversed waypoints) rather than accidentally reusing `snapFrames[0..M-1]` (the forward waypoints) during backward playback.

**Right-click context menu** on draw segments:
- Hit-test via point-to-segment distance in canvas physical pixels (threshold: 12 CSS px).
- Menu items: "Disable/Enable" (toggles `enabled` flag, dims segment to alpha 0.22) and "Delete" (splices from `rawDrawSegments`).
- Both actions rebuild `userTrajectory` and call `autoRegenerate()`.

**Segment colors** in renderer:
- Enabled: `#ffffff`, `globalAlpha 0.80`
- Disabled: `#ffffff`, `globalAlpha 0.22`
- Segment start dots (non-first segments): `globalAlpha 0.85` / `0.25`

**Mouse button guard**: `mousedown` in draw mode checks `e.button === 0` so right-click never starts a spurious draw stroke.

---

## computeWaypoints() — waypoints computed once

Runs synchronously when `WAYPOINTS` is dirty. All three path modes share the same function.

### Code mode

```js
const N_total = codeDurMode === 'frames'
  ? Math.max(2, codeKVal)
  : Math.max(2, Math.round(codeSecVal * fps));
const { wps: fwdWPs } = computeSnapWaypoints(snapWaypointArgs());
const bwdWPs = [...fwdWPs].reverse();
const orderedWPs = trajDir === 'forward'  ? fwdWPs
                 : trajDir === 'backward' ? bwdWPs
                 : [...fwdWPs, ...bwdWPs];
const K = Math.max(1, Math.round(N_total / orderedWPs.length));
const sampledWPs = orderedWPs.flatMap(wp => Array.from({ length: K }, () => wp));
codePathFrames = sampledWPs;
genPlaySeq     = sampledWPs;
```

K-per-waypoint expansion via `flatMap`. Pingpong automatically halves K because `orderedWPs` is 2× the forward length.

### Snap mode

```js
const { wps: fwdWPs, segBoundaries: fwdSegBounds } = computeSnapWaypoints(snapWaypointArgs());
const bwdWPs = [...fwdWPs].reverse();
const M = fwdWPs.length;
const bwdSegBounds = fwdSegBounds.map(b => M - b).reverse();

// snapFrames and snapSegBoundaries set per trajDir
// pingpong: snapFrames = [...fwdWPs, ...bwdWPs], segBoundaries merged with bwd shifted by M

const buildSeq = (wps, N_target, kOverride, ctxArg) =>
  snapMode === 'k_frames'
    ? buildSnapSequence(wps, kOverride, K_post, ..., ctxMode)
    : buildSnapSequenceFixed(wps, Math.max(wps.length, N_target), K_post, ..., ctxArg ?? snapCtxArg(), ctxMode);

// pingpong: bwdSeq._wp tags offset by M so renderer indexes snapFrames[M..2M-1]
seq = [...fwdSeq, ...bwdSeq.map(f => ({ ...f, _wp: f._wp + M }))];
snapPlayFrames = seq;
genPlaySeq     = seq;
```

**k_frames pingpong — ceil/floor split:** `kFwd = ceil(K/2)`, `kBwd = floor(K/2)`. Every K value maps to a unique pair.

**equal/prop pingpong — halfCtxArg:** `snapDurVal / 2` passed so `getContextKs` budgets correctly for each half.

### Draw mode

No-op in `computeWaypoints()`. `playTrajectory()` interpolates embeddings inline at decode time using the two nearest UMAP frames (weighted by inverse distance). Only enabled sources are searched.

---

## snapWaypointArgs()

```js
function snapWaypointArgs() {
  return {
    canvasMode, pinPoints: pinCoords(), coordsA, coordsB,
    userTrajectory, pathMode, snapMode, snapDurVal, codeSecVal,
    trajDuration, framesPerSec,
    drawSegmentBoundaries,
    enabledSrcs: getEnabledSrcs(),
  };
}
```

`pinCoords()` maps `pinPoints[{src,idx}]` to live UMAP coords. `enabledSrcs` filters which source frames are valid snap targets.

---

## Grain context — `ctxMode`, `walkBackward` / `walkForward`

Per-waypoint context shape is model-aware (`ctxMode`, derived from `mp.getModelHz()` in `computeWaypoints` and passed to both sequence builders and the renderer):

- **`'pre'` (24k, causal decoder)** — pure pre-roll: K frames of `walkBackward` context lead into each waypoint (decoder warm-up). The old tail-only `K_post` applies (0 for 24k in practice).
- **`'sym'` (48k, non-causal decoder)** — each waypoint gets `⌈K/2⌉` pre (`walkBackward`) + waypoint + `⌊K/2⌋` post (`walkForward`). The 48k decoder renders each chunk non-causally, so the audio *at* the waypoint is shaped by embeddings on both sides — symmetric context keeps the grain's identity intact instead of giving it an unrelated right context. The tail-only `K_post` is ignored: the last waypoint's post-roll serves that role. Per-waypoint frame count is K + 1 in both modes, so `getContextKs` duration budgets are unchanged.

The continuity shortcut (consecutive same-source waypoints) accounts for the previous waypoint's post-roll: it walks from `prev.idx + kF + 1` to the next waypoint. `pingPongExtendFwd` mirrors `pingPongExtend` for forward walks (pads at the far end).

`walkForward` mirrors `walkBackward` including the context-swap branch — a swap in the post-roll replaces everything *newer* than the branch with the chosen frame's temporal post-roll, morphing the grain's **release** rather than its attack.

## Context swap — `walkBackward`

Builds the pre-roll context (K frames) that leads into each snap waypoint. Two parameters control behavior:

- **`ctxSwapProb`** (0–1 slider) — probability of attempting a swap at each step
- **`ctxSwapDist`** (number input, UMAP units) — radius for the nearby-frame candidate search

### Algorithm

The decision is **per-waypoint**, not per-step:

1. **Pure temporal walk** — walk K steps backward: `[A33, A34, A35, A36, A37, A38, A39]` → waypoint A40.

2. **Waypoint swap decision** — roll `ctxSwapProb` once. If it doesn't fire, return the pure temporal walk unchanged.

3. **Random branch point** — pick a uniformly random position in the pre-roll (e.g. A36 at index 3).

4. **Find nearby frames** — collect all frames from **both sources** within `ctxSwapDist` UMAP units of the branch frame, excluding the branch frame itself.  If none exist, return the pure temporal walk unchanged.

5. **Pick a candidate randomly** (e.g. B78) and walk its temporal pre-roll backward to fill the older portion (3 steps → B75, B76, B77).

6. **Reassemble**: `[B75, B76, B77, B78 | A36, A37, A38, A39]` → waypoint A40.
   - Chosen frame B78 is inserted at the branch position.
   - Everything newer than the branch (A36, A37…) stays unchanged.
   - Everything older than the branch is replaced by the chosen frame's temporal pre-roll.

```
temporal walk:  A33  A34  A35 [A36] A37  A38  A39  →  A40
                               ↓ branch
chosen B78:     B75  B76  B77  B78
reassembled:    B75  B76  B77  B78  A37  A38  A39  →  A40
```

**Key properties:**
- Swap decision is per-waypoint — at most one branch point per pre-roll.
- Pure temporal walk on both sides of the branch.
- Swap includes same-source candidates (can jump to a different position within the same source).
- At `ctxSwapProb = 0` or `ctxSwapDist = 0` swap is fully disabled.
- If no candidate exists within range at the branch point, the pure temporal walk is returned unchanged.

`pingPongExtend` pads the result if the walk terminates early (cIdx reached 0), bouncing the short sequence to fill exactly K slots.

---

## Post-processing — fade and volume

Generated audio is post-processed before being handed to `playerG`:

```js
// Stored raw (unfaded) PCM for live re-apply
let lastRawPcm   = null;
let lastRawPcmSr = 44100;

function applyFade(pcm, sr, fadeInMs, fadeOutMs) {
  const out = pcm.slice();
  const fadeIn  = Math.round(fadeInMs  / 1000 * sr);
  const fadeOut = Math.round(fadeOutMs / 1000 * sr);
  for (let i = 0; i < Math.min(fadeIn,  out.length); i++) out[i]                  *= i / fadeIn;
  for (let i = 0; i < Math.min(fadeOut, out.length); i++) out[out.length - 1 - i] *= i / fadeOut;
  return out;
}

function applyGenFade(pcm, sr) {
  const fadeIn  = parseFloat(fadeInMsEl.value)  || 0;
  const fadeOut = parseFloat(fadeOutMsEl.value) || 0;
  return (fadeIn > 0 || fadeOut > 0) ? applyFade(pcm, sr, fadeIn, fadeOut) : pcm;
}
```

**Fade inputs** (`fadeInMs`, `fadeOutMs`): live in the Generated waveform header (default 10 ms each). On `input` event, re-applies `applyGenFade(lastRawPcm, lastRawPcmSr)` and calls `playerG.setAudio()` — no re-generation needed.

**Volume slider** (`genVolSlider`): 0–2× range, default 1.0. Calls `playerG.setVolume(v)` which uses `gainNode.gain.setTargetAtTime` for smooth real-time adjustment and stores the value for new playbacks.

---

## playTrajectory() — reads from state

The code/snap branches read `genPlaySeq` (already set by `computeWaypoints()`), extract embeddings column-wise into a `Float32Array`, and call `decodeTrajectory()` — which dispatches to the parallel pool (`runParallelDecode48k`) on 48k-stateless, or the primary worker otherwise (see "Parallel decode" above). The draw branch interpolates between the two nearest enabled-source frames inline. The decoded buffer is handed to `onTrajectoryDecoded`, which stores `lastRawPcm` and applies `applyGenFade` before `playerG.setAudio`.

**48k loudness** (`runParallelDecode48k` on the main thread, or the worker's `runDecodeTrajectoryExp6`): when `frameScales` are supplied, every chunk decodes at scale = 1 and a per-SAMPLE gain envelope (linear interp between per-frame scales) is applied after OLA. Scale is a linear post-multiply in the decoder graph, so this is exact — and it removes the loudness steps the old per-chunk mean scale produced when loud and quiet grains mixed within one 150-frame chunk. Without `frameScales` the old per-segment fallback scale applies.

## Channels — 48k stereo, 24k mono

The 48k model is natively stereo (C=2, joint embedding for the pair); the 24k model is mono. `startEncode` sets `srcCh = modelHz === '48k' ? 2 : 1` and the whole audio path is channel-aware with **planar layout** (`Float32Array(ch*T)`, L plane then R plane):

- **Load** (`loadFile` / `buildAudioAuto` with `channels`): files keep real L/R (mono files upmix to both planes); synth sources duplicate to both planes. Per-channel volume/gate (`applyPulseGate` restarts its phase per plane).
- **Encode** (`encodeAndCapture48k` / `encodeCapture` with `channels`): segment chunks are sliced per plane and fed as real `[1, 2, T]` — no more dual-mono duplication. One embedding per frame still describes both channels (incl. the stereo image), so the map / grains / UMAP pipeline are untouched.
- **Decode** (`decodeFromLatents(..., stereoOut)` / `runDecodeTrajectoryExp6`): stereo OLA with planar accumulators (mono weight buffer — the triangular window is channel-independent); the loudness envelope applies to both planes. `trajectory_audio_exp6` / `frame_audio_exp6` messages carry `channels`.
- **Playback & post**: `wave-player.setAudio(..., channels)` (peaks span both planes, real stereo `AudioBuffer`), `applyFade` / `applyNotchOffline` / `encodeWav` are planar-aware; bookmarks store `ch` and download as stereo WAVs.
- **Sidecars**: codes/scales are stored planar-aware; `channels` is part of the sidecar header identity so a stereo (48k) entry never matches a mono (24k) one.
- **Consequence**: grain concatenation splices stereo images too — two timbrally similar grains with opposite panning are *different* latents and may sit apart on the map. Panning is part of the granular texture now.

`durA`/`durB` and all FPS math use frames (`audio.length / srcCh / sr`). The 24k path is byte-identical to before (channels = 1 defaults everywhere).

---

## Encode performance — `skipPreview`, `statefulOLA`

Two encode-message flags, both 48k-relevant:

- **`skipPreview`** (exp6 always sends `true`): the per-segment `decodeFromLatents` in `encodeAndCapture48k`/`24k` produces a decoded *preview* (`outBuf`) that exp6's `'result'` handler discards — only the captures (codes/embQuant/scales) are used. Skipping it removes a full decoder conv stack per segment plus the decoder LSTM chain: **measured 1.86× faster** 48k encode (the decode was ~46% of encode time). Default `false`, so experiments 2–5 keep their preview.

- **`statefulOLA`** (UI toggle "48k OLA", shown only on 48k, default `false` = "Stateless"): controls whether the encoder/decoder LSTM state is carried across the 1 s OLA segments (`if (statefulOLA) { hEnc = cap.hEncNew; ... }`) or reset to zero per segment. Stateful = the fork's cross-boundary continuity (sequential); **Stateless** (default) makes segments independent and closer to stock EnCodec, at the cost of faint boundary seams (guard frames still warm the conv stack). It changes the embeddings — chunk 0 is identical between modes (both start zero-state), divergence begins after the first segment boundary (measured ~2% on later frames). Toggling re-encodes (`currentEmbeds = null; scheduleEncode`). The OLA mode is recorded in the sidecar header (`ola: 'stateful' | 'stateless'`) and validated on read, so stateful and stateless codes never collide; the whole-file folder path (which is stateless-only) and the per-window sidecars both cache correctly.

### Parallel encode (48k stateless)

Because stateless segments are independent, the 1 s chunks encode concurrently across a **persistent worker pool** (`ensureEncPool`, K = min(cores, 6) workers each with their own ONNX sessions). Flow:

1. `startEncode` branches to `runParallelEncode48k` when `modelHz === '48k' && !statefulOla`.
2. `sliceSegments48k` cuts each source's planar audio into the exact same guarded chunks as `encodeAndCapture48k` (must match — same geometry constants `SEG_48_M/STRIDE_48_M/HOP_48_M/GUARD_48_M`).
3. All A+B segment chunks dispatch across the pool (`doParallelBatch`); each pool worker runs `encode_segment` → `encodeCapture` with zero state and returns the cap (chunk transferred in).
4. Main thread reassembles per-source `{mode:'ola48', totalLen, segments}`, ships them to the **primary** worker via `set_captures`, then sends the usual `get_embeddings_exp6`. The caps arrive pre-computed.

Batches are serialised on a shared pool lock (`encodeBatchPromise` / `withPoolLock`) so encode/decode/clip-encode batches never overlap (no stale-slot reuse); a superseded run is discarded via the `myRun`/`encodeRunSeq` guard before results are applied; a segment error rebuilds the pool clean. **Output is bit-identical to the sequential stateless path** (same chunks, zero state each — verified: early/late embedding checksums match to full float precision). Measured: clean sequential 2-source encode 8.1 s → parallel end-to-end ~3.0 s.

The same pool also encodes **whole folder clips** (`encodeClipFullCodes`, see "Folder sources") and **decodes trajectories** in parallel.

24k and 48k-stateful keep the sequential primary-worker path.

### Parallel decode + explore playback (48k stateless)

Trajectory generation decode mirrors the parallel encode: `decodeTrajectory` routes 48k-stateless to `runParallelDecode48k`, which slices the trajectory `embQuant` into 150-frame / 135-stride chunks, decodes them concurrently across the pool (`decode_segment` worker message → stateless `decodeFromLatents`), and **OLA-blends on the main thread** (same triangular window + per-sample loudness envelope as the sequential decoder). The scale tensor shape comes from `get_decode_scale` (cached caps, or the `[1,1]` unit fallback). The whole generation is handed to `playerG` once all chunks are in (`onTrajectoryDecoded`). 24k / 48k-stateful still decode via the primary worker (`runDecodeTrajectoryExp6`).

**Explore-click playback** (`play_frame_exp6`) sends the codes window **straight from `currentEmbeds`** (always populated) rather than relying on the worker's `captureCache` — so it works for sources resolved from cache/sidecars (which have no captures). The worker decodes from the supplied codes (+ the clicked frame's scale on 48k), falling back to `captureCache` only when no codes are sent.

---

## autoRegenerate()

```js
function autoRegenerate() {
  if (userTrajectory.length >= 2 && (!currentEmbeds || !coordsA || !coordsB)) {
    setStatus('Path drawn. Encode sources and run UMAP to play.');
  }
  markDirty('WAYPOINTS', 'AUDIO', 'RENDER');
}
```

Called from: trajectory draws/drags (mouseUp), pin add/remove/drag/reorder, all path param changes, source visibility toggles, canvas mode switches.

---

## Waveform color coding

The generated waveform is color-coded per pixel by the source of the corresponding frame:
- Source A frames → orange (`#c87800` family)
- Source B frames → blue (`#2090c0` family)
- Draw mode (interpolated) → `null` (uses default waveform color)

`makeGenColorFn()` builds a `(pos: 0..1) → CSS color` closure over `genPlaySeq`. It is passed to `playerG.setAudio()` as the third argument and re-passed whenever fade is re-applied.

**Folder per-clip colors.** Folder sources tint each clip distinctly (waveform + scatter points) via `clipColor(k, n, source)` over `folderClipFracsA/B`. Each source uses its own hue **band** so A and B stay distinguishable with many files: **A = warm** (hue ≈ 18–110, orange→yellow→green), **B = cool** (hue ≈ 172–294, cyan→blue→violet); within a band the hue ramps across clips. `makeGenColorFn` reuses these per-frame point colors (`pointColorsA/B`) so a folder clip's color carries through to the generated signal.

---

## Full trigger chains

```
SOURCE / MODEL CHANGE
  └─► scheduleEncode() [debounced]
        └─► startEncode()
              ├─ buildAudioAuto(panelA/B, sr) — loads each source (folder → windowed display audio
              │     + folderSelections); sets durA / durB = audio.length / srcCh / sr
              ├─ resolveSourceCodes(panelA) then (panelB)  — sequential, see "Per-source resolution"
              │     48k folder → encodeFolderSampledCodes (full-clip encode + slice; always resolves)
              │     24k/stateful folder → readFolderSidecars (per-window cache; null on miss)
              │     file / synth → null
              ├─ BOTH resolved  → injectCodesPair → currentEmbeds, skip worker entirely
              │                   → markDirty('COORDS','WAYPOINTS','AUDIO','RENDER')
              ├─ ONE/NONE       → resolved source gets a dummyEncodeAudio placeholder; the worker
              │                   encodes the rest: encode A → encode B → get_embeddings_exp6
              │     on embeddings_exp6 (onWorkerMsg): merge any pendingResolvedA/B, then write any
              │     pendingSidecar (24k/stateful per-window). 48k whole-file writes 0-full inline.
              └─► markDirty('COORDS', 'WAYPOINTS', 'AUDIO', 'RENDER')
                    └─► runEffects()
                          └─► triggerProjection()
                                ├─ UMAP mode: startUMAPReduction()
                                │    [dedup (sync) → reduce_features → reduce_result → launchUMAPSlot(0)]
                                └─► onUMAPWorkerMsg (umap_result / pca_result)
                                      └─► markDirty('WAYPOINTS', 'AUDIO', 'RENDER')
                                            └─► runEffects()
                                                  ├─► computeWaypoints()
                                                  ├─► playTrajectory() → worker decode
                                                  │     └─► playerG.setAudio(applyGenFade(raw))
                                                  └─► scheduleRender()
```

```
CANVAS INTERACTION (draw / pin / shift-drag)
  └─► live: scheduleRender() [shows path while dragging]
  └─► mouseUp: autoRegenerate()
                 └─► markDirty('WAYPOINTS', 'AUDIO', 'RENDER')
                       └─► runEffects() [same cascade as above]
```

```
SOURCE VISIBILITY TOGGLE
  └─► srcAEnabled / srcBEnabled updated
  └─► scheduleRender()     [dots hide immediately]
  └─► autoRegenerate()     [refit curve + regenerate audio]
```

```
FADE INPUT CHANGE
  └─► applyGenFade(lastRawPcm) → playerG.setAudio()  [no re-generation]
```

```
VOLUME SLIDER
  └─► playerG.setVolume(v)  [immediate gain ramp, no re-generation]
```

```
PAN / ZOOM / PLAYBACK TICK
  └─► scheduleRender()  [rAF-deduped, no upstream work]
```

---

## Folder sources, code sidecars, and whole-file encoding

A source can be a **folder** ("Folder (join WAVs)") whose clips are concatenated into one source.
Any browser-decodable container is accepted (wav/wave/mp3/m4a/m4b/mp4/aac/flac/ogg/opus/aif/aiff/
aifc/caf/webm), name-sorted. *(This replaces the old opt-in IndexedDB analysis cache, which is
removed — `lib/encodec-cache.js` is deleted; its `crc32`/`crc32FromFile` moved into
`encodec-sidecar.js`.)*

### Length budget + sampling — `planFolderSelections`
The joined source is capped at `MAX_FOLDER_TOTAL_S = 600` (10 min). Durations are probed cheaply
(`probeAudioDurationSeconds`: WAV header → media-element metadata → full decode as last resort).
When the sum exceeds the budget, an overflow modal (`askFolderStrategy`) picks a per-clip **sample
window** `{file, startS, lenS}`:

- `first` — keep whole clips in name order until the budget runs out
- `trim` — first 600/N s of every clip
- `random` / `random:K` — a random window of every clip (K ∈ {2, 5, 10} s, or 600/N for the equal share)
- `truncate` — concatenate and hard-cut at 600 s
- `all` — load past the budget anyway

Random offsets are **deterministic**, seeded from `salt:name:size:index`; `salt` is the source
('A'/'B'), so the SAME folder loaded into both A and B samples *different* windows, yet each is
reproducible across reloads (cache-stable). The modal picks the SAMPLE window, not what is encoded
(48k always encodes full clips — see below).

### Whole-file encoding + code sampling (48k stateless) — `encodeFolderSampledCodes`
`resolveSourceCodes` for a 48k-stateless folder **encodes each clip in full once** and samples
windows by **slicing codes** (not audio):

- `getClipFullCodes(file)` → in-memory `_fullCodesCache` → the clip's `0-full` sidecar entry →
  else `encodeClipFullCodes` (load full clip, pool-encode its 1 s segments via `doParallelBatch`
  under `withPoolLock`, reassemble the full code sequence on the main thread with
  `reassembleClipCodes` — a mirror of the worker's `flattenCaps`/`flattenScales`, codes + scales only).
- Sample: `sliceClipCodes(full, round(startS·fps), round((startS+lenS)·fps))` (fps = 150) per clip,
  then `joinClipCodes`. `embQuant` is reconstructed from codes downstream (`reconstructEmbQuant`).

Consequence: re-rolling a window, the same folder in A vs B, a model revisit, or a reload are all
**free code-slices** — no re-encode. First load encodes everything (cached after), processed
per-clip so memory stays bounded. Codes carry full-clip context (cleaner than the old
windowed-concat encode). 24k / 48k-stateful folders keep the older per-window path
(`readFolderSidecars` → worker encode on miss → `writeFolderSidecars` per window).

### Code sidecars — `lib/encodec-sidecar.js`
Codes are persisted in a file **next to each clip** so reloads (and other tools, e.g. a Max/MSP
patch) can reuse them. One sidecar per (wav × model × bandwidth):

    <clip>.encodec.<hz>.<bw>      e.g.  break01.wav.encodec.48.24

Binary: `"ENCSIDE1"` magic + `uint32 headerLen` + JSON header + raw Int16 codes + Float32 scales.
The header carries `filename/fileSize/crc32/modelHz/bwKbps/sampleRate/channels/ola` (all validated
on read via `sidecarMatches`) and one or more **window entries** keyed `<startMs>-<lenMs>` (`0-full`
for the whole-file workflow). Writing needs a writable `FileSystemDirectoryHandle`, so sidecars
work for **folder sources in Chromium only** (`showDirectoryPicker({ mode: 'readwrite' })` at pick
time). `FS_ACCESS_OK` is feature-detected; unsupported browsers (Brave **default** — gated behind
`brave://flags/#file-system-access-api`; Firefox/Safari) get a one-shot "needs Chrome/Edge" notice
and fall back to the in-memory `_fullCodesCache` (re-sampling still free within a session). The save
outcome is shown in a persistent `#noticeModal`. Single FILE sources are not cached (no writable
parent directory).

### Per-source resolution — `resolveSourceCodes`
A and B resolve **independently** (any type combination):

- 48k folder → `encodeFolderSampledCodes` (always resolves — it encodes/samples itself)
- 24k / 48k-stateful folder → `readFolderSidecars` (per-window cache; `null` → worker encode)
- file / synth → `null` (encoded via the worker)

Both resolved → `injectCodesPair` assembles `currentEmbeds` and **skips the worker entirely**. When
exactly **one** source resolves (e.g. a cached folder + a Sine), the resolved source's real encode
is **skipped** via a silent placeholder (`dummyEncodeAudio`); the worker encodes only the other
source, and `onWorkerMsg` (now async) merges the resolved codes back, rebuilding `embQuant`.

---

## Key design properties

1. **Guards in one place** — `runEffects()` is the only location that checks `currentEmbeds`, `coordsA`, `hasWaypointSource()`, etc.
2. **Mode-aware source guard** — `hasWaypointSource()` allows snap/code to generate from a single pin; draw mode still requires `userTrajectory.length >= 2`.
3. **Waypoints computed once** — `computeWaypoints()` runs once per change; `playTrajectory()` consumes `genPlaySeq` from state.
4. **Levels are independent** — a pan gesture only sets `RENDER` dirty. A `trajDir` change sets `WAYPOINTS + AUDIO + RENDER`. A UMAP param change sets `COORDS + WAYPOINTS + AUDIO + RENDER`.
5. **One draw per frame** — `scheduleRender()` rAF guard prevents redundant redraws.
6. **Pins survive UMAP/PCA changes** — `pinPoints` stores `{src, idx}` (frame identity). `pinCoords()` derives `[x,y]` live from `coordsA`/`coordsB`, so pins auto-remap when projection changes. Draw trajectories are cleared on projection result since they have no frame-index backing.
7. **Pingpong same duration** — code mode: K auto-halves via 2× waypoint list. Snap k_frames: ceil/floor split. Snap equal/prop: half-duration `snapCtxArg` keeps all waypoints within each half.
8. **Per-source durations** — `buildAudioAuto` loads each file at its own natural length (or trimmed by panel setting); A and B need not be the same length. `durA` / `durB` track the actual loaded durations. All FPS calculations use `srcDuration(src)` which returns `durA`/`durB` with a `defaultDur` fallback before first encode. The Duration panel is not shown in experiment 6 (`showDuration: false` in `createDoubleSourceWidget`).
9. **Per-source resolution + whole-file folder caching** — A and B resolve independently (`resolveSourceCodes`). 48k folders encode each clip in full once (cached as `0-full` sidecars next to the wav + in-memory `_fullCodesCache`) and sample windows by slicing codes, so re-rolling a window / A-vs-B / reload are free. A source resolved from cache skips its real encode (placeholder + merge). Replaces the old opt-in IndexedDB analysis cache (deleted).
10. **Multi-segment independence** — each draw stroke is sampled independently in `computeSnapWaypoints`; the renderer inserts `moveTo` gaps so no connecting line is drawn or traversed.
11. **Source disable is non-destructive** — `pinPoints` is never modified; disabling a source re-snaps pins at `computeWaypoints()` time; re-enabling restores the original snapping automatically.
12. **Fade is post-processing** — `lastRawPcm` stores the unmodified decoded audio. Changing fade values re-applies to the stored PCM instantly without touching the WAYPOINTS/AUDIO pipeline.
13. **Normalise, then weight** — `zscoreFrames` runs on base feature frames BEFORE the context window and the loudness dim get their weights; z-scoring afterwards would normalise every weighted dim back to unit variance and cancel the weights.
14. **Model-aware grain context** — `ctxMode` is derived from the model: 24k (causal decoder) gets pure pre-roll warm-up; 48k (non-causal) gets symmetric ⌈K/2⌉ pre + ⌊K/2⌋ post per waypoint. Per-waypoint frame count is K + 1 either way, so duration budgets are mode-independent.
15. **Full vs UMAP-side matrix** — `fd.flat` (full feature matrix) is immutable after build; dedup and pre-PCA reduction only ever produce `fd.umapFlat`. PCA mode always projects the full matrix; UMAP results expand back to full length via `repMapA/B`, so `coordsA/coordsB` always have one entry per frame.
16. **Loudness is exact at decode** — 48k chunks decode at scale = 1 and a per-sample envelope (linear interp of per-frame scales) is applied after OLA. Scale is a linear post-multiply in the decoder graph, so this is mathematically identical per grain and removes per-chunk loudness steps.
