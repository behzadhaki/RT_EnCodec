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
durA, durB               // actual loaded duration (s) per source; null until first encode

// Analysis cache (IndexedDB via encodec-cache.js)
encCache             // IDBDatabase | null  — null if disabled or not yet opened
cachingEnabled       // bool — true once DB is open and user opted in
pendingCacheKeyA/B   // cache key objects held between startEncode and embeddings_exp6;
                     // null means no save is pending (synth source or cache disabled)

// EMBEDS (async)
currentEmbeds        // {embEncA/B, embEncDimsA/B, embQuantA/B, embQuantDimsA/B,
                     //  codesA/B, codesDimsA/B, scalesA/B}
                     // populated either from worker (encode path) or from cache (cache hit)
encodePhase          // 'A' | 'B' | null

// COORDS inputs
projectionMode       // 'umap' | 'pca'
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

`projectionMode` (`'umap'` | `'pca'`) controls which algorithm runs in the web worker.

- **UMAP**: standard UMAP via `umap-js`. Six slots with distinct seeds (`mulberry32`) for reproducible but visually varied layouts.
- **PCA**: two-component PCA via power iteration (covariance matrix, O(N·D²)). Deterministic, no seeds. Runs in the same worker on `compute_pca` message.

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

The code/snap branches read `genPlaySeq` (already set by `computeWaypoints()`), extract embeddings column-wise into a `Float32Array`, and dispatch to the worker. The draw branch interpolates between the two nearest enabled-source frames inline. On worker reply (`trajectory_audio_exp6`), `lastRawPcm` is stored and `applyGenFade` is applied before `playerG.setAudio`.

**48k loudness (worker, `runDecodeTrajectoryExp6`)**: when `frameScales` are supplied, every chunk decodes at scale = 1 and a per-SAMPLE gain envelope (linear interp between per-frame scales) is applied after OLA. Scale is a linear post-multiply in the decoder graph, so this is exact — and it removes the loudness steps the old per-chunk mean scale produced when loud and quiet grains mixed within one 150-frame chunk. Without `frameScales` the old per-segment fallback scale applies.

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

---

## Full trigger chains

```
SOURCE / MODEL CHANGE
  └─► scheduleEncode() [debounced]
        └─► startEncode()
              ├─ crc32FromFile(fileA/B) — CRC-32 of raw file bytes (WeakMap-cached)
              ├─ buildAudioAuto(panelA/B, sr) — loads per panel's "load full" / trim setting
              │     sets durA / durB = audio.length / sr (A and B may differ)
              ├─ [if cachingEnabled && both files] cacheGet(keyA) + cacheGet(keyB)
              │     CACHE HIT  → inject currentEmbeds directly, skip worker entirely
              │                  → markDirty('COORDS','WAYPOINTS','AUDIO','RENDER')
              │     CACHE MISS → store pendingCacheKeyA/B, fall through to worker
              ├─ worker: encode A → encode B → get_embeddings_exp6
              │     on embeddings_exp6: cachePut(A) + cachePut(B) [fire-and-forget]
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

## Analysis cache — `lib/encodec-cache.js`

Opt-in IndexedDB cache that stores per-source encoder outputs so repeated loads of the same file+model combination skip the worker entirely.

### Cache key
`(filename, fileSize, crc32, modelHz, bwKbps, sampleRate, segStart=0, segEnd)`

`crc32` is computed over the raw file `ArrayBuffer` and WeakMap-cached per File object.
`segEnd` is `audio.length` after resampling to `sampleRate`, so changing the trim setting produces a different key and correctly misses.

### Stored per record
| Field | Type | Description |
|---|---|---|
| `embEncData` | `ArrayBuffer` (Float32) | Pre-VQ encoder embeddings |
| `embEncDims` | `number[]` | Shape of embEnc |
| `embQuantData` | `ArrayBuffer` (Float32) | Post-VQ quantised embeddings |
| `embQuantDims` | `number[]` | Shape of embQuant |
| `codesData` | `ArrayBuffer` (Int32) | VQ codebook indices |
| `codesDims` | `number[]` | Shape of codes |
| `scalesData` | `ArrayBuffer` (Float32) \| `null` | Per-frame scales (48 kHz model; null for 24 kHz) |
| `timestamp` | `number` | `Date.now()` at save time |

Storing `embQuant` + `codes` alongside `embEnc` means a cache hit requires **zero additional inference** — `currentEmbeds` is assembled directly from the stored buffers.

### Preference
Stored in `localStorage` under key `encodec_cache_enabled` (`'1'` / `'0'`). A first-visit prompt in the controls sidebar lets the user opt in or out; the toggle is always visible for later changes.

### Partial hits
If only one source is a file, or if one source misses, the full encode pipeline runs for **both** sources (no partial injection). This keeps the worker's internal capture state consistent.

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
9. **Analysis cache** — `encodec-cache.js` stores per-source embeddings in IndexedDB keyed by `(filename, fileSize, crc32, modelHz, bwKbps, sampleRate, segStart, segEnd)`. A full cache hit (both sources) skips the worker entirely and injects `currentEmbeds` directly. Cache is opt-in; preference stored in `localStorage`. CRC-32 is computed from raw file bytes and WeakMap-cached per File object so large files are only hashed once per session.
10. **Multi-segment independence** — each draw stroke is sampled independently in `computeSnapWaypoints`; the renderer inserts `moveTo` gaps so no connecting line is drawn or traversed.
11. **Source disable is non-destructive** — `pinPoints` is never modified; disabling a source re-snaps pins at `computeWaypoints()` time; re-enabling restores the original snapping automatically.
12. **Fade is post-processing** — `lastRawPcm` stores the unmodified decoded audio. Changing fade values re-applies to the stored PCM instantly without touching the WAYPOINTS/AUDIO pipeline.
13. **Normalise, then weight** — `zscoreFrames` runs on base feature frames BEFORE the context window and the loudness dim get their weights; z-scoring afterwards would normalise every weighted dim back to unit variance and cancel the weights.
14. **Model-aware grain context** — `ctxMode` is derived from the model: 24k (causal decoder) gets pure pre-roll warm-up; 48k (non-causal) gets symmetric ⌈K/2⌉ pre + ⌊K/2⌋ post per waypoint. Per-waypoint frame count is K + 1 either way, so duration budgets are mode-independent.
15. **Full vs UMAP-side matrix** — `fd.flat` (full feature matrix) is immutable after build; dedup and pre-PCA reduction only ever produce `fd.umapFlat`. PCA mode always projects the full matrix; UMAP results expand back to full length via `repMapA/B`, so `coordsA/coordsB` always have one entry per frame.
16. **Loudness is exact at decode** — 48k chunks decode at scale = 1 and a per-sample envelope (linear interp of per-frame scales) is applied after OLA. Scale is a linear post-multiply in the decoder graph, so this is mathematically identical per grain and removes per-chunk loudness steps.
