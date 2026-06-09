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

// EMBEDS (async)
currentEmbeds        // {embQuantA, embQuantB, embQuantDimsA, embQuantDimsB, scalesA, scalesB}
encodePhase          // 'A' | 'B' | null

// COORDS inputs
projectionMode       // 'umap' | 'pca'
umapMode, umapLevel, embSpace, embMode
nNeighbors, minDist, umapWindow

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
    ? buildSnapSequence(wps, kOverride, K_post, ...)
    : buildSnapSequenceFixed(wps, Math.max(wps.length, N_target), K_post, ..., ctxArg ?? snapCtxArg());

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
              ├─ worker: encode A → encode B → embeddings
              └─► markDirty('COORDS', 'WAYPOINTS', 'AUDIO', 'RENDER')
                    └─► runEffects()
                          └─► triggerProjection()
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

## Key design properties

1. **Guards in one place** — `runEffects()` is the only location that checks `currentEmbeds`, `coordsA`, `hasWaypointSource()`, etc.
2. **Mode-aware source guard** — `hasWaypointSource()` allows snap/code to generate from a single pin; draw mode still requires `userTrajectory.length >= 2`.
3. **Waypoints computed once** — `computeWaypoints()` runs once per change; `playTrajectory()` consumes `genPlaySeq` from state.
4. **Levels are independent** — a pan gesture only sets `RENDER` dirty. A `trajDir` change sets `WAYPOINTS + AUDIO + RENDER`. A UMAP param change sets `COORDS + WAYPOINTS + AUDIO + RENDER`.
5. **One draw per frame** — `scheduleRender()` rAF guard prevents redundant redraws.
6. **Pins survive UMAP/PCA changes** — `pinPoints` stores `{src, idx}` (frame identity). `pinCoords()` derives `[x,y]` live from `coordsA`/`coordsB`, so pins auto-remap when projection changes. Draw trajectories are cleared on projection result since they have no frame-index backing.
7. **Pingpong same duration** — code mode: K auto-halves via 2× waypoint list. Snap k_frames: ceil/floor split. Snap equal/prop: half-duration `snapCtxArg` keeps all waypoints within each half.
8. **Multi-segment independence** — each draw stroke is sampled independently in `computeSnapWaypoints`; the renderer inserts `moveTo` gaps so no connecting line is drawn or traversed.
9. **Source disable is non-destructive** — `pinPoints` is never modified; disabling a source re-snaps pins at `computeWaypoints()` time; re-enabling restores the original snapping automatically.
10. **Fade is post-processing** — `lastRawPcm` stores the unmodified decoded audio. Changing fade values re-applies to the stored PCM instantly without touching the WAYPOINTS/AUDIO pipeline.
