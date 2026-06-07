# Experiment 6 — Architecture

## Computation levels

Everything derives from a strict cascade. When something changes at level N, only levels N+1 and below need to update.

```
SOURCES ──► EMBEDS ──► COORDS ──► WAYPOINTS ──► AUDIO
                                                   │
VIEWPORT ──────────────────────────────────────────┤
PLAYBACK_TICK ─────────────────────────────────────┤
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
umapMode, umapLevel, embSpace, embMode
nNeighbors, minDist, umapWindow

// COORDS (async)
coordsA, coordsB

// WAYPOINTS inputs
userTrajectory       // resampled path — draw mode source of truth
rawDrawPoints        // draw mode scratch (before mouseUp resample)
pinPoints            // [{src, idx}, ...] — stores frame identity, not UMAP coords
pathMode             // 'draw' | 'code' | 'snap'
trajDir              // 'forward' | 'backward' | 'pingpong'
snapMode             // 'k_frames' | 'equal' | 'proportional'
snapKVal             // K pre-roll frames per waypoint (k_frames mode)
snapDurVal           // total duration in seconds (equal/prop mode)
codeDurMode          // 'frames' | 'seconds'
codeKVal             // total frames (code/frames mode)
codeSecVal           // total seconds (code/seconds mode)
ctxSwapRange         // UMAP-unit radius for context cross-source swap

// WAYPOINTS (sync, derived by computeWaypoints())
snapFrames           // visual waypoint dots for snap mode
codePathFrames       // visual waypoint dots for code mode
snapPlayFrames       // full frame sequence with _wp tags for snap playback
genPlaySeq           // [{src,idx},...] frame sequence consumed by audio decode

// AUDIO (async)
trajAudioBuf         // decoded audio buffer

// PLAYBACK (driven by AudioContext time, render-only)
trajAnchorPos        // 0..1 playback progress, -1 = stopped
trajLooping

// RENDER-ONLY (no downstream computation)
canvasMode, viewZoom, viewPanX, viewPanY
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
| Any UMAP param change | `COORDS WAYPOINTS AUDIO RENDER` |
| UMAP coords arrive | `WAYPOINTS AUDIO RENDER` |
| userTrajectory / pinPoints change | `WAYPOINTS AUDIO RENDER` (via autoRegenerate) |
| Path params change (pathMode, trajDir, snapMode, codeDurMode, etc.) | `WAYPOINTS AUDIO RENDER` (via autoRegenerate) |
| Pan / zoom / resize | `RENDER` only |
| Waveform player cursor move | `RENDER` only (via scheduleRender) |
| Playback tick (animateTrajAnchor) | `RENDER` only (via scheduleRender) |
| Explore click (selectedPoint) | `RENDER` only |

---

## runEffects() — the single scheduler

```js
function runEffects() {
  if (dirty.COORDS && currentEmbeds) {
    dirty.COORDS = false;
    triggerUMAP();             // async → dispatches WAYPOINTS AUDIO RENDER on complete
  }
  if (dirty.WAYPOINTS && coordsA && coordsB && hasWaypointSource()) {
    dirty.WAYPOINTS = false;
    computeWaypoints();        // sync — updates snapFrames/codePathFrames/genPlaySeq
    dirty.AUDIO  = true;
    dirty.RENDER = true;
  }
  if (dirty.AUDIO && currentEmbeds && coordsA && coordsB
      && hasWaypointSource()
      && (genPlaySeq !== null || pathMode === 'draw')) {
    dirty.AUDIO = false;
    playTrajectory();          // async → worker decode → playerG.setAudio
  }
  if (dirty.RENDER) {
    dirty.RENDER = false;
    scheduleRender();          // rAF-deduped
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

All 20+ former direct `renderScatter()` call sites now go through `scheduleRender()`. At most one canvas draw fires per animation frame, regardless of how many state mutations happen synchronously.

---

## computeWaypoints() — waypoints computed once

Runs synchronously when `WAYPOINTS` is dirty. All three path modes share the same function.

### Code mode

```js
const N_total = codeDurMode === 'frames'
  ? Math.max(2, codeKVal)
  : Math.max(2, Math.round(codeSecVal * fps));
const fwdWPs = computeSnapWaypoints(snapWaypointArgs());
const bwdWPs = [...fwdWPs].reverse();
const orderedWPs = trajDir === 'forward'  ? fwdWPs
                 : trajDir === 'backward' ? bwdWPs
                 : [...fwdWPs, ...bwdWPs];   // pingpong: 2× length → K halves automatically
const K = Math.max(1, Math.round(N_total / orderedWPs.length));
const sampledWPs = orderedWPs.flatMap(wp => Array.from({ length: K }, () => wp));
codePathFrames = sampledWPs;
genPlaySeq     = sampledWPs;
```

K-per-waypoint expansion via `flatMap`. Pingpong automatically halves K because `orderedWPs` is 2× the forward length, so `N_total / orderedWPs.length = K/2`. Total duration stays constant across all three directions.

### Snap mode

```js
const fwdWPs = computeSnapWaypoints(snapWaypointArgs());
const bwdWPs = [...fwdWPs].reverse();
snapFrames   = trajDir === 'pingpong' ? [...fwdWPs, ...bwdWPs] : (trajDir === 'backward' ? bwdWPs : fwdWPs);
const N_total = Math.max(2, Math.round(snapDurVal * fps));
const K_post  = is48k ? Math.max(2, Math.round(fps * 0.03)) : 0;

// buildSeq helper — ctxArg defaults to snapCtxArg(); pingpong halves pass halfCtxArg
const buildSeq = (wps, N_target = N_total, kOverride = snapKVal, ctxArg = null) =>
  snapMode === 'k_frames'
    ? buildSnapSequence(wps, kOverride, K_post, ...)
    : buildSnapSequenceFixed(wps, Math.max(wps.length, N_target), K_post, ..., ctxArg ?? snapCtxArg());

if (trajDir === 'forward')       seq = buildSeq(fwdWPs);
else if (trajDir === 'backward') seq = buildSeq(bwdWPs);
else {
  const half       = Math.floor(N_total / 2);
  const kFwd       = Math.max(1, Math.ceil(snapKVal / 2));   // ceil
  const kBwd       = Math.max(1, Math.floor(snapKVal / 2));  // floor
  const halfCtxArg = { ...snapCtxArg(), snapDurVal: snapDurVal / 2 };
  seq = [...buildSeq(fwdWPs, half, kFwd, halfCtxArg), ...buildSeq(bwdWPs, half, kBwd, halfCtxArg)];
}
snapPlayFrames = seq;
genPlaySeq     = seq;
```

**k_frames pingpong — ceil/floor split:** `kFwd = ceil(K/2)`, `kBwd = floor(K/2)`. Every K value maps to a unique `(kFwd, kBwd)` pair, so every change to K produces audibly different audio. With plain `floor`, K=8 and K=9 both gave kHalf=4 → identical sequence.

**equal/prop pingpong — halfCtxArg:** `getContextKs` inside `buildSnapSequenceFixed` uses `snapDurVal` to budget per-waypoint context. Without halving, it over-budgets: the sequence exceeds `half` frames, truncation cuts the final waypoints from each half entirely. With `snapDurVal / 2`, K_each fits the half-length target and all waypoints are preserved.

### Draw mode

No-op in `computeWaypoints()`. The draw branch in `playTrajectory()` interpolates embeddings inline at decode time. The seqInfo just shows `'interpolated points along path'`.

---

## Sequence info display (`#seqInfo`)

A short inline summary of the generated sequence, shown in the status bar.

### Code mode — `setCodeSeqInfo(seq, kRepeat)`

Shows waypoints with `x{K}` multiplier. At most 5 points; extras skipped with `→ … →`.

```
[ A8x4 → B29x4 → B75x4 → B75x4 → A19x4 ]        ← ≤6 waypoints, no skip
[ A19x4 → … → B75x4 → B75x4 → … → A19x4 ]       ← pingpong with skip
```

Turnaround detection (`findTurnaround`) ensures the doubled midpoint is preserved in the skip display. For pingpong, `pickDisplayIndices` mirrors left picks symmetrically to the right half.

### Snap mode — `setSnapSeqInfo(snapFrames, snapPlayFrames)`

Each waypoint segment becomes `B{minIdx}:{src}{wpIdx}` where `minIdx` is the lowest frame index reached by the pre-roll (from `pingPongExtend`). If no pre-roll extends below the waypoint, just `{src}{idx}`.

```
[ B0:B6, B21:B29, A39:A47, A39:A47, B21:B29, B0:B6 ]   ← pingpong, 6 segs
[ A9 → … → A240 → A240 → … → A9 ]                       ← skip notation
```

`snapPlayFrames` frames carry a `_wp` tag (waypoint index). Segment boundaries are detected by `_wp` transitions. For pingpong, `_wp` restarts at 0 in the backward half — the junction (nFwd-1 → 0) is a natural boundary.

---

## playTrajectory() — reads from state

After the refactor, the code/snap branches just read `genPlaySeq` (already set by `computeWaypoints()`), extract embeddings, and dispatch to the worker. The draw branch still computes blended embeddings inline.

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

Called from: trajectory draws/drags (mouseUp), pin add/remove/drag/reorder, all path param changes (pathMode, trajDir, snapMode, snapParam, codeParam, codeDurMode toggle, trajDuration), canvas mode switches.

---

## Full trigger chains

```
SOURCE / MODEL CHANGE
  └─► scheduleEncode() [debounced]
        └─► startEncode()
              ├─ worker: encode A → encode B → embeddings
              └─► markDirty('COORDS', 'WAYPOINTS', 'AUDIO', 'RENDER')
                    └─► runEffects()
                          └─► triggerUMAP()
                                └─► onUMAPWorkerMsg (umap_result)
                                      └─► markDirty('WAYPOINTS', 'AUDIO', 'RENDER')
                                            └─► runEffects()
                                                  ├─► computeWaypoints()
                                                  ├─► playTrajectory() → worker decode
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
PAN / ZOOM / PLAYBACK TICK
  └─► scheduleRender()  [rAF-deduped, no upstream work]
```

---

## Key design properties

1. **Guards in one place** — `runEffects()` is the only location that checks `currentEmbeds`, `coordsA`, `hasWaypointSource()`, etc.
2. **Mode-aware source guard** — `hasWaypointSource()` allows snap/code to generate from a single pin (`pinPoints.length >= 1`); draw mode still requires `userTrajectory.length >= 2`.
3. **Waypoints computed once** — `computeWaypoints()` runs once per change; `playTrajectory()` consumes `genPlaySeq` from state.
4. **Levels are independent** — a pan gesture only sets `RENDER` dirty. A `trajDir` change sets `WAYPOINTS + AUDIO + RENDER`. A UMAP param change sets `COORDS + WAYPOINTS + AUDIO + RENDER`.
5. **One draw per frame** — `scheduleRender()` rAF guard prevents redundant redraws regardless of how many mutations happen synchronously.
6. **Pins survive UMAP changes** — `pinPoints` stores `{src, idx}` (frame identity), not UMAP coordinates. `pinCoords()` derives `[x,y]` live from `coordsA`/`coordsB`, so pins auto-remap when new UMAP coords arrive. Draw trajectories (`rawDrawPoints`) are cleared on UMAP result since they have no frame-index backing.
7. **Pingpong same duration** — code mode: K auto-halves via 2× waypoint list. Snap k_frames: ceil/floor split. Snap equal/prop: half-duration `snapCtxArg` keeps all waypoints within each half.
8. **codeDurMode toggle re-triggers** — `setCodeMode()` calls `applyCodeParamUI()` then `autoRegenerate()`, so switching between K frames and seconds immediately regenerates with the correct new K.
