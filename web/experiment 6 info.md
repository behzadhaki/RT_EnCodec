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
currentEmbeds        // {framesA, framesB, n_q, fps, dim}
encodePhase          // 'A' | 'B' | null

// COORDS inputs
umapMode, umapLevel, embSpace, embMode
nNeighbors, minDist, umapWindow

// COORDS (async)
coordsA, coordsB

// WAYPOINTS inputs
userTrajectory       // resampled path — single source of truth
rawDrawPoints        // draw mode scratch (before mouseUp resample)
pinPoints            // [{src, idx}, ...] — stores frame identity, not UMAP coords
pathMode, trajDir, snapMode
snapKVal, snapDurVal, codeKVal, codeSecVal, codeDurMode
trajDuration, ctxSwapRange

// WAYPOINTS (sync, derived by computeWaypoints())
snapFrames           // visual waypoint dots for snap mode
codePathFrames       // visual waypoint dots for code mode
snapPlayFrames       // full sequence with context for snap playback
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
| userTrajectory changes (draw/pin) | `WAYPOINTS AUDIO RENDER` (via autoRegenerate) |
| Path params change (pathMode, trajDir, snapMode, etc.) | `WAYPOINTS AUDIO RENDER` (via autoRegenerate) |
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
  if (dirty.WAYPOINTS && coordsA && coordsB && userTrajectory.length >= 2) {
    dirty.WAYPOINTS = false;
    computeWaypoints();        // sync — updates snapFrames/codePathFrames/genPlaySeq
    dirty.AUDIO  = true;
    dirty.RENDER = true;
  }
  if (dirty.AUDIO && currentEmbeds && coordsA && coordsB
      && userTrajectory.length >= 2
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

Extracted from both `autoRegenerate()` and `playTrajectory()`. Runs synchronously when `WAYPOINTS` is dirty.

- **code mode**: calls `computeSnapWaypoints()` once, samples N frames → stores in `codePathFrames` and `genPlaySeq`
- **snap mode**: calls `computeSnapWaypoints()` once, builds sequence → stores in `snapFrames`, `snapPlayFrames`, `genPlaySeq`
- **draw mode**: no-op (embedding interpolation happens at decode time in `playTrajectory`)

Previously `computeSnapWaypoints()` was called twice per change — once in `autoRegenerate()` for the visual, again inside `playTrajectory()` for audio decode. Now it runs once and both consumers read from state.

---

## playTrajectory() — reads from state

After the refactor, the code/snap branches just read `genPlaySeq` (already set by `computeWaypoints()`), extract embeddings, and dispatch to the worker. The draw branch still computes blended embeddings inline since it has no discrete frame sequence.

---

## autoRegenerate()

Reduced to a one-liner:

```js
function autoRegenerate() {
  if (userTrajectory.length >= 2 && (!currentEmbeds || !coordsA || !coordsB)) {
    setStatus('Path drawn. Encode sources and run UMAP to play.');
  }
  markDirty('WAYPOINTS', 'AUDIO', 'RENDER');
}
```

Called from: trajectory draws/drags (mouseUp), pin add/remove/drag/reorder, path param changes, canvas mode switches.

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

1. **Guards in one place** — `runEffects()` is the only location that checks `currentEmbeds`, `coordsA`, `userTrajectory.length >= 2`, etc.
2. **Waypoints computed once** — `computeWaypoints()` runs once per change; `playTrajectory()` consumes `genPlaySeq` from state.
3. **Levels are independent** — a pan gesture only sets `RENDER` dirty. A `trajDir` change sets `WAYPOINTS + AUDIO + RENDER`. A UMAP param change sets `COORDS + WAYPOINTS + AUDIO + RENDER`.
4. **One draw per frame** — `scheduleRender()` rAF guard prevents redundant redraws regardless of how many mutations happen synchronously.
5. **No autoRegenerate sprawl** — all trajectory-param listeners call `autoRegenerate()` which delegates entirely to `markDirty`.
6. **Pins survive UMAP changes** — `pinPoints` stores `{src, idx}` (frame identity), not UMAP coordinates. `pinCoords()` derives `[x,y]` live from `coordsA`/`coordsB`, so pins auto-remap when new UMAP coords arrive. Draw trajectories (`rawDrawPoints`) are cleared on UMAP result since they have no frame-index backing.
