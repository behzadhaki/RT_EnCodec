import { getContextKs, walkBackward } from './trajectory-utils.js';

// Render the UMAP scatter plot onto `canvas`.
// Returns the lastTransform object (needed by coordinate helpers in the main file),
// or null if there is nothing to draw yet.
//
// state fields:
//   coordsA, coordsB        — UMAP point arrays [[x,y],…]
//   viewZoom, viewPanX, viewPanY
//   pinPoints               — [[umapX,umapY],…]
//   selectedPoint           — {src,idx} | null
//   waveHighlight           — {src,idx} | null
//   userTrajectory          — [[umapX,umapY],…]  (concatenated flat path for playback)
//   drawSegments            — [[[umapX,umapY],…],…]  (per-stroke for rendering; draw mode only)
//   pathMode                — 'draw'|'code'|'snap'
//   snapFrames, codePathFrames
//   trajAnchorPos           — 0..1 during playback, -1 = off
//   snapPlayFrames          — full snap sequence with _wp tags
//   trajDir                 — 'forward'|'backward'|'pingpong'
//   showTrail               — bool
//   canvasMode              — 'explore'|'draw'|'pins'
//   ctxSwap                 — { prob, range } for context swap (walkBackward)
//   snapMode, snapKVal, snapDurVal, framesPerSec   — for snap trail getContextKs
export function renderScatter(canvas, dpr, state) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const { coordsA, coordsB, viewZoom, viewPanX, viewPanY } = state;
  if (!coordsA || !coordsB || !W || !H) return null;

  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const [x, y] of [...coordsA, ...coordsB]) {
    if (x < xMin) xMin = x; if (x > xMax) xMax = x;
    if (y < yMin) yMin = y; if (y > yMax) yMax = y;
  }

  const pad    = 28 * dpr;
  const rangeX = xMax - xMin || 1;
  const rangeY = yMax - yMin || 1;
  const scale  = Math.min((W - 2 * pad) / rangeX, (H - 2 * pad) / rangeY);
  const offX   = (W - rangeX * scale) / 2;
  const offY   = (H - rangeY * scale) / 2;
  const toX    = x => viewPanX + (offX + (x - xMin) * scale) * viewZoom;
  const toY    = y => viewPanY + (H - offY - (y - yMin) * scale) * viewZoom;

  const lastTransform = { xMin, yMin, scale, offX, offY, H };

  const {
    pinPoints, highlightedPinIdx = -1, selectedPoint, waveHighlight, userTrajectory,
    drawSegments,
    pathMode, snapFrames, snapSegBoundaries, codePathFrames, trajAnchorPos,
    snapPlayFrames, trajDir, showTrail, ctxSwap = { prob: 0, range: 0 },
    snapMode, snapKVal, snapDurVal, framesPerSec, canvasMode,
    srcAEnabled = true, srcBEnabled = true,
  } = state;

  function drawTrail(coords, color) {
    if (coords.length < 2) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth   = 0.8 * dpr;
    ctx.globalAlpha = 0.07;
    ctx.beginPath();
    ctx.moveTo(toX(coords[0][0]), toY(coords[0][1]));
    for (let i = 1; i < coords.length; i++) ctx.lineTo(toX(coords[i][0]), toY(coords[i][1]));
    ctx.stroke();
    ctx.restore();
  }

  function drawCirclesEmpty(coords, color) {
    const r = 2.5 * dpr;
    ctx.save();
    ctx.globalAlpha = 0.35;
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1 * dpr;
    for (const [x, y] of coords) {
      ctx.beginPath();
      ctx.arc(toX(x), toY(y), r, 0, 2 * Math.PI);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawTrianglesEmpty(coords, color) {
    const r = 3 * dpr;
    ctx.save();
    ctx.globalAlpha = 0.35;
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1 * dpr;
    for (const [x, y] of coords) {
      const px = toX(x), py = toY(y);
      ctx.beginPath();
      ctx.moveTo(px,             py - r);
      ctx.lineTo(px + r * 0.866, py + r * 0.5);
      ctx.lineTo(px - r * 0.866, py + r * 0.5);
      ctx.closePath();
      ctx.stroke();
    }
    ctx.restore();
  }

  if (srcAEnabled) drawTrail(coordsA, '#c87800');
  if (srcBEnabled) drawTrail(coordsB, '#2090c0');

  if (srcAEnabled) drawCirclesEmpty(coordsA,   '#c87800');
  if (srcBEnabled) drawTrianglesEmpty(coordsB, '#2090c0');

  // Pins drawn AFTER data points so they sit on the highest layer
  if (canvasMode === 'pins' && pinPoints.length > 0) {
    const r = 8 * dpr;
    if (pinPoints.length >= 2) {
      ctx.save();
      ctx.strokeStyle = '#aaaaaa44';
      ctx.lineWidth   = 1 * dpr;
      ctx.beginPath();
      ctx.moveTo(toX(pinPoints[0][0]), toY(pinPoints[0][1]));
      for (let i = 1; i < pinPoints.length; i++)
        ctx.lineTo(toX(pinPoints[i][0]), toY(pinPoints[i][1]));
      ctx.stroke();
      ctx.restore();
    }
    ctx.save();
    ctx.font = `bold ${10 * dpr}px system-ui`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < pinPoints.length; i++) {
      const highlighted = i === highlightedPinIdx;
      const cx = toX(pinPoints[i][0]), cy = toY(pinPoints[i][1]);
      ctx.fillStyle   = highlighted ? '#ffd700' : '#aaaaaa';
      ctx.strokeStyle = '#333333';
      ctx.lineWidth   = 1 * dpr;
      ctx.beginPath(); ctx.rect(cx - r, cy - r, 2 * r, 2 * r);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#000000';
      ctx.fillText(String(i + 1), cx, cy);
    }
    ctx.restore();
  }

  if (selectedPoint) {
    const coords = selectedPoint.src === 'A' ? coordsA : coordsB;
    const pt = coords[selectedPoint.idx];
    if (pt) {
      ctx.save();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth   = 1.5 * dpr;
      ctx.beginPath();
      ctx.arc(toX(pt[0]), toY(pt[1]), 6 * dpr, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.restore();
    }
  }

  if (waveHighlight) {
    const coords = waveHighlight.src === 'A' ? coordsA : coordsB;
    if (coords && waveHighlight.idx < coords.length) {
      const [hx, hy] = coords[waveHighlight.idx];
      ctx.save();
      ctx.strokeStyle = waveHighlight.src === 'A' ? '#e89a20' : '#30aadc';
      ctx.lineWidth   = 2 * dpr;
      ctx.globalAlpha = 0.95;
      ctx.beginPath();
      ctx.arc(toX(hx), toY(hy), 7 * dpr, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.restore();
    }
  }

  if (canvasMode === 'draw' && drawSegments && drawSegments.length > 0) {
    // Each entry is {pts:[[x,y]…], enabled:bool}.
    // Enabled → solid white dashed line; disabled → dimmed gray dashed line.
    ctx.save();
    ctx.lineWidth = 2 * dpr;
    ctx.setLineDash([4 * dpr, 3 * dpr]);
    for (const seg of drawSegments) {
      const pts     = seg.pts ?? seg;          // fallback for plain arrays
      const enabled = seg.pts ? seg.enabled !== false : true;
      if (pts.length < 2) continue;
      ctx.strokeStyle = '#ffffff';
      ctx.globalAlpha = enabled ? 0.80 : 0.22;
      ctx.beginPath();
      ctx.moveTo(toX(pts[0][0]), toY(pts[0][1]));
      for (let i = 1; i < pts.length; i++)
        ctx.lineTo(toX(pts[i][0]), toY(pts[i][1]));
      ctx.stroke();
    }
    ctx.restore();

    // Small filled circle at the start of each non-first segment.
    ctx.save();
    ctx.setLineDash([]);
    for (let si = 1; si < drawSegments.length; si++) {
      const seg     = drawSegments[si];
      const pts     = seg.pts ?? seg;
      const enabled = seg.pts ? seg.enabled !== false : true;
      if (pts.length < 1) continue;
      ctx.fillStyle   = '#ffffff';
      ctx.globalAlpha = enabled ? 0.85 : 0.25;
      ctx.beginPath();
      ctx.arc(toX(pts[0][0]), toY(pts[0][1]), 3.5 * dpr, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.restore();
  }

  // ── Snap overlay ──────────────────────────────────────────────────────────
  if (pathMode === 'snap' && snapFrames && snapFrames.length >= 2) {
    const ptOf = ({ src, idx }) => src === 'A' ? coordsA[idx] : coordsB[idx];

    if (trajAnchorPos >= 0 && snapPlayFrames && snapPlayFrames.length >= 2) {
      const fi       = Math.min(Math.floor(trajAnchorPos * (snapPlayFrames.length - 1)), snapPlayFrames.length - 1);
      const activeWi = snapPlayFrames[fi]?._wp;

      if (activeWi !== undefined && activeWi < snapFrames.length) {
        const wp = snapFrames[activeWi];

        if (showTrail) {
          const ctxKs     = getContextKs(snapFrames, { snapMode, snapKVal, snapDurVal, framesPerSec, coordsA, coordsB });
          const ctxFrames = walkBackward(wp.src, wp.idx, ctxKs[activeWi], ctxSwap, coordsA, coordsB);

          if (ctxFrames.length > 0) {
            const allCtx = [...ctxFrames, wp];
            let contextStart = fi;
            while (contextStart > 0 && snapPlayFrames[contextStart - 1]?._wp === activeWi) contextStart--;
            const elapsed_  = fi - contextStart;
            const segTotal  = allCtx.length - 1;
            const progress  = Math.min(1, elapsed_ / Math.max(1, segTotal));
            const segCount  = Math.ceil(progress * segTotal);

            ctx.save();
            ctx.globalAlpha = 0.85;
            for (let i = 1; i <= segCount && i < allCtx.length; i++) {
              const prev = allCtx[i - 1], curr = allCtx[i];
              const [px, py] = ptOf(prev), [cx, cy] = ptOf(curr);
              ctx.beginPath();
              ctx.moveTo(toX(px), toY(py));
              ctx.lineTo(toX(cx), toY(cy));
              if (prev.src === curr.src) {
                ctx.strokeStyle = curr.src === 'A' ? '#c87800' : '#2090c0';
                ctx.lineWidth   = 0.9 * dpr;
              } else {
                ctx.strokeStyle = '#bbbbbb';
                ctx.lineWidth   = 1 * dpr;
              }
              ctx.stroke();
            }
            ctx.restore();
          }
        }

        ctx.save();
        const [wx, wy] = ptOf(wp);
        ctx.strokeStyle = '#bbbbbb';
        ctx.lineWidth   = 1.5 * dpr;
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        ctx.arc(toX(wx), toY(wy), 5 * dpr, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.restore();
      }
    }

    ctx.save();
    ctx.strokeStyle = '#888888';
    ctx.lineWidth   = 1.5 * dpr;
    ctx.globalAlpha = 0.75;
    const segBoundSet = new Set(snapSegBoundaries || []);
    ctx.beginPath();
    for (let i = 0; i < snapFrames.length; i++) {
      const [x, y] = ptOf(snapFrames[i]);
      (i === 0 || segBoundSet.has(i)) ? ctx.moveTo(toX(x), toY(y)) : ctx.lineTo(toX(x), toY(y));
    }
    ctx.stroke();

    ctx.fillStyle   = '#888888';
    ctx.globalAlpha = 0.55;
    for (const f of snapFrames) {
      const [x, y] = ptOf(f);
      ctx.beginPath(); ctx.arc(toX(x), toY(y), 3 * dpr, 0, 2 * Math.PI); ctx.fill();
    }

    if (canvasMode !== 'pins') {
      ctx.globalAlpha = 1;
      const [sx, sy] = ptOf(snapFrames[0]);
      const [ex, ey] = ptOf(snapFrames[snapFrames.length - 1]);
      ctx.fillStyle = '#44ff88';
      ctx.beginPath(); ctx.arc(toX(sx), toY(sy), 4 * dpr, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = '#ff4466';
      ctx.beginPath(); ctx.arc(toX(ex), toY(ey), 4 * dpr, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.restore();
  }

  // ── Code path overlay ─────────────────────────────────────────────────────
  if (pathMode === 'code' && codePathFrames && codePathFrames.length >= 2) {
    const ptOf = ({ src, idx }) => src === 'A' ? coordsA[idx] : coordsB[idx];
    ctx.save();
    ctx.strokeStyle = '#44ddaa';
    ctx.lineWidth   = 1.5 * dpr;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    for (let i = 0; i < codePathFrames.length; i++) {
      const [x, y] = ptOf(codePathFrames[i]);
      i === 0 ? ctx.moveTo(toX(x), toY(y)) : ctx.lineTo(toX(x), toY(y));
    }
    ctx.stroke();
    ctx.fillStyle   = '#44ddaa';
    ctx.globalAlpha = 0.55;
    for (const f of codePathFrames) {
      const [x, y] = ptOf(f);
      ctx.beginPath(); ctx.arc(toX(x), toY(y), 2.5 * dpr, 0, 2 * Math.PI); ctx.fill();
    }
    if (canvasMode !== 'pins') {
      ctx.globalAlpha = 1;
      const [sx, sy] = ptOf(codePathFrames[0]);
      const [ex, ey] = ptOf(codePathFrames[codePathFrames.length - 1]);
      ctx.fillStyle = '#44ff88';
      ctx.beginPath(); ctx.arc(toX(sx), toY(sy), 4 * dpr, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = '#ff4466';
      ctx.beginPath(); ctx.arc(toX(ex), toY(ey), 4 * dpr, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.restore();
  }

  // ── Moving anchor ─────────────────────────────────────────────────────────
  if (trajAnchorPos >= 0) {
    let ax, ay;
    if (pathMode === 'snap' && snapPlayFrames && snapPlayFrames.length >= 2) {
      const fi = Math.min(Math.floor(trajAnchorPos * (snapPlayFrames.length - 1)), snapPlayFrames.length - 1);
      const f  = snapPlayFrames[fi];
      [ax, ay] = f.src === 'A' ? coordsA[f.idx] : coordsB[f.idx];
    } else if (pathMode === 'code' && codePathFrames && codePathFrames.length >= 2) {
      const fi = Math.min(Math.floor(trajAnchorPos * (codePathFrames.length - 1)), codePathFrames.length - 1);
      const f  = codePathFrames[fi];
      [ax, ay] = f.src === 'A' ? coordsA[f.idx] : coordsB[f.idx];
    } else if (userTrajectory.length >= 2) {
      const p = trajDir === 'backward' ? 1 - trajAnchorPos
              : trajDir === 'pingpong' ? (trajAnchorPos <= 0.5 ? trajAnchorPos * 2 : 2 - trajAnchorPos * 2)
              : trajAnchorPos;
      const ti = p * (userTrajectory.length - 1);
      const lo = Math.floor(ti), hi = Math.min(lo + 1, userTrajectory.length - 1);
      const t  = ti - lo;
      ax = userTrajectory[lo][0] + t * (userTrajectory[hi][0] - userTrajectory[lo][0]);
      ay = userTrajectory[lo][1] + t * (userTrajectory[hi][1] - userTrajectory[lo][1]);
    }
    if (ax !== undefined) {
      ctx.save();
      ctx.fillStyle   = '#ffd700';
      ctx.strokeStyle = '#fff';
      ctx.lineWidth   = 1.5 * dpr;
      ctx.beginPath();
      ctx.arc(toX(ax), toY(ay), 6 * dpr, 0, 2 * Math.PI);
      ctx.fill(); ctx.stroke();
      ctx.restore();
    }
  }

  return lastTransform;
}
