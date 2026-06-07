// Pure trajectory computation utilities — no DOM or global state.
// All functions receive their dependencies explicitly as parameters.

// Extend `frames` to exactly K entries using ping-pong bouncing when too short.
export function pingPongExtend(frames, K) {
  if (!frames.length) return [];
  if (frames.length >= K) return frames;
  const n = frames.length;
  if (n === 1) return Array.from({ length: K }, () => ({ ...frames[0] }));
  const period = 2 * (n - 1);
  return Array.from({ length: K }, (_, i) => {
    const stepsBack = K - 1 - i;
    const pos = stepsBack % period;
    const idx = pos < n ? n - 1 - pos : pos - n + 1;
    return { ...frames[idx] };
  });
}

// Walk backward K steps through the latent graph.
// At each step prefers the cross-source frame nearest to the current position
// if it is within `range` UMAP units AND closer than the temporal predecessor.
// Returns [{src,idx}] in forward (oldest-first) order, length ≤ K.
export function walkBackward(src, idx, K, range, coordsA, coordsB) {
  const frames = [];
  let cSrc = src, cIdx = idx;

  for (let step = 0; step < K; step++) {
    const cCoords = cSrc === 'A' ? coordsA : coordsB;
    const [cx, cy] = cCoords[cIdx];

    let bSrc = null, bIdx = -1, bDist = Infinity;
    if (cIdx > 0) {
      const [px, py] = cCoords[cIdx - 1];
      bDist = Math.sqrt((cx - px) ** 2 + (cy - py) ** 2);
      bSrc = cSrc; bIdx = cIdx - 1;
    }

    if (range > 0) {
      const oSrc    = cSrc === 'A' ? 'B' : 'A';
      const oCoords = oSrc === 'A' ? coordsA : coordsB;
      for (let j = 0; j < oCoords.length; j++) {
        const d = Math.sqrt((oCoords[j][0] - cx) ** 2 + (oCoords[j][1] - cy) ** 2);
        if (d < range && d < bDist) { bDist = d; bSrc = oSrc; bIdx = j; }
      }
    }

    if (bIdx < 0) break;
    cSrc = bSrc; cIdx = bIdx;
    frames.unshift({ src: cSrc, idx: cIdx });
  }
  return frames;
}

export function buildCodePathGraph(coordsA, coordsB, nNeighbors) {
  const nA = coordsA.length, nB = coordsB.length;
  const N  = nA + nB;
  const K  = Math.min(Math.max(2, nNeighbors), 20);
  const adj = Array.from({ length: N }, () => []);

  const cx = i => (i < nA ? coordsA[i] : coordsB[i - nA])[0];
  const cy = i => (i < nA ? coordsA[i] : coordsB[i - nA])[1];
  const w  = (i, j) => Math.sqrt((cx(i) - cx(j)) ** 2 + (cy(i) - cy(j)) ** 2);

  for (let i = 0; i < N; i++) {
    const dists = [];
    for (let j = 0; j < N; j++) {
      if (j === i) continue;
      dists.push([j, (cx(i) - cx(j)) ** 2 + (cy(i) - cy(j)) ** 2]);
    }
    dists.sort((a, b) => a[1] - b[1]);
    for (let k = 0; k < Math.min(K, dists.length); k++) {
      const [j, d2] = dists[k];
      adj[i].push({ j, w: Math.sqrt(d2) });
    }
  }
  for (let i = 1; i < nA; i++) {
    const d = w(i - 1, i);
    adj[i - 1].push({ j: i, w: d }); adj[i].push({ j: i - 1, w: d });
  }
  for (let i = 1; i < nB; i++) {
    const ni = nA + i, np = nA + i - 1;
    const d = w(ni, np);
    adj[np].push({ j: ni, w: d }); adj[ni].push({ j: np, w: d });
  }
  return adj;
}

export function dijkstra(adj, start, end) {
  const N    = adj.length;
  const dist = new Float64Array(N).fill(Infinity);
  const prev = new Int32Array(N).fill(-1);
  const done = new Uint8Array(N);
  dist[start] = 0;
  for (let iter = 0; iter < N; iter++) {
    let u = -1;
    for (let i = 0; i < N; i++) if (!done[i] && (u < 0 || dist[i] < dist[u])) u = i;
    if (u < 0 || dist[u] === Infinity || u === end) break;
    done[u] = 1;
    for (const { j, w } of adj[u]) {
      const nd = dist[u] + w;
      if (nd < dist[j]) { dist[j] = nd; prev[j] = u; }
    }
  }
  if (dist[end] === Infinity) return null;
  const path = [];
  for (let cur = end; cur >= 0; cur = prev[cur]) path.unshift(cur);
  return path;
}

export function findCodePath(userTrajectory, coordsA, coordsB, nNeighbors) {
  if (!userTrajectory || userTrajectory.length < 2 || !coordsA || !coordsB) return null;
  const nA = coordsA.length;

  const nearestNode = ([px, py]) => {
    let best = 0, bestD = Infinity;
    for (let i = 0; i < coordsA.length; i++) {
      const d = (coordsA[i][0] - px) ** 2 + (coordsA[i][1] - py) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    for (let i = 0; i < coordsB.length; i++) {
      const d = (coordsB[i][0] - px) ** 2 + (coordsB[i][1] - py) ** 2;
      if (d < bestD) { bestD = d; best = nA + i; }
    }
    return best;
  };

  const startNode = nearestNode(userTrajectory[0]);
  const endNode   = nearestNode(userTrajectory[userTrajectory.length - 1]);
  if (startNode === endNode) {
    const s = startNode < nA ? { src: 'A', idx: startNode } : { src: 'B', idx: startNode - nA };
    return [s];
  }

  const nodePath = dijkstra(buildCodePathGraph(coordsA, coordsB, nNeighbors), startNode, endNode);
  if (!nodePath) return null;
  return nodePath.map(n => n < nA ? { src: 'A', idx: n } : { src: 'B', idx: n - nA });
}

// Returns { src, idx, dist2 } of the nearest UMAP point to (umapX, umapY).
export function findNearestUmapPoint(umapX, umapY, coordsA, coordsB) {
  let bestDist2 = Infinity, bestSrc = 'A', bestIdx = 0;
  for (let i = 0; i < coordsA.length; i++) {
    const d2 = (coordsA[i][0] - umapX) ** 2 + (coordsA[i][1] - umapY) ** 2;
    if (d2 < bestDist2) { bestDist2 = d2; bestSrc = 'A'; bestIdx = i; }
  }
  for (let i = 0; i < coordsB.length; i++) {
    const d2 = (coordsB[i][0] - umapX) ** 2 + (coordsB[i][1] - umapY) ** 2;
    if (d2 < bestDist2) { bestDist2 = d2; bestSrc = 'B'; bestIdx = i; }
  }
  return { src: bestSrc, idx: bestIdx, dist2: bestDist2 };
}

// Sample the drawn path / pin list and snap each sample to the nearest UMAP frame.
// Returns deduplicated [{src,idx}] waypoints.
export function computeSnapWaypoints({
  canvasMode, pinPoints, coordsA, coordsB,
  userTrajectory, pathMode, snapMode, snapDurVal, codeSecVal,
  trajDuration, framesPerSec,
}) {
  if (!coordsA || !coordsB) return [];

  if (canvasMode === 'pins' && pinPoints.length >= 1) {
    const raw = [];
    for (const [px, py] of pinPoints) {
      const { src: bestSrc, idx: bestIdx } = findNearestUmapPoint(px, py, coordsA, coordsB);
      const last = raw[raw.length - 1];
      if (!last || last.src !== bestSrc || last.idx !== bestIdx) raw.push({ src: bestSrc, idx: bestIdx });
    }
    return raw;
  }

  if (!userTrajectory || userTrajectory.length < 2) return [];

  const dur = pathMode === 'snap' ? snapDurVal
            : pathMode === 'code' ? codeSecVal
            : (trajDuration || 4);
  const N = Math.max(2, Math.round(dur * framesPerSec));

  const raw = [];
  for (let i = 0; i < N; i++) {
    const frac = i / (N - 1);
    const ti   = frac * (userTrajectory.length - 1);
    const lo   = Math.floor(ti), hi = Math.min(lo + 1, userTrajectory.length - 1);
    const tt   = ti - lo;
    const px   = userTrajectory[lo][0] + tt * (userTrajectory[hi][0] - userTrajectory[lo][0]);
    const py   = userTrajectory[lo][1] + tt * (userTrajectory[hi][1] - userTrajectory[lo][1]);
    const { src: bestSrc, idx: bestIdx } = findNearestUmapPoint(px, py, coordsA, coordsB);
    const last = raw[raw.length - 1];
    if (!last || last.src !== bestSrc || last.idx !== bestIdx) raw.push({ src: bestSrc, idx: bestIdx });
  }
  return raw;
}

// Compute per-waypoint context (pre-roll) frame counts.
export function getContextKs(waypoints, { snapMode, snapKVal, snapDurVal, framesPerSec, coordsA, coordsB }) {
  const N_wp = waypoints.length;
  if (snapMode === 'k_frames') return waypoints.map(() => snapKVal);

  const N_target = Math.max(N_wp, Math.round(snapDurVal * framesPerSec));

  if (snapMode === 'equal' || N_wp <= 1) {
    const K_each = Math.max(1, Math.floor(Math.max(0, N_target - N_wp) / N_wp));
    return waypoints.map(() => K_each);
  }

  // Proportional: context length proportional to UMAP segment distance.
  // Waypoint 0 has no prior segment, so give it a virtual weight equal to
  // the average of the real segments — ensuring fair allocation for wp0.
  // (Old: K_each[0] = K_min=4 always → with 2 pins, one got 4 frames, the
  //  other got nearly the entire budget, making wp0 almost inaudible.)
  const total_context = Math.max(0, N_target - N_wp);
  const segDists = [];
  let totalDist  = 0;
  for (let i = 1; i < N_wp; i++) {
    const [x0, y0] = (waypoints[i - 1].src === 'A' ? coordsA : coordsB)[waypoints[i - 1].idx];
    const [x1, y1] = (waypoints[i].src   === 'A' ? coordsA : coordsB)[waypoints[i].idx];
    const d = Math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2);
    segDists.push(d);
    totalDist += d;
  }

  // weights[0] = avgDist (virtual), weights[i] = segDists[i-1] for i >= 1
  const avgDist     = totalDist / (N_wp - 1);
  const weights     = [avgDist, ...segDists];
  const totalWeight = weights.reduce((a, b) => a + b, 0);

  const K_each = new Array(N_wp).fill(0);
  if (totalWeight <= 0) {
    // All waypoints at identical UMAP position — fall back to equal distribution.
    const K_eq = Math.max(1, Math.floor(total_context / N_wp));
    return waypoints.map(() => K_eq);
  }
  for (let i = 0; i < N_wp; i++) {
    K_each[i] = Math.max(1, Math.floor(total_context * weights[i] / totalWeight));
  }
  return K_each;
}

export function buildSnapSequence(waypoints, K_pre, K_post, ctxSwap, coordsA, coordsB, currentEmbeds) {
  const T_A = currentEmbeds.embQuantDimsA[2];
  const T_B = currentEmbeds.embQuantDimsB[2];
  const getT = src => src === 'A' ? T_A : T_B;
  const seq  = [];

  const addPreroll = (src, idx, K, wi) => {
    const frames = pingPongExtend(walkBackward(src, idx, K, ctxSwap, coordsA, coordsB), K);
    for (const f of frames) seq.push({ ...f, _wp: wi });
    seq.push({ src, idx, _wp: wi });
  };

  addPreroll(waypoints[0].src, waypoints[0].idx, K_pre, 0);

  for (let w = 1; w < waypoints.length; w++) {
    const { src, idx } = waypoints[w];
    const prev = waypoints[w - 1];
    if (prev.src === src && idx > prev.idx && (idx - prev.idx) <= K_pre) {
      for (let i = prev.idx + 1; i <= idx; i++) seq.push({ src, idx: i, _wp: w });
    } else {
      addPreroll(src, idx, K_pre, w);
    }
  }

  const lastWi = waypoints.length - 1;
  if (K_post > 0) {
    const { src: sL, idx: iL } = waypoints[lastWi];
    const T = getT(sL);
    for (let i = iL + 1; i <= Math.min(T - 1, iL + K_post); i++) seq.push({ src: sL, idx: i, _wp: lastWi });
  }

  if (!seq.length) return [];
  const out = [seq[0]];
  for (let i = 1; i < seq.length; i++) {
    const p = out[out.length - 1];
    if (seq[i].src !== p.src || seq[i].idx !== p.idx) out.push(seq[i]);
  }
  return out;
}

export function buildSnapSequenceFixed(waypoints, N_target, K_post, ctxSwap, coordsA, coordsB, currentEmbeds, snapCtx) {
  const N_wp   = waypoints.length;
  const K_each = getContextKs(waypoints, snapCtx);

  const seq = [];
  for (let w = 0; w < N_wp; w++) {
    const { src, idx } = waypoints[w];
    const K_this = K_each[w];
    if (K_this > 0) {
      const frames = pingPongExtend(walkBackward(src, idx, K_this, ctxSwap, coordsA, coordsB), K_this);
      for (const f of frames) seq.push({ ...f, _wp: w });
    }
    seq.push({ src, idx, _wp: w });
  }

  if (seq.length > N_target) seq.length = N_target;

  if (K_post > 0) {
    const { src: sL, idx: iL } = waypoints[N_wp - 1];
    const T_L = sL === 'A' ? currentEmbeds.embQuantDimsA[2] : currentEmbeds.embQuantDimsB[2];
    for (let i = iL + 1; i <= Math.min(T_L - 1, iL + K_post); i++) seq.push({ src: sL, idx: i, _wp: N_wp - 1 });
  }

  return seq;
}

export function makeFrameScales(seq, currentEmbeds) {
  if (!currentEmbeds.scalesA && !currentEmbeds.scalesB) return null;
  const out = new Float32Array(seq.length);
  for (let i = 0; i < seq.length; i++) {
    const { src, idx } = seq[i];
    const arr = src === 'A' ? currentEmbeds.scalesA : currentEmbeds.scalesB;
    out[i] = arr ? (arr[idx] || 1.0) : 1.0;
  }
  return out;
}
