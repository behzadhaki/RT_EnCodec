import { UMAP } from 'https://esm.sh/umap-js';

// ── PRNG ──────────────────────────────────────────────────────────────────────

function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── PCA ───────────────────────────────────────────────────────────────────────
// Standard covariance PCA — top-2 principal components via power iteration.
// Runs in O(N·D²) for the covariance build + O(D²) per iteration.

function computePCA(flat, N, D) {
  // 1. Mean-centre
  const mean = new Float64Array(D);
  for (let i = 0; i < N; i++)
    for (let d = 0; d < D; d++)
      mean[d] += flat[i * D + d];
  for (let d = 0; d < D; d++) mean[d] /= N;

  const X = new Float64Array(N * D);
  for (let i = 0; i < N; i++)
    for (let d = 0; d < D; d++)
      X[i * D + d] = flat[i * D + d] - mean[d];

  // 2. Covariance matrix C = X^T X / N  (D×D, row-major)
  const C = new Float64Array(D * D);
  for (let i = 0; i < N; i++) {
    const off = i * D;
    for (let a = 0; a < D; a++) {
      const xa = X[off + a];
      if (xa === 0) continue;
      for (let b = 0; b < D; b++)
        C[a * D + b] += xa * X[off + b];
    }
  }
  for (let k = 0; k < D * D; k++) C[k] /= N;

  // 3. Power iteration — extract one eigenvector, optionally deflating by a known one.
  function powerIter(deflect) {
    const v = new Float64Array(D);
    for (let d = 0; d < D; d++) v[d] = 1 / Math.sqrt(D); // uniform start

    for (let iter = 0; iter < 200; iter++) {
      // Mv = C @ v
      const Mv = new Float64Array(D);
      for (let a = 0; a < D; a++)
        for (let b = 0; b < D; b++)
          Mv[a] += C[a * D + b] * v[b];

      // Deflate: project out the already-found component
      if (deflect) {
        let dot = 0;
        for (let d = 0; d < D; d++) dot += Mv[d] * deflect[d];
        for (let d = 0; d < D; d++) Mv[d] -= dot * deflect[d];
      }

      // Normalise
      let norm = 0;
      for (let d = 0; d < D; d++) norm += Mv[d] * Mv[d];
      norm = Math.sqrt(norm);
      if (norm < 1e-14) break;
      for (let d = 0; d < D; d++) v[d] = Mv[d] / norm;
    }
    return v;
  }

  const pc1 = powerIter(null);
  const pc2 = powerIter(pc1);

  // 4. Project data
  const result = new Float32Array(N * 2);
  for (let i = 0; i < N; i++) {
    let x = 0, y = 0;
    const off = i * D;
    for (let d = 0; d < D; d++) {
      const v = X[off + d];
      x += v * pc1[d];
      y += v * pc2[d];
    }
    result[i * 2]     = x;
    result[i * 2 + 1] = y;
  }
  return result;
}

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = ({ data: msg }) => {
  // ── PCA ───────────────────────────────────────────────────────────────────
  if (msg.type === 'compute_pca') {
    const { jobId, flat, T_A, T_B, D } = msg;
    try {
      const result = computePCA(flat, T_A + T_B, D);
      self.postMessage({ type: 'pca_result', jobId, result, T_A, T_B }, [result.buffer]);
    } catch (err) {
      self.postMessage({ type: 'pca_error', jobId, message: err.message });
    }
    return;
  }

  // ── UMAP ──────────────────────────────────────────────────────────────────
  if (msg.type !== 'compute_umap') return;

  const { jobId, slotIdx = 0, seed = 1, flat, T_A, T_B, D, nNeighbors, minDist } = msg;

  // Each slot uses its own seed so layouts are reproducible but visually distinct.
  Math.random = mulberry32(seed);

  try {
    const n = T_A + T_B;
    const all = [];
    for (let i = 0; i < n; i++)
      all.push(Array.from(flat.subarray(i * D, (i + 1) * D)));

    const umap    = new UMAP({ nComponents: 2, nNeighbors, minDist });
    const nEpochs = umap.initializeFit(all);

    for (let e = 0; e < nEpochs; e++) {
      umap.step();
      if (e % 10 === 0 || e === nEpochs - 1)
        self.postMessage({ type: 'umap_progress', jobId, slotIdx, epoch: e, nEpochs });
    }

    const coords = umap.getEmbedding();
    const result = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      result[i * 2]     = coords[i][0];
      result[i * 2 + 1] = coords[i][1];
    }

    self.postMessage({ type: 'umap_result', jobId, slotIdx, result, T_A, T_B }, [result.buffer]);
  } catch (err) {
    self.postMessage({ type: 'umap_error', jobId, slotIdx, message: err.message });
  }
};
