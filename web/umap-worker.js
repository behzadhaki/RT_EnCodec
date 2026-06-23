import { UMAP } from 'https://esm.sh/umap-js';

// ── perf self-stats (busy% via timer-drift; heap on demand) ──
let _busyMs = 0, _wallMs = 0, _lastTick = performance.now();
setInterval(() => {
  const now = performance.now(), dt = now - _lastTick;
  _wallMs += dt; _busyMs += Math.max(0, dt - 250);
  _lastTick = now;
}, 250);
function _statReply(id) {
  self.postMessage({ type: '__stat', id,
    heap: (typeof performance !== 'undefined' && performance.memory) ? performance.memory.usedJSHeapSize : 0,
    busyMs: _busyMs, wallMs: _wallMs });
}

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

// ── Classical MDS (PCoA) on a similarity matrix ───────────────────────────────
// sim is N×N (row-major), cosine ∈ [-1,1] ⇒ d² = 2-2·sim, so the MDS Gram
// matrix B = -½·J·d²·J reduces (the constant 2 and rank-1 terms vanish under
// centering) to B = J·sim·J — the double-centred similarity. We take its top-2
// eigenvectors by power iteration (B is PSD for a cosine Gram), with B·v formed
// implicitly so no N×N matrix is materialised. coords = vₖ·√λₖ.
function computeMDS(sim, N) {
  const r = new Float64Array(N); // row means (= col means; sim symmetric)
  let g = 0;
  for (let i = 0; i < N; i++) {
    let s = 0; const off = i * N;
    for (let j = 0; j < N; j++) s += sim[off + j];
    r[i] = s / N; g += r[i];
  }
  g /= N;

  function Bmul(v) {
    let sumv = 0, rv = 0;
    for (let i = 0; i < N; i++) { sumv += v[i]; rv += r[i] * v[i]; }
    const out = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let sv = 0; const off = i * N;
      for (let j = 0; j < N; j++) sv += sim[off + j] * v[j];
      out[i] = sv - r[i] * sumv - rv + g * sumv;
    }
    return out;
  }

  function powerIter(deflect) {
    const v = new Float64Array(N);
    for (let i = 0; i < N; i++) v[i] = Math.sin(i + 1); // non-constant start (B kills the constant)
    for (let it = 0; it < 120; it++) {
      const Mv = Bmul(v);
      if (deflect) {
        let d = 0; for (let i = 0; i < N; i++) d += Mv[i] * deflect[i];
        for (let i = 0; i < N; i++) Mv[i] -= d * deflect[i];
      }
      let norm = 0; for (let i = 0; i < N; i++) norm += Mv[i] * Mv[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-14) break;
      for (let i = 0; i < N; i++) v[i] = Mv[i] / norm;
    }
    const Bv = Bmul(v);
    let lambda = 0; for (let i = 0; i < N; i++) lambda += v[i] * Bv[i];
    return { v, s: Math.sqrt(Math.max(0, lambda)) };
  }

  const e1 = powerIter(null);
  const e2 = powerIter(e1.v);
  const result = new Float32Array(N * 2);
  for (let i = 0; i < N; i++) {
    result[i * 2]     = e1.v[i] * e1.s;
    result[i * 2 + 1] = e2.v[i] * e2.s;
  }
  return result;
}

// ── Randomized PCA — feature pre-reduction ────────────────────────────────────
// Reduce flat (N×D row-major) to its top-m principal-component scores, where m
// is the smallest count whose cumulative explained variance reaches varTarget
// (clamped to [2, kMax]). Randomized range-finding + one power iteration:
// never materialises the centered matrix or a D×D covariance — centering is
// folded into every product via the column means. Cost ≈ 3·N·D·L flops,
// L = kMax + 8 oversampling. Deterministic (fixed seed).
// Returns { reduced (N×m Float32), m, explained } or null when D is already
// small or the data is degenerate.

function gaussianFill(arr, rand) {
  for (let i = 0; i < arr.length; i += 2) {
    const u = Math.max(rand(), 1e-12), v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    arr[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < arr.length) arr[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
}

// Modified Gram-Schmidt on the L columns of row-major M (R rows × L cols), in place.
function mgsOrthonormalize(M, R, L) {
  for (let j = 0; j < L; j++) {
    for (let k = 0; k < j; k++) {
      let dot = 0;
      for (let r = 0; r < R; r++) dot += M[r * L + k] * M[r * L + j];
      for (let r = 0; r < R; r++) M[r * L + j] -= dot * M[r * L + k];
    }
    let norm = 0;
    for (let r = 0; r < R; r++) norm += M[r * L + j] ** 2;
    norm = Math.sqrt(norm);
    if (norm < 1e-12) for (let r = 0; r < R; r++) M[r * L + j]  = 0;
    else              for (let r = 0; r < R; r++) M[r * L + j] /= norm;
  }
}

// Cyclic Jacobi eigendecomposition of symmetric n×n A (row-major).
// Returns { vals, vecs } sorted by descending eigenvalue; eigenvectors are
// the COLUMNS of vecs.
function jacobiEigSym(Ain, n) {
  const A = Float64Array.from(Ain);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) off += A[p * n + q] ** 2;
    if (off < 1e-20) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-18) continue;
        const theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => A[b * n + b] - A[a * n + a]);
  const vals = new Float64Array(n);
  const vecs = new Float64Array(n * n);
  order.forEach((src, dst) => {
    vals[dst] = A[src * n + src];
    for (let k = 0; k < n; k++) vecs[k * n + dst] = V[k * n + src];
  });
  return { vals, vecs };
}

function pcaReduceFeatures(flat, N, D, kMax, varTarget) {
  if (D <= kMax) return null;
  const rand = mulberry32(0xC0FFEE);
  const L = Math.min(kMax + 8, D, N);

  // Column means + total centered variance ‖X_c‖²
  const mean = new Float64Array(D);
  for (let i = 0; i < N; i++)
    for (let d = 0; d < D; d++) mean[d] += flat[i * D + d];
  for (let d = 0; d < D; d++) mean[d] /= N;
  let totalVar = 0;
  for (let i = 0; i < N; i++)
    for (let d = 0; d < D; d++) { const e = flat[i * D + d] - mean[d]; totalVar += e * e; }
  if (totalVar < 1e-12) return null;

  // Y = X_c·Ω  (Ω: D×L gaussian); centering via X·Ω − 1·(μᵀΩ)
  const omega = new Float64Array(D * L);
  gaussianFill(omega, rand);
  const muOmega = new Float64Array(L);
  for (let d = 0; d < D; d++) {
    const m = mean[d];
    for (let l = 0; l < L; l++) muOmega[l] += m * omega[d * L + l];
  }
  let Q = new Float64Array(N * L);
  for (let i = 0; i < N; i++) {
    const off = i * D, qoff = i * L;
    for (let d = 0; d < D; d++) {
      const x = flat[off + d];
      if (x === 0) continue;
      const ooff = d * L;
      for (let l = 0; l < L; l++) Q[qoff + l] += x * omega[ooff + l];
    }
    for (let l = 0; l < L; l++) Q[qoff + l] -= muOmega[l];
  }
  mgsOrthonormalize(Q, N, L);

  // One power iteration: Z = X_cᵀ·Q → orthonormalize → Q = X_c·Z → orthonormalize
  const qColSum = new Float64Array(L);
  for (let i = 0; i < N; i++)
    for (let l = 0; l < L; l++) qColSum[l] += Q[i * L + l];
  const Z = new Float64Array(D * L);
  for (let i = 0; i < N; i++) {
    const off = i * D, qoff = i * L;
    for (let d = 0; d < D; d++) {
      const x = flat[off + d];
      if (x === 0) continue;
      const zoff = d * L;
      for (let l = 0; l < L; l++) Z[zoff + l] += x * Q[qoff + l];
    }
  }
  for (let d = 0; d < D; d++)
    for (let l = 0; l < L; l++) Z[d * L + l] -= mean[d] * qColSum[l];
  mgsOrthonormalize(Z, D, L);
  const muZ = new Float64Array(L);
  for (let d = 0; d < D; d++) {
    const m = mean[d];
    for (let l = 0; l < L; l++) muZ[l] += m * Z[d * L + l];
  }
  Q = new Float64Array(N * L);
  for (let i = 0; i < N; i++) {
    const off = i * D, qoff = i * L;
    for (let d = 0; d < D; d++) {
      const x = flat[off + d];
      if (x === 0) continue;
      const zoff = d * L;
      for (let l = 0; l < L; l++) Q[qoff + l] += x * Z[zoff + l];
    }
    for (let l = 0; l < L; l++) Q[qoff + l] -= muZ[l];
  }
  mgsOrthonormalize(Q, N, L);

  // B = Qᵀ·X_c (L×D), then G = B·Bᵀ (L×L) → eig gives σ² (vals) and U (vecs)
  const qSum2 = new Float64Array(L);
  for (let i = 0; i < N; i++)
    for (let l = 0; l < L; l++) qSum2[l] += Q[i * L + l];
  const B = new Float64Array(L * D);
  for (let i = 0; i < N; i++) {
    const off = i * D, qoff = i * L;
    for (let l = 0; l < L; l++) {
      const q = Q[qoff + l];
      if (q === 0) continue;
      const boff = l * D;
      for (let d = 0; d < D; d++) B[boff + d] += q * flat[off + d];
    }
  }
  for (let l = 0; l < L; l++)
    for (let d = 0; d < D; d++) B[l * D + d] -= qSum2[l] * mean[d];

  const G = new Float64Array(L * L);
  for (let a = 0; a < L; a++)
    for (let b = a; b < L; b++) {
      let s = 0;
      for (let d = 0; d < D; d++) s += B[a * D + d] * B[b * D + d];
      G[a * L + b] = s; G[b * L + a] = s;
    }
  const { vals, vecs } = jacobiEigSym(G, L);

  // m = smallest count reaching varTarget cumulative variance, in [2, kMax]
  let m = 0, cum = 0;
  for (let c = 0; c < L; c++) {
    if (vals[c] <= 0) break;
    cum += vals[c] / totalVar;
    m = c + 1;
    if (cum >= varTarget) break;
  }
  m = Math.max(2, Math.min(m, kMax));
  let explained = 0;
  for (let c = 0; c < m; c++) explained += Math.max(0, vals[c]) / totalVar;
  explained = Math.min(1, explained);

  // PCA scores = X_c·V = Q·(U·diag σ) — columns ordered by variance, so the
  // first two double as a 2-D PCA layout (used to seed UMAP init).
  const W = new Float64Array(L * m);
  for (let l = 0; l < L; l++)
    for (let c = 0; c < m; c++) W[l * m + c] = vecs[l * L + c] * Math.sqrt(Math.max(0, vals[c]));
  const reduced = new Float32Array(N * m);
  for (let i = 0; i < N; i++) {
    const qoff = i * L, roff = i * m;
    for (let l = 0; l < L; l++) {
      const q = Q[qoff + l];
      if (q === 0) continue;
      const woff = l * m;
      for (let c = 0; c < m; c++) reduced[roff + c] += q * W[woff + c];
    }
  }
  return { reduced, m, explained };
}

// ── Distance metrics ──────────────────────────────────────────────────────────

function cosineDist(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const denom = Math.sqrt(na * nb);
  return denom > 1e-12 ? 1 - dot / denom : 0;
}

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = ({ data: msg }) => {
  if (msg.type === '__stat') { _statReply(msg.id); return; }
  // ── Feature pre-reduction ──────────────────────────────────────────────────
  if (msg.type === 'reduce_features') {
    const { jobId, flat, N, D, kMax, varTarget } = msg;
    try {
      const res = pcaReduceFeatures(flat, N, D, kMax, varTarget);
      if (!res) {
        self.postMessage({ type: 'reduce_result', jobId, reduced: null });
      } else {
        self.postMessage(
          { type: 'reduce_result', jobId, reduced: res.reduced, m: res.m, explained: res.explained },
          [res.reduced.buffer]
        );
      }
    } catch (err) {
      self.postMessage({ type: 'reduce_error', jobId, message: 'PCA reduce: ' + err.message });
    }
    return;
  }

  // ── MDS (classical / PCoA on the SSM) ──────────────────────────────────────
  if (msg.type === 'compute_mds') {
    const { jobId, sim, N } = msg;
    try {
      const result = computeMDS(sim, N);
      self.postMessage({ type: 'mds_result', jobId, result, N }, [result.buffer]);
    } catch (err) {
      self.postMessage({ type: 'mds_error', jobId, message: err.message });
    }
    return;
  }

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

  // ── 1-D UMAP (wheel-mode angle) ───────────────────────────────────────────
  // Returns one scalar per frame; the main thread rank-wraps it to [0, 2π).
  // Neighbourhood-preserving, so timbre families stay angularly contiguous —
  // unlike PC1, which is a single linear axis.
  if (msg.type === 'compute_umap_1d') {
    const { jobId, seed = 7, flat, T_A, T_B, D, nNeighbors, minDist, metric = 'euclidean' } = msg;
    Math.random = mulberry32(seed);
    try {
      const n = T_A + T_B;
      const all = [];
      for (let i = 0; i < n; i++)
        all.push(Array.from(flat.subarray(i * D, (i + 1) * D)));

      const opts = { nComponents: 1, nNeighbors, minDist };
      if (metric === 'cosine') opts.distanceFn = cosineDist;
      const umap    = new UMAP(opts);
      const nEpochs = umap.initializeFit(all);

      for (let e = 0; e < nEpochs; e++) {
        umap.step();
        if (e % 10 === 0 || e === nEpochs - 1)
          self.postMessage({ type: 'umap1d_progress', jobId, epoch: e, nEpochs });
      }

      const coords = umap.getEmbedding();
      const result = new Float32Array(n);
      for (let i = 0; i < n; i++) result[i] = coords[i][0];

      self.postMessage({ type: 'umap1d_result', jobId, result, T_A, T_B }, [result.buffer]);
    } catch (err) {
      self.postMessage({ type: 'umap1d_error', jobId, message: err.message });
    }
    return;
  }

  // ── UMAP ──────────────────────────────────────────────────────────────────
  if (msg.type !== 'compute_umap') return;

  const { jobId, slotIdx = 0, seed = 1, flat, T_A, T_B, D, nNeighbors, minDist,
          metric = 'euclidean', initCoords = null } = msg;

  // Each slot uses its own seed so layouts are reproducible but visually distinct.
  Math.random = mulberry32(seed);

  try {
    const n = T_A + T_B;
    const all = [];
    for (let i = 0; i < n; i++)
      all.push(Array.from(flat.subarray(i * D, (i + 1) * D)));

    const opts = { nComponents: 2, nNeighbors, minDist };
    if (metric === 'cosine') opts.distanceFn = cosineDist;
    const umap    = new UMAP(opts);
    const nEpochs = umap.initializeFit(all);

    // PCA-seeded init: overwrite the random initial embedding IN PLACE (the
    // optimizer may hold a reference to the same array) with the top-2 PCA
    // scores scaled to umap-js's native ±10 init range. Tiny per-slot jitter
    // breaks exact ties; slot variety then comes from negative sampling.
    // Best-effort: if umap-js internals change shape, fall back to random init.
    if (initCoords) {
      try {
        const emb = umap.embedding;
        if (Array.isArray(emb) && emb.length === n && emb[0]?.length === 2) {
          let maxAbs = 0;
          for (let i = 0; i < initCoords.length; i++)
            maxAbs = Math.max(maxAbs, Math.abs(initCoords[i]));
          const s = maxAbs > 0 ? 10 / maxAbs : 1;
          for (let i = 0; i < n; i++) {
            emb[i][0] = initCoords[i * 2]     * s + (Math.random() - 0.5) * 0.02;
            emb[i][1] = initCoords[i * 2 + 1] * s + (Math.random() - 0.5) * 0.02;
          }
        }
      } catch { /* seeding is best-effort */ }
    }

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
