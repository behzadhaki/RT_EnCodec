// ── SSM worker ──────────────────────────────────────────────────────────────
// Self- / cross-similarity matrix over the shared feature matrix `flat`
// ((T_A+T_B)×D, z-scored, A rows first). Temporal pooling caps the matrix at
// P×P regardless of file length, so it scales to arbitrarily long audio:
// each source is mean-pooled to ≤P super-frames before the similarity matmul.
//
// Scopes:
//   joint — [A;B] concatenation → (pa+pb)² block matrix; A·A and B·B on the
//           diagonal blocks, A·B (the cross-similarity) off-diagonal. Seam at pa.
//   cross — A rows × B cols → pa×pb (strictly frame-of-A vs frame-of-B).
//   self  — one source only → p×p.
//
// Metric: cosine (L2-normalise super-frames, S = X·Xᵀ ∈ [-1,1]) or euclidean
// (distance mapped to a similarity by min/max). Output is a colour-mapped RGBA
// image (transferable), plus geometry so the main thread can place seams/axes.

// ── magma colormap (9 control stops, linear interp) ─────────────────────────
const MAGMA = [
  [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122],
  [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191],
];
function magma(t, out, o) {
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const x = t * (MAGMA.length - 1);
  const i = Math.min(MAGMA.length - 2, x | 0);
  const f = x - i, a = MAGMA[i], b = MAGMA[i + 1];
  out[o]     = (a[0] + (b[0] - a[0]) * f) | 0;
  out[o + 1] = (a[1] + (b[1] - a[1]) * f) | 0;
  out[o + 2] = (a[2] + (b[2] - a[2]) * f) | 0;
  out[o + 3] = 255;
}

// Mean-pool the rows [start, start+T) of `flat` into `p` super-frames (≤T).
// Returns a Float32Array (p×D), row-major. p === T is an identity copy.
function poolRows(flat, start, T, D, p) {
  const out = new Float32Array(p * D);
  for (let i = 0; i < p; i++) {
    const lo = start + Math.floor((i * T) / p);
    const hi = start + Math.floor(((i + 1) * T) / p);
    const n = Math.max(1, hi - lo);
    const oo = i * D;
    for (let r = lo; r < hi; r++) {
      const ro = r * D;
      for (let d = 0; d < D; d++) out[oo + d] += flat[ro + d];
    }
    for (let d = 0; d < D; d++) out[oo + d] /= n;
  }
  return out;
}

// In-place L2-normalise each row (for cosine). Returns the same array.
function l2normRows(X, p, D) {
  for (let i = 0; i < p; i++) {
    const o = i * D;
    let s = 0;
    for (let d = 0; d < D; d++) s += X[o + d] * X[o + d];
    const inv = s > 1e-12 ? 1 / Math.sqrt(s) : 0;
    for (let d = 0; d < D; d++) X[o + d] *= inv;
  }
  return X;
}

// Squared-norm of each row (for euclidean distance via |a-b|² = |a|²+|b|²-2a·b).
function rowNorms2(X, p, D) {
  const out = new Float32Array(p);
  for (let i = 0; i < p; i++) {
    const o = i * D;
    let s = 0;
    for (let d = 0; d < D; d++) s += X[o + d] * X[o + d];
    out[i] = s;
  }
  return out;
}

// Dot product of row i of A and row j of B (both row-major, width D).
function dot(A, i, B, j, D) {
  const oa = i * D, ob = j * D;
  let s = 0;
  for (let d = 0; d < D; d++) s += A[oa + d] * B[ob + d];
  return s;
}

// Pooled joint cosine SSM used as a projection input (MDS / UMAP-on-SSM /
// PCA-on-SSM). Returns the raw (pA+pB)² similarity matrix — no colormap — plus
// the pooled per-source counts so the main thread can map super-frames → frames.
function computeSSMInput(flat, T_A, T_B, D, P) {
  const budget = Math.max(2, P | 0);
  const pA = T_A > 0 ? Math.min(budget, T_A) : 0;
  const pB = T_B > 0 ? Math.min(budget, T_B) : 0;
  const M  = pA + pB;
  const X  = new Float32Array(M * D);
  if (pA) X.set(poolRows(flat, 0,   T_A, D, pA), 0);
  if (pB) X.set(poolRows(flat, T_A, T_B, D, pB), pA * D);
  l2normRows(X, M, D);                 // cosine ⇒ S = X·Xᵀ ∈ [-1,1], PSD
  const sim = new Float32Array(M * M);
  for (let i = 0; i < M; i++) {
    for (let j = i; j < M; j++) {
      const v = dot(X, i, X, j, D);
      sim[i * M + j] = v;
      sim[j * M + i] = v;
    }
  }
  return { sim, pA, pB, T_A, T_B };
}

self.onmessage = ({ data: msg }) => {
  if (msg.type === 'compute_ssm_input') {
    const { jobId, flat, T_A, T_B, D, P } = msg;
    const r = computeSSMInput(flat, T_A, T_B, D, P);
    self.postMessage({ type: 'ssm_input_result', jobId, sim: r.sim.buffer,
      pA: r.pA, pB: r.pB, T_A: r.T_A, T_B: r.T_B }, [r.sim.buffer]);
    return;
  }
  if (msg.type !== 'compute_ssm') return;
  const { jobId, flat, T_A, T_B, D, scope, which, P, metric } = msg;

  const budget = Math.max(2, P | 0);
  const cosine = metric !== 'euclidean';

  // Pool each source independently (never across the A/B seam).
  const pA = T_A > 0 ? Math.min(budget, T_A) : 0;
  const pB = T_B > 0 ? Math.min(budget, T_B) : 0;

  // Choose the row-source and col-source per scope.
  // rowsX/colsX are pooled matrices; rowP/colP their counts.
  let rowsX, colsX, rowP, colP, seam = -1;

  if (scope === 'cross') {
    rowsX = poolRows(flat, 0, T_A, D, pA);  rowP = pA;
    colsX = poolRows(flat, T_A, T_B, D, pB); colP = pB;
  } else if (scope === 'self') {
    const useA = which !== 'B';
    const start = useA ? 0 : T_A;
    const T = useA ? T_A : T_B;
    const p = useA ? pA : pB;
    rowsX = poolRows(flat, start, T, D, p); rowP = p;
    colsX = rowsX; colP = p;
  } else { // joint — concat pooled A then pooled B
    const a = poolRows(flat, 0, T_A, D, pA);
    const b = poolRows(flat, T_A, T_B, D, pB);
    const M = pA + pB;
    const cat = new Float32Array(M * D);
    cat.set(a, 0);
    cat.set(b, pA * D);
    rowsX = cat; colsX = cat; rowP = M; colP = M;
    seam = pA; // first B index
  }

  if (rowP === 0 || colP === 0) {
    self.postMessage({ type: 'ssm_result', jobId, empty: true });
    return;
  }

  const symmetric = rowsX === colsX;

  // Prep per metric.
  let rNorm2, cNorm2;
  if (cosine) {
    l2normRows(rowsX, rowP, D);
    if (!symmetric) l2normRows(colsX, colP, D);
  } else {
    rNorm2 = rowNorms2(rowsX, rowP, D);
    cNorm2 = symmetric ? rNorm2 : rowNorms2(colsX, colP, D);
  }

  // Compute similarity into a Float32 buffer, tracking min/max for normalisation.
  const S = new Float32Array(rowP * colP);
  let vmin = Infinity, vmax = -Infinity;

  for (let i = 0; i < rowP; i++) {
    const jStart = symmetric ? i : 0;
    for (let j = jStart; j < colP; j++) {
      let v;
      if (cosine) {
        v = dot(rowsX, i, colsX, j, D);
      } else {
        // negative distance → larger = more similar
        const d2 = rNorm2[i] + cNorm2[j] - 2 * dot(rowsX, i, colsX, j, D);
        v = -Math.sqrt(Math.max(0, d2));
      }
      S[i * colP + j] = v;
      if (symmetric && j !== i) S[j * colP + i] = v;
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
    }
  }

  // Colour-map to RGBA. Cosine uses a fixed [-1,1] domain so colours mean the
  // same thing across files; euclidean has no fixed scale, so it auto-ranges.
  const lo = cosine ? -1 : vmin;
  const hi = cosine ? 1 : vmax;
  const span = hi - lo > 1e-9 ? hi - lo : 1;
  const rgba = new Uint8ClampedArray(rowP * colP * 4);
  for (let k = 0; k < S.length; k++) magma((S[k] - lo) / span, rgba, k * 4);

  self.postMessage({
    type: 'ssm_result', jobId,
    rgba: rgba.buffer, sim: S.buffer, w: colP, h: rowP,
    scope, seam, pA, pB, T_A, T_B,
    vmin, vmax, metric: cosine ? 'cosine' : 'euclidean',
  }, [rgba.buffer, S.buffer]);
};
