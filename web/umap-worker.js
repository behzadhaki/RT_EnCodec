import { UMAP } from 'https://esm.sh/umap-js';

// Mulberry32 — fast 32-bit seeded PRNG.
function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Override Math.random at module level so umap-js picks it up via
// `this.random = Math.random` in its constructor. Workers have their own
// global scope so this doesn't affect the main thread.
Math.random = mulberry32(42);

self.onmessage = ({ data: msg }) => {
  if (msg.type !== 'compute_umap') return;

  const { jobId, flat, T_A, T_B, D, nNeighbors, minDist } = msg;

  try {
    // Reconstruct number[][] from flat Float32Array
    const n = T_A + T_B;
    const all = [];
    for (let i = 0; i < n; i++) {
      all.push(Array.from(flat.subarray(i * D, (i + 1) * D)));
    }

    const umap    = new UMAP({ nComponents: 2, nNeighbors, minDist });
    const nEpochs = umap.initializeFit(all);

    for (let e = 0; e < nEpochs; e++) {
      umap.step();
      if (e % 10 === 0 || e === nEpochs - 1) {
        self.postMessage({ type: 'umap_progress', jobId, epoch: e, nEpochs });
      }
    }

    const coords = umap.getEmbedding(); // number[][]
    const result = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      result[i * 2]     = coords[i][0];
      result[i * 2 + 1] = coords[i][1];
    }

    self.postMessage({ type: 'umap_result', jobId, result, T_A, T_B }, [result.buffer]);
  } catch (err) {
    self.postMessage({ type: 'umap_error', jobId, message: err.message });
  }
};
