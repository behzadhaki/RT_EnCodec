/**
 * snac-worker.js — SNAC ONNX inference in a Web Worker (onnxruntime-web / wasm).
 *
 * Mirrors the message-protocol shape of encodec-worker.js but is much smaller:
 * SNAC is fully convolutional (no LSTM state) and has no bandwidth axis — just
 * N multi-scale codebook levels. The codes stay a LIST of per-level tensors and
 * decode_codes returns the per-level embedding contributions UNSUMMED, so level
 * mute/solo is just "re-sum the kept levels + decode_audio" — no re-encode.
 *
 * Processing modes (set on 'encode'):
 *   - whole  : encode/decode the entire clip in one pass (exact, memory ∝ length)
 *   - chunked: split into fixed-size chunks; each chunk is an independent
 *              encode→codes→decode unit. SNAC convs are non-causal, so chunk
 *              edges lack neighbour context → seam artifacts. The OLA option
 *              overlaps chunks and linearly crossfades them to smooth the seams.
 *   Chunk length / overlap are snapped to preprocess_pad multiples so the
 *   multi-scale level strides stay aligned.
 *
 * Inbound:
 *   { type: 'init',   modelsBaseUrl? }
 *   { type: 'encode', jobId, audio: Float32Array (mono @ model SR), model,
 *                     chunk: { lenSec, olaSec } | null }   // null => whole
 *   { type: 'decode', jobId, keep: boolean[] }             // which levels to sum
 *
 * Outbound:
 *   { type: 'ready' }
 *   { type: 'encoded', jobId, model, sampleRate, frameRate, origLen, nLevels,
 *                      levels: [{ index, T, stride, preview }],
 *                      chunking: { mode, nChunks, chunkSamples, olaSamples } }
 *   { type: 'result',  jobId, decoded: Float32Array }      // trimmed to origLen
 *   { type: 'progress', jobId, value, status }
 *   { type: 'error',   jobId, message }
 */

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js');

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
ort.env.wasm.numThreads = 1;   // avoid SharedArrayBuffer requirement

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let modelsBaseUrl = '/serialization/snac_onnx_exports';
let cachedModel = null;
let sessions = null;   // { encSeg, quantEnc, decCodes, decAudio }
let meta = null;       // model.json
// cache.chunks: [{ start, zq: Float32Array[], zqDims: [1,D,Tb] }]
let cache = null;

// ---------------------------------------------------------------------------
// Session loading
// ---------------------------------------------------------------------------

let _loadPromise = null, _loadModel = null;

// Single-flight: concurrent callers for the same model share ONE load. Without
// this, rapid decode requests on a cold worker each kick off a separate ~75MB
// model load → memory blows up and the worker wedges (only the first job ever
// completes). Calls for an already-loaded model return immediately.
async function ensureSessions(model, jobId) {
  if (model === cachedModel && sessions) return;
  if (_loadPromise && _loadModel === model) return _loadPromise;
  _loadModel = model;
  _loadPromise = (async () => {
    const base = `${modelsBaseUrl}/${model}`;
    progress(jobId, 0.03, 'Loading model layout…');
    const m = await (await fetch(`${base}/model.json`)).json();
    progress(jobId, 0.06, 'Loading ONNX graphs…');
    const s = {
      encSeg:   await ort.InferenceSession.create(`${base}/encode_audio_segment.onnx`),
      quantEnc: await ort.InferenceSession.create(`${base}/quantize_encodings.onnx`),
      decCodes: await ort.InferenceSession.create(`${base}/decode_codes.onnx`),
      decAudio: await ort.InferenceSession.create(`${base}/decode_audio.onnx`),
    };
    meta = m; sessions = s; cachedModel = model;
  })();
  try { await _loadPromise; } finally { _loadPromise = null; }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function progress(jobId, value, status) {
  self.postMessage({ type: 'progress', jobId, value, status });
}

function roundToMultiple(x, m) {
  return Math.max(m, Math.round(x / m) * m);
}

// Plan chunk start offsets over [0, paddedLen). hop = chunkSamples - olaSamples.
function planChunks(paddedLen, chunkSamples, olaSamples) {
  if (!chunkSamples || chunkSamples >= paddedLen) {
    return { starts: [0], chunkSamples: paddedLen, olaSamples: 0 };
  }
  const hop = chunkSamples - olaSamples;
  const starts = [];
  for (let s = 0; s < paddedLen; s += hop) starts.push(s);
  return { starts, chunkSamples, olaSamples };
}

// Per-chunk crossfade window: linear fade in/out over the overlap, flat 1 in
// the middle. Never exactly 0 (so the normalized accumulator below is safe).
function olaWindow(chunkSamples, olaSamples) {
  const w = new Float32Array(chunkSamples).fill(1);
  for (let k = 0; k < olaSamples; k++) {
    const v = (k + 1) / (olaSamples + 1);
    w[k] = v;                       // fade in
    w[chunkSamples - 1 - k] = v;    // fade out
  }
  return w;
}

// ---------------------------------------------------------------------------
// Encode: audio -> per-chunk per-level zq cache
// ---------------------------------------------------------------------------

async function encode(audio, model, chunkCfg, jobId) {
  await ensureSessions(model, jobId);
  const N = meta.n_codebooks;
  const padTo = meta.preprocess_pad;
  const origLen = audio.length;
  const paddedLen = Math.ceil(origLen / padTo) * padTo;

  // resolve chunking (snap to padTo multiples)
  let chunkSamples = 0, olaSamples = 0, mode = 'whole';
  if (chunkCfg && chunkCfg.lenSec > 0) {
    mode = 'chunked';
    chunkSamples = Math.min(paddedLen, roundToMultiple(chunkCfg.lenSec * meta.sampling_rate, padTo));
    if (chunkCfg.olaSec > 0) {
      olaSamples = roundToMultiple(chunkCfg.olaSec * meta.sampling_rate, padTo);
      olaSamples = Math.min(olaSamples, chunkSamples - padTo);  // keep hop ≥ padTo
      if (olaSamples < 0) olaSamples = 0;
    }
  }
  const plan = planChunks(paddedLen, chunkSamples, olaSamples);

  const chunks = [];
  const seg = new Float32Array(plan.chunkSamples);
  for (let ci = 0; ci < plan.starts.length; ci++) {
    const start = plan.starts[ci];
    progress(jobId, 0.1 + 0.85 * ci / plan.starts.length,
             `Encoding chunk ${ci + 1}/${plan.starts.length}…`);
    seg.fill(0);
    const srcEnd = Math.min(start + plan.chunkSamples, audio.length);
    if (start < audio.length) seg.set(audio.subarray(start, srcEnd));

    const audioT = new ort.Tensor('float32', seg, [1, 1, plan.chunkSamples]);
    const { z } = await sessions.encSeg.run({ audio: audioT });
    const qOut = await sessions.quantEnc.run({ z });
    const codeTensors = {};
    for (let i = 0; i < N; i++) codeTensors[`codes_${i}`] = qOut[`codes_${i}`];
    const dOut = await sessions.decCodes.run(codeTensors);

    const zq = [];
    let zqDims = null;
    for (let i = 0; i < N; i++) { const t = dOut[`zq_${i}`]; zq.push(t.data); zqDims = t.dims; }
    chunks.push({ start, zq, zqDims });

    if (ci === 0) var firstCodes = codeTensors;  // for preview/level table
  }

  cache = { model, origLen, paddedLen, chunkSamples: plan.chunkSamples,
            olaSamples: plan.olaSamples, chunks };

  // level table: stride + whole-file-equivalent token count + preview (chunk 0)
  const baseFrames = paddedLen / meta.hop_length;
  const levels = [];
  for (let i = 0; i < N; i++) {
    const t = firstCodes[`codes_${i}`];
    const preview = Array.from(t.data.slice(0, Math.min(16, t.data.length)), (v) => Number(v));
    levels.push({ index: i, stride: meta.vq_strides[i],
                  T: Math.ceil(baseFrames / meta.vq_strides[i]), preview });
  }

  return {
    model, sampleRate: meta.sampling_rate, frameRate: meta.frame_rate,
    origLen, nLevels: N, levels,
    chunking: { mode, nChunks: plan.starts.length,
                chunkSamples: plan.chunkSamples, olaSamples: plan.olaSamples },
  };
}

// ---------------------------------------------------------------------------
// Decode: per chunk sum kept levels -> decode_audio -> OLA reassemble -> trim
// ---------------------------------------------------------------------------

async function decode(keep, jobId) {
  if (!cache) throw new Error('Nothing encoded yet — send an "encode" message first.');
  const { paddedLen, chunkSamples, olaSamples, chunks } = cache;
  const out = new Float32Array(paddedLen);
  const wsum = new Float32Array(paddedLen);
  const win = olaWindow(chunkSamples, olaSamples);

  for (let ci = 0; ci < chunks.length; ci++) {
    const ch = chunks[ci];
    progress(jobId, 0.1 + 0.85 * ci / chunks.length, `Decoding chunk ${ci + 1}/${chunks.length}…`);
    const [B, D, Tb] = ch.zqDims;
    const total = B * D * Tb;
    const sum = new Float32Array(total);
    for (let i = 0; i < ch.zq.length; i++) {
      if (keep && keep[i] === false) continue;
      const zi = ch.zq[i];
      for (let k = 0; k < total; k++) sum[k] += zi[k];
    }
    const zqT = new ort.Tensor('float32', sum, ch.zqDims);
    const { audio } = await sessions.decAudio.run({ z_q: zqT });
    const a = audio.data;  // length == chunkSamples

    const n = Math.min(a.length, paddedLen - ch.start);
    for (let k = 0; k < n; k++) {
      const w = win[k];
      out[ch.start + k] += a[k] * w;
      wsum[ch.start + k] += w;
    }
  }

  // normalize the crossfaded overlaps, then trim to original length
  for (let k = 0; k < paddedLen; k++) if (wsum[k] > 0) out[k] /= wsum[k];
  return new Float32Array(out.subarray(0, cache.origLen));
}

// ===========================================================================
// Experiment-4 support: per-source embedding capture + SVD blend.
// SNAC's continuous embedding is single-rate, so exp4's SVD-on-[C,T] machinery
// ports directly. encoder_latents = z (pre-quant); quantized_embeddings = z_q
// (summed levels). Decode-from-embedding uses decode_audio. Whole-file only.
// ===========================================================================

let svdCaptures = { A: null, B: null };   // { zData,zDims, zqData,zqDims, origLen, svd? }

// Encode a whole clip → cache its z and summed z_q, return decoded preview.
async function encodeSourceWhole(audio, model, source, jobId) {
  await ensureSessions(model, jobId);
  const N = meta.n_codebooks;
  const padTo = meta.preprocess_pad;
  const origLen = audio.length;
  const paddedLen = Math.ceil(origLen / padTo) * padTo;
  const seg = new Float32Array(paddedLen);
  seg.set(audio.subarray(0, Math.min(origLen, paddedLen)));

  progress(jobId, 0.4, `Encoding ${source}…`);
  const { z } = await sessions.encSeg.run({ audio: new ort.Tensor('float32', seg, [1, 1, paddedLen]) });
  const qOut = await sessions.quantEnc.run({ z });
  const codeTensors = {};
  for (let i = 0; i < N; i++) codeTensors[`codes_${i}`] = qOut[`codes_${i}`];
  const dOut = await sessions.decCodes.run(codeTensors);

  const [B, D, Tb] = dOut.zq_0.dims;
  const zqSum = new Float32Array(B * D * Tb);
  for (let i = 0; i < N; i++) { const zi = dOut[`zq_${i}`].data; for (let k = 0; k < zqSum.length; k++) zqSum[k] += zi[k]; }

  svdCaptures[source] = {
    zData: new Float32Array(z.data), zDims: z.dims.slice(),
    zqData: zqSum, zqDims: [B, D, Tb], origLen,
  };

  progress(jobId, 0.8, `Decoding ${source}…`);
  const { audio: dec } = await sessions.decAudio.run({ z_q: new ort.Tensor('float32', zqSum, [B, D, Tb]) });
  return new Float32Array(dec.data.subarray(0, origLen));
}

// Encode one clip → continuous embeddings for experiment6 (no decode).
// Returns z (embEnc), summed z_q (embQuant), and per-level codes. Whole-file.
async function encodeEmb(audio, model, jobId) {
  await ensureSessions(model, jobId);
  const N = meta.n_codebooks, padTo = meta.preprocess_pad;
  const origLen = audio.length, paddedLen = Math.ceil(origLen / padTo) * padTo;
  const seg = new Float32Array(paddedLen);
  seg.set(audio.subarray(0, Math.min(origLen, paddedLen)));

  const { z } = await sessions.encSeg.run({ audio: new ort.Tensor('float32', seg, [1, 1, paddedLen]) });
  const qOut = await sessions.quantEnc.run({ z });
  const codeTensors = {}, levels = [];
  for (let i = 0; i < N; i++) {
    const t = qOut[`codes_${i}`];
    codeTensors[`codes_${i}`] = t;
    levels.push({ data: t.data, stride: meta.vq_strides[i] });
  }
  const dOut = await sessions.decCodes.run(codeTensors);
  const [B, D, Tb] = dOut.zq_0.dims;
  const zqSum = new Float32Array(B * D * Tb);
  for (let i = 0; i < N; i++) { const zi = dOut[`zq_${i}`].data; for (let k = 0; k < zqSum.length; k++) zqSum[k] += zi[k]; }

  // Pack per-level codes into [1, N, Tb] int32, each level upsampled to base rate.
  const codes = new Int32Array(N * Tb);
  for (let i = 0; i < N; i++) {
    const ld = levels[i].data, st = levels[i].stride;
    for (let t = 0; t < Tb; t++) codes[i * Tb + t] = Number(ld[Math.floor(t / st)]);
  }
  // raw per-level codes (ragged, original rate) for file caching
  const rawLevels = levels.map(l => Int32Array.from(l.data, Number));

  return {
    zData: new Float32Array(z.data), zDims: z.dims.slice(),
    zqData: zqSum, zqDims: [B, D, Tb],
    codes, codesDims: [1, N, Tb], rawLevels,
  };
}

// Rebuild embQuant (z_q) + packed codes from cached per-level codes — no encode.
async function reconEmb(levels, model, jobId) {
  await ensureSessions(model, jobId);
  const N = meta.n_codebooks;
  const codeTensors = {};
  for (let i = 0; i < N; i++) {
    const lv = levels[i];
    const big = BigInt64Array.from(lv, v => BigInt(v));
    codeTensors[`codes_${i}`] = new ort.Tensor('int64', big, [1, lv.length]);
  }
  const dOut = await sessions.decCodes.run(codeTensors);
  const [B, D, Tb] = dOut.zq_0.dims;
  const zqSum = new Float32Array(B * D * Tb);
  for (let i = 0; i < N; i++) { const zi = dOut[`zq_${i}`].data; for (let k = 0; k < zqSum.length; k++) zqSum[k] += zi[k]; }
  const codes = new Int32Array(N * Tb);
  for (let i = 0; i < N; i++) {
    const lv = levels[i], st = meta.vq_strides[i];
    for (let t = 0; t < Tb; t++) codes[i * Tb + t] = lv[Math.floor(t / st)];
  }
  return { zqData: zqSum, zqDims: [B, D, Tb], codes, codesDims: [1, N, Tb] };
}

// Decode an arbitrary-length embedding [1,D,N] → audio (trimmed to N frames).
// The 32k/44k decoder has windowed attention that requires N to be a multiple
// of attn_window_size; trajectory/frame windows are arbitrary, so pad the time
// dim up (replicating the last frame) and trim the extra output samples.
async function decodeEmbAudio(embFlat, dims) {
  const [B, D, N] = dims;
  const aw = meta.attn_window_size;
  let inFlat = embFlat, inN = N;
  if (aw && N % aw !== 0) {
    inN = Math.ceil(N / aw) * aw;
    const padded = new Float32Array(B * D * inN);
    for (let d = 0; d < D; d++) {
      const src = d * N, dst = d * inN;
      for (let t = 0; t < N; t++) padded[dst + t] = embFlat[src + t];
      const last = embFlat[src + N - 1];
      for (let t = N; t < inN; t++) padded[dst + t] = last;   // replicate last frame
    }
    inFlat = padded;
  }
  const { audio } = await sessions.decAudio.run({ z_q: new ort.Tensor('float32', inFlat, [B, D, inN]) });
  const valid = N * meta.hop_length;   // trim the padded tail
  return new Float32Array(audio.data.subarray(0, valid));
}

function capEmb(source, applyPoint) {
  const c = svdCaptures[source];
  return applyPoint === 'encoder_latents'
    ? { emb: c.zData, dims: c.zDims }
    : { emb: c.zqData, dims: c.zqDims };
}

// Jacobi eigen-decomposition of a symmetric C×C matrix (ported from encodec-worker).
function symmetricEigen(flatA, n, maxSweeps = 8) {
  const A = flatA.slice();
  const V = new Float32Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1.0;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let offNorm = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) offNorm += A[p*n+q] * A[p*n+q];
    if (offNorm < 1e-20) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      const Apq = A[p*n+q]; if (Math.abs(Apq) < 1e-12) continue;
      const App = A[p*n+p], Aqq = A[q*n+q];
      const tau = (Aqq - App) / (2 * Apq);
      const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau*tau));
      const c = 1 / Math.sqrt(1 + t*t), s = t * c;
      A[p*n+p] = App - t*Apq; A[q*n+q] = Aqq + t*Apq; A[p*n+q] = A[q*n+p] = 0;
      for (let r = 0; r < n; r++) { if (r===p||r===q) continue;
        const Arp = A[r*n+p], Arq = A[r*n+q];
        A[r*n+p] = A[p*n+r] = c*Arp - s*Arq; A[r*n+q] = A[q*n+r] = s*Arp + c*Arq; }
      for (let r = 0; r < n; r++) { const Vrp = V[r*n+p], Vrq = V[r*n+q];
        V[r*n+p] = c*Vrp - s*Vrq; V[r*n+q] = s*Vrp + c*Vrq; }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[b*n+b] - A[a*n+a]);
  const eigenvalues = new Float32Array(n), eigenvectors = new Float32Array(n * n);
  for (let k = 0; k < n; k++) {
    eigenvalues[k] = Math.max(0, A[order[k]*n+order[k]]);
    for (let r = 0; r < n; r++) eigenvectors[r*n+k] = V[r*n+order[k]];
  }
  return { eigenvalues, eigenvectors };
}

// Full SVD of one [C, totalT] embedding (second-moment, no centering).
function computeFullSVD(emb, dims, totalT) {
  const C = dims[1], T = dims[2];
  const cov = new Float32Array(C * C);
  for (let i = 0; i < C; i++) for (let j = i; j < C; j++) {
    let s = 0; for (let t = 0; t < totalT; t++) s += emb[i*T + t] * emb[j*T + t];
    cov[i*C+j] = cov[j*C+i] = s / totalT;
  }
  const { eigenvalues, eigenvectors } = symmetricEigen(cov, C, 8);
  const S = new Float32Array(C);
  for (let k = 0; k < C; k++) S[k] = Math.sqrt(Math.max(0, eigenvalues[k]));
  const Vt = new Float32Array(C * totalT);
  for (let k = 0; k < C; k++) {
    const sk = Math.max(S[k], 1e-10);
    for (let t = 0; t < totalT; t++) {
      let v = 0; for (let c = 0; c < C; c++) v += eigenvectors[c*C + k] * emb[c*T + t];
      Vt[k*totalT + t] = v / sk;
    }
  }
  return { U: eigenvectors, S, Vt, C, totalT };
}

function runComputeSVD4(msg) {
  try {
    const applyPoint = msg.applyPoint || 'encoder_latents';
    if (!svdCaptures.A || !svdCaptures.B) {
      self.postMessage({ type: 'svd4_error', message: 'Encode both sources first.' }); return;
    }
    const eA = capEmb('A', applyPoint), eB = capEmb('B', applyPoint);
    const totalT = Math.min(eA.dims[2], eB.dims[2]);
    self.postMessage({ type: 'svd4_progress', message: 'SVD A…', value: 0.1 });
    svdCaptures.A.svd = computeFullSVD(eA.emb, eA.dims, totalT);
    self.postMessage({ type: 'svd4_progress', message: 'SVD B…', value: 0.55 });
    svdCaptures.B.svd = computeFullSVD(eB.emb, eB.dims, totalT);
    const sA = svdCaptures.A.svd.S.slice(), sB = svdCaptures.B.svd.S.slice();
    self.postMessage({ type: 'svd4_ready', sA, sB, C: svdCaptures.A.svd.C, totalT }, [sA.buffer, sB.buffer]);
  } catch (err) {
    self.postMessage({ type: 'svd4_error', message: 'SVD failed: ' + err.message });
  }
}

async function runApplySVD4(msg) {
  const { jobId, applyPoint = 'encoder_latents', compMask, mode } = msg;
  const a = svdCaptures.A.svd, b = svdCaptures.B.svd;
  const C = a.C, totalT = a.totalT;
  const aU = msg.alphaU || 0, aS = msg.alphaS || 0, aV = msg.alphaV || 0, aUS = msg.alphaUS || 0;

  // E_new[c,t] = Σ_active-k Ueff[c,k]·Seff[k]·Veff[k,t]   (or US combined)
  const E = new Float32Array(C * totalT);
  for (let k = 0; k < C; k++) {
    if (compMask && !compMask[k]) continue;
    const SA = a.S[k], SB = b.S[k];
    for (let t = 0; t < totalT; t++) {
      const VA = a.Vt[k*totalT + t], VB = b.Vt[k*totalT + t];
      const Veff = (1 - aV) * VA + aV * VB;
      for (let c = 0; c < C; c++) {
        const UA = a.U[c*C + k], UB = b.U[c*C + k];
        let coeff;
        if (mode === 'us_v') coeff = ((1 - aUS) * (UA * SA) + aUS * (UB * SB)) * Veff;
        else { const Ueff = (1 - aU) * UA + aU * UB, Seff = (1 - aS) * SA + aS * SB; coeff = Ueff * Seff * Veff; }
        E[c*totalT + t] += coeff;
      }
    }
  }

  progress(jobId, 0.6, 'Decoding…');
  const N = meta.n_codebooks;
  let zq = E, zqDims = [1, C, totalT];
  if (applyPoint === 'encoder_latents') {
    // re-quantize the blended latent, then re-expand to z_q
    const qOut = await sessions.quantEnc.run({ z: new ort.Tensor('float32', E, [1, C, totalT]) });
    const codeTensors = {}; for (let i = 0; i < N; i++) codeTensors[`codes_${i}`] = qOut[`codes_${i}`];
    const dOut = await sessions.decCodes.run(codeTensors);
    const [B, D, Tb] = dOut.zq_0.dims;
    zq = new Float32Array(B * D * Tb); zqDims = [B, D, Tb];
    for (let i = 0; i < N; i++) { const zi = dOut[`zq_${i}`].data; for (let kk = 0; kk < zq.length; kk++) zq[kk] += zi[kk]; }
  }
  const { audio } = await sessions.decAudio.run({ z_q: new ort.Tensor('float32', zq, zqDims) });
  return new Float32Array(audio.data);
}

// ---------------------------------------------------------------------------
// Message router
// ---------------------------------------------------------------------------

self.onmessage = async (e) => {
  const msg = e.data;
  const jobId = msg.jobId;
  try {
    switch (msg.type) {
      case 'init':
        if (msg.modelsBaseUrl) modelsBaseUrl = msg.modelsBaseUrl;
        self.postMessage({ type: 'ready' });
        break;

      case 'encode': {
        // Two callers: snac.html (no source → 'encoded' + level cache) and
        // experiment4 (source 'A'/'B' → whole-file capture + decoded preview).
        if (msg.source) {
          const decoded = await encodeSourceWhole(msg.audio, msg.model, msg.source, jobId);
          progress(jobId, 1, `${msg.source} encoded.`);
          self.postMessage({ type: 'result', jobId, decoded }, [decoded.buffer]);
        } else {
          const info = await encode(msg.audio, msg.model, msg.chunk, jobId);
          progress(jobId, 1, 'Encoded.');
          self.postMessage({ type: 'encoded', jobId, ...info });
        }
        break;
      }

      case 'encode_exp6': {
        // experiment6: encode both sources → embeddings_exp6 (z=embEnc, z_q=embQuant).
        progress(jobId, 0.2, 'Encoding A…');
        const A = await encodeEmb(msg.audioA, msg.model, jobId);
        progress(jobId, 0.6, 'Encoding B…');
        const B = await encodeEmb(msg.audioB, msg.model, jobId);
        progress(jobId, 1, 'Embeddings ready');
        self.postMessage({
          type: 'embeddings_exp6', jobId,
          embEncA: A.zData, embEncDimsA: A.zDims, embEncB: B.zData, embEncDimsB: B.zDims,
          embQuantA: A.zqData, embQuantDimsA: A.zqDims, embQuantB: B.zqData, embQuantDimsB: B.zqDims,
          codesA: A.codes, codesDimsA: A.codesDims, codesB: B.codes, codesDimsB: B.codesDims,
          scalesA: null, scalesB: null,
          // raw per-level codes + layout for file caching
          rawLevelsA: A.rawLevels, rawLevelsB: B.rawLevels,
          frameRate: meta.frame_rate, vqStrides: meta.vq_strides,
        });
        break;
      }

      case 'warm':
        // Pre-load sessions on the primary worker so the first trajectory/frame
        // decode isn't a cold ~75MB load (which raced with rapid redraws).
        await ensureSessions(msg.model, jobId);
        break;

      case 'decode_window_snac': {
        // One embedding chunk for the parallel trajectory decode pool.
        await ensureSessions(msg.model, jobId);
        const audio = await decodeEmbAudio(msg.embQuant, msg.dims);
        self.postMessage({ type: 'decode_window_done', jobId, idx: msg.idx, audio }, [audio.buffer]);
        break;
      }

      case 'encode_window_snac': {
        // One chunk window for the experiment6 parallel pool. Returns the full
        // window's z / z_q / per-level codes; the main thread trims to central.
        const r = await encodeEmb(msg.audio, msg.model, jobId);
        self.postMessage({
          type: 'window_snac', jobId, idx: msg.idx,
          z: r.zData, zDims: r.zDims, zq: r.zqData, zqDims: r.zqDims, rawLevels: r.rawLevels,
        }, [r.zData.buffer, r.zqData.buffer]);
        break;
      }

      case 'recon_exp6': {
        // Rebuild both sources' embeddings from cached per-level codes (no encode).
        progress(jobId, 0.3, 'Reconstructing from cache…');
        const A = await reconEmb(msg.levelsA, msg.model, jobId);
        const B = await reconEmb(msg.levelsB, msg.model, jobId);
        progress(jobId, 1, 'Cache loaded');
        self.postMessage({
          type: 'embeddings_exp6', jobId,
          embEncA: null, embEncDimsA: null, embEncB: null, embEncDimsB: null,
          embQuantA: A.zqData, embQuantDimsA: A.zqDims, embQuantB: B.zqData, embQuantDimsB: B.zqDims,
          codesA: A.codes, codesDimsA: A.codesDims, codesB: B.codes, codesDimsB: B.codesDims,
          scalesA: null, scalesB: null,
        }, [A.zqData.buffer, B.zqData.buffer, A.codes.buffer, B.codes.buffer]);
        break;
      }

      case 'decode_trajectory_exp6': {
        // ensureSessions: the primary worker may not have encoded (chunked uses the pool).
        await ensureSessions(msg.model, jobId);
        const decoded = await decodeEmbAudio(msg.embQuant, msg.dims);
        self.postMessage({ type: 'trajectory_audio_exp6', jobId, decoded, channels: 1 }, [decoded.buffer]);
        break;
      }

      case 'decode_frame_snac': {
        await ensureSessions(msg.model, jobId);
        const decoded = await decodeEmbAudio(msg.embQuant, msg.dims);
        self.postMessage({ type: 'frame_audio_exp6', jobId, decoded, channels: 1 }, [decoded.buffer]);
        break;
      }

      case 'compute_svd4':
        runComputeSVD4(msg);
        break;

      case 'apply_svd4': {
        const decoded = await runApplySVD4(msg);
        progress(jobId, 1, 'Done.');
        self.postMessage({ type: 'result', jobId, decoded }, [decoded.buffer]);
        break;
      }

      case 'cancel':
        break;

      case 'decode': {
        const decoded = await decode(msg.keep, jobId);
        progress(jobId, 1, 'Done.');
        self.postMessage({ type: 'result', jobId, decoded }, [decoded.buffer]);
        break;
      }

      default:
        throw new Error(`Unknown message type: ${msg.type}`);
    }
  } catch (err) {
    self.postMessage({ type: 'error', jobId, message: err && (err.message || err.name) ? (err.message || err.name) : String(err) });
  }
};
