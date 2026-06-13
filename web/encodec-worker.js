/**
 * encodec-worker.js
 *
 * Web Worker that loads onnxruntime-web and runs the four-stage EnCodec pipeline.
 * Sessions are loaded lazily and cached per (modelHz, bwKbps) pair.
 *
 * Messages IN
 * -----------
 *   { type: 'init',    modelsBaseUrl }
 *   { type: 'process', jobId, audio: Float32Array, modelHz, bwKbps,
 *                      streaming, frameSize }
 *   { type: 'encode',  jobId, source: 'A'|'B', audio: Float32Array,
 *                      modelHz, bwKbps, streaming, frameSize }
 *   { type: 'interpolate', jobId, modelHz, bwKbps, streaming, frameSize,
 *                          alpha, interpPoint, vqMode, levelAlphas }
 *   { type: 'cancel',  jobId }
 *
 * Messages OUT
 * ------------
 *   { type: 'ready' }
 *   { type: 'progress', jobId, value, status }
 *   { type: 'result',   jobId, decoded: Float32Array }
 *   { type: 'error',    jobId, message }
 *
 * Capture cache
 * -------------
 * 'encode' messages populate captureCache.A / captureCache.B inside the worker.
 * Subsequent 'interpolate' messages reuse these captures — no re-encoding.
 * The cache is automatically replaced whenever a fresh 'encode' message arrives.
 */

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js');

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
ort.env.wasm.numThreads = 1;   // avoid SharedArrayBuffer requirement

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LSTM_LAYERS = 2;
const LSTM_HIDDEN = 512;

// ---------------------------------------------------------------------------
// Session cache
// ---------------------------------------------------------------------------

let modelsBaseUrl = '/serialization/encodec_onnx_exports';
let cachedKey = null;
let sessions  = null;   // { encSeg, quantEnc, decCodes, decAudio }

// Codebook cache (keyed by modelHz; shared across all bandwidth variants)
let codebookCache = {};  // { '24k': { embeddings, norms, n_q, vocab_size, dim }, … }

async function ensureSessions(modelHz, bwKbps) {
  const key = `${modelHz}-${bwKbps}`;
  if (key === cachedKey) return;

  const base = `${modelsBaseUrl}/${modelHz}/${bwKbps}kbps`;
  sessions = {
    encSeg:   await ort.InferenceSession.create(`${base}/encode_audio_segment.onnx`),
    quantEnc: await ort.InferenceSession.create(`${base}/quantize_encodings.onnx`),
    decCodes: await ort.InferenceSession.create(`${base}/decode_codes.onnx`),
    decAudio: await ort.InferenceSession.create(`${base}/decode_audio.onnx`),
  };
  cachedKey = key;
}

// ---------------------------------------------------------------------------
// LSTM state helpers
// ---------------------------------------------------------------------------

function zeroState(B) {
  return new Float32Array(LSTM_LAYERS * B * LSTM_HIDDEN);
}

function stateTensor(data, B) {
  return new ort.Tensor('float32', data, [LSTM_LAYERS, B, LSTM_HIDDEN]);
}

// Trim the first `startFrame` frames from a flat [B, D, T] typed-array tensor.
// Works for Float32Array (embeddings) and BigInt64Array (codes) alike.
function sliceFrames(data, dims, startFrame) {
  const [B, D, T] = dims;
  const keep = T - startFrame;
  const dst  = new data.constructor(B * D * keep);
  for (let b = 0; b < B; b++)
    for (let d = 0; d < D; d++)
      dst.set(data.subarray(b*D*T + d*T + startFrame, b*D*T + d*T + T),
              b*D*keep + d*keep);
  return { data: dst, dims: [B, D, keep] };
}

// ---------------------------------------------------------------------------
// Single-chunk pipeline  (used by experiment1 'process' handler only)
// ---------------------------------------------------------------------------

async function processChunk(chunk, modelHz, hEnc, cEnc, hDec, cDec, leftGuardFrames = 0) {
  const T = chunk.length;

  let audioTensor;
  if (modelHz === '48k') {
    const stereo = new Float32Array(2 * T);
    stereo.set(chunk, 0);
    stereo.set(chunk, T);
    audioTensor = new ort.Tensor('float32', stereo, [1, 2, T]);
  } else {
    audioTensor = new ort.Tensor('float32', chunk.slice(), [1, 1, T]);
  }

  const encOut = await sessions.encSeg.run({
    audio: audioTensor,
    h_in: stateTensor(hEnc, 1),
    c_in: stateTensor(cEnc, 1),
  });

  // Discard left guard frames so the VQ and decoder only see valid-region embeddings.
  let embForVQ = encOut.emb;
  if (leftGuardFrames > 0) {
    const { data, dims } = sliceFrames(encOut.emb.data, encOut.emb.dims, leftGuardFrames);
    embForVQ = new ort.Tensor('float32', data, dims);
  }

  const quantOut = await sessions.quantEnc.run({ emb: embForVQ });
  const decCodesOut = await sessions.decCodes.run({ codes: quantOut.codes });

  const decAudioInputs = { emb: decCodesOut.emb, h_in: stateTensor(hDec, 1), c_in: stateTensor(cDec, 1) };
  if (modelHz === '48k') decAudioInputs.scale = encOut.scale;
  const decOut = await sessions.decAudio.run(decAudioInputs);

  let outAudio;
  if (modelHz === '48k') {
    const raw  = decOut.audio.data;
    const tOut = raw.length >> 1;
    outAudio   = new Float32Array(tOut);
    for (let i = 0; i < tOut; i++) outAudio[i] = (raw[i] + raw[tOut + i]) * 0.5;
  } else {
    outAudio = new Float32Array(decOut.audio.data);
  }

  return {
    outAudio,
    hEncNew: new Float32Array(encOut.h_out.data),
    cEncNew: new Float32Array(encOut.c_out.data),
    hDecNew: new Float32Array(decOut.h_out.data),
    cDecNew: new Float32Array(decOut.c_out.data),
  };
}

// ---------------------------------------------------------------------------
// Overlap-add constants
// ---------------------------------------------------------------------------

const HALF_OV_24 = 640;

const SEG_48          = 48000;
const STRIDE_48       = 43200;
const HOP_48          = 320;    // samples per encoder frame
const GUARD_FRAMES_48 = 16;     // guard frames each side (~107 ms at 48 kHz)
const GUARD_48        = GUARD_FRAMES_48 * HOP_48;

const TRI_WIN_48 = new Float32Array(SEG_48);
for (let i = 0; i < SEG_48; i++) {
  const t = (i + 1) / (SEG_48 + 1);
  TRI_WIN_48[i] = 0.5 - Math.abs(t - 0.5);
}

// ---------------------------------------------------------------------------
// Experiment1 job runner  (unchanged — uses processChunk)
// ---------------------------------------------------------------------------

let currentJobId = -1;

async function runJob(msg) {
  const { jobId, audio, modelHz, bwKbps, streaming, frameSize } = msg;

  try {
    self.postMessage({ type: 'progress', jobId, value: 0, status: 'Loading models…' });
    await ensureSessions(modelHz, bwKbps);
    if (currentJobId !== jobId) return;

    const decoded = modelHz === '48k'
      ? await run48k(audio, streaming, jobId)
      : await run24k(audio, streaming, frameSize, jobId);

    if (decoded === null) return;
    self.postMessage({ type: 'result', jobId, decoded }, [decoded.buffer]);

  } catch (err) {
    self.postMessage({ type: 'error', jobId, message: err.message });
  }
}

async function run24k(audio, streaming, frameSize, jobId) {
  if (!streaming) {
    if (currentJobId !== jobId) return null;
    self.postMessage({ type: 'progress', jobId, value: 0.5, status: 'Processing… 50 %' });
    const result = await processChunk(
      audio.slice(), '24k', zeroState(1), zeroState(1), zeroState(1), zeroState(1));
    if (currentJobId !== jobId) return null;
    self.postMessage({ type: 'progress', jobId, value: 1, status: 'Processing… 100 %' });
    return result.outAudio.slice(0, audio.length);
  }

  const stride = frameSize;
  const seg    = stride + 2 * HALF_OV_24;
  const win = new Float32Array(seg);
  for (let i = 0; i < seg; i++) {
    const t = (i + 1) / (seg + 1);
    win[i] = 0.5 - Math.abs(t - 0.5);
  }

  const totalLen = audio.length;
  const outBuf   = new Float32Array(totalLen);
  const wgtBuf   = new Float32Array(totalLen);
  const numSegs  = Math.ceil(totalLen / stride);

  let hEnc = zeroState(1), cEnc = zeroState(1);
  let hDec = zeroState(1), cDec = zeroState(1);

  for (let s = 0; s < numSegs; s++) {
    if (currentJobId !== jobId) return null;

    const off  = s * stride - HALF_OV_24;
    const srcS = Math.max(off, 0);
    const srcE = Math.min(off + seg, totalLen);
    const chunk = new Float32Array(seg);
    if (srcE > srcS) chunk.set(audio.subarray(srcS, srcE), srcS - off);

    const result = await processChunk(chunk, '24k', hEnc, cEnc, hDec, cDec);
    hEnc = result.hEncNew; cEnc = result.cEncNew;
    hDec = result.hDecNew; cDec = result.cDecNew;

    for (let i = srcS - off; i < srcE - off; i++) {
      const pos = off + i;
      outBuf[pos] += win[i] * result.outAudio[i];
      wgtBuf[pos] += win[i];
    }

    self.postMessage({
      type: 'progress', jobId,
      value: Math.min((s + 1) / numSegs, 1),
      status: `Processing… ${Math.round((s + 1) / numSegs * 100)} %`,
    });
  }

  for (let i = 0; i < totalLen; i++) {
    if (wgtBuf[i] > 0) outBuf[i] /= wgtBuf[i];
  }
  return outBuf;
}

async function run48k(audio, streaming, jobId) {
  const totalLen = audio.length;
  const outBuf   = new Float32Array(totalLen);
  const wgtBuf   = new Float32Array(totalLen);
  const numSegs  = Math.ceil(totalLen / STRIDE_48);

  let hEnc = zeroState(1), cEnc = zeroState(1);
  let hDec = zeroState(1), cDec = zeroState(1);

  for (let seg = 0; seg < numSegs; seg++) {
    if (currentJobId !== jobId) return null;

    const offset         = seg * STRIDE_48;
    const leftGuard      = Math.min(offset, GUARD_48);
    const leftGuardFrames = leftGuard / HOP_48;

    const extLen = leftGuard + SEG_48;
    const chunk  = new Float32Array(extLen);
    chunk.set(audio.subarray(offset - leftGuard, Math.min(offset - leftGuard + extLen, totalLen)));

    const result = await processChunk(chunk, '48k', hEnc, cEnc, hDec, cDec, leftGuardFrames);

    hEnc = result.hEncNew; cEnc = result.cEncNew;
    hDec = result.hDecNew; cDec = result.cDecNew;

    const validEnd = Math.min(offset + SEG_48, totalLen);
    for (let i = 0; i < validEnd - offset; i++) {
      outBuf[offset + i] += TRI_WIN_48[i] * result.outAudio[i];
      wgtBuf[offset + i] += TRI_WIN_48[i];
    }

    self.postMessage({
      type: 'progress', jobId,
      value: Math.min((seg + 1) / numSegs, 1),
      status: `Processing… ${Math.round((seg + 1) / numSegs * 100)} %`,
    });
  }

  for (let i = 0; i < totalLen; i++) {
    if (wgtBuf[i] > 0) outBuf[i] /= wgtBuf[i];
  }
  return outBuf;
}

// ---------------------------------------------------------------------------
// Interpolation helpers
// ---------------------------------------------------------------------------

function lerpF32(a, b, alpha) {
  const n = Math.min(a.length, b.length);
  const out = new Float32Array(n);
  const t = 1 - alpha;
  for (let i = 0; i < n; i++) out[i] = a[i] * t + b[i] * alpha;
  return out;
}

// ── VQ code mixing helpers ────────────────────────────────────────────────

function mixCodesElementWise(codes_a, codes_b, alpha, deterministic) {
  const out = new (codes_a.constructor)(codes_a.length);
  for (let i = 0; i < out.length; i++) {
    const frac = deterministic
      ? ((Math.imul(i + 1, 0x9E3779B9) >>> 0) / 0xFFFFFFFF)
      : Math.random();
    out[i] = frac < alpha ? codes_b[i] : codes_a[i];
  }
  return out;
}

function mixCodesByFrame(codes_a, codes_b, dims, alpha, deterministic) {
  const N_q = dims[1], T = dims[2];
  const out = new (codes_a.constructor)(codes_a.length);
  for (let t = 0; t < T; t++) {
    const frac = deterministic
      ? ((Math.imul(t + 1, 0x9E3779B9) >>> 0) / 0xFFFFFFFF)
      : Math.random();
    const useB = frac < alpha;
    for (let k = 0; k < N_q; k++) {
      const idx = k * T + t;
      out[idx] = useB ? codes_b[idx] : codes_a[idx];
    }
  }
  return out;
}

function mixCodesByLevel(codes_a, codes_b, dims, levelAlphas, deterministic) {
  const N_q = dims[1], T = dims[2];
  const out = new (codes_a.constructor)(codes_a.length);
  for (let k = 0; k < N_q; k++) {
    const alpha_k = levelAlphas[k] ?? 0.5;
    for (let t = 0; t < T; t++) {
      const frac = deterministic
        ? ((Math.imul((Math.imul(k + 1, 49999) + t + 1) | 0, 0x9E3779B9) >>> 0) / 0xFFFFFFFF)
        : Math.random();
      const idx  = k * T + t;
      out[idx] = frac < alpha_k ? codes_b[idx] : codes_a[idx];
    }
  }
  return out;
}

async function ensureCodebooks(modelHz) {
  if (codebookCache[modelHz]) return;
  const base = `${modelsBaseUrl}/${modelHz}`;
  const meta = await (await fetch(`${base}/codebooks.json`)).json();
  const { n_q, vocab_size, dim } = meta;
  const buf        = await (await fetch(`${base}/codebooks.bin`)).arrayBuffer();
  const embeddings = new Float32Array(buf);

  // Precompute per-level centrality ranking.
  //
  // score[j] = −||cⱼ − μ||²   (higher = closer to centroid = more "typical")
  //
  // Sum of pairwise squared distances reduces to distance-to-centroid:
  //   Σᵢ ||cⱼ−cᵢ||² = V·||cⱼ||² + Σᵢ||cᵢ||² − 2·cⱼ·Σᵢcᵢ  ∝  ||cⱼ−μ||²
  // So this is O(V·dim) per level, not O(V²·dim).
  //
  // globalRank[k*V + j] = rank of code j at level k  (0=most peripheral, V-1=most central)
  // rankToCode[k*V + r] = code index at rank r for level k
  const globalRank = new Int32Array(n_q * vocab_size);
  const rankToCode = new Int32Array(n_q * vocab_size);
  const centroid   = new Float32Array(dim);
  const scores     = new Float32Array(vocab_size);
  const order      = new Int32Array(vocab_size);

  for (let k = 0; k < n_q; k++) {
    const kBase = k * vocab_size;

    // Centroid of level-k codebook
    centroid.fill(0);
    for (let j = 0; j < vocab_size; j++) {
      const off = (kBase + j) * dim;
      for (let d = 0; d < dim; d++) centroid[d] += embeddings[off + d];
    }
    for (let d = 0; d < dim; d++) centroid[d] /= vocab_size;

    // Score = −||cⱼ − μ||²
    for (let j = 0; j < vocab_size; j++) {
      const off = (kBase + j) * dim;
      let sq = 0;
      for (let d = 0; d < dim; d++) {
        const diff = embeddings[off + d] - centroid[d];
        sq += diff * diff;
      }
      scores[j] = -sq;
      order[j]  = j;
    }

    // Ascending sort: rank 0 = most peripheral, rank V−1 = most central
    order.sort((a, b) => scores[a] - scores[b]);

    for (let r = 0; r < vocab_size; r++) {
      const code = order[r];
      rankToCode[kBase + r]    = code;
      globalRank[kBase + code] = r;
    }
  }

  codebookCache[modelHz] = { embeddings, n_q, vocab_size, dim, globalRank, rankToCode };
}

// Use the precomputed global centrality ranking to interpolate between cA and cB.
// For each (level k, frame t): look up the fixed ranks of cA and cB in the
// centrality ordering, lerp between them, and return the code at that rank.
//
// Rank 0 = most peripheral (furthest from codebook centroid).
// Rank V−1 = most central (closest to centroid, most "typical").
//
// alpha=0 → exactly cA,  alpha=1 → exactly cB,
// alpha=0.5 → the code whose centrality sits midway between cA and cB.
// O(1) per (level, frame) — no per-frame sort.
function mixCodesSimInterp(codes_a, codes_b, dims, levelAlphas, cb) {
  const { vocab_size, globalRank, rankToCode } = cb;
  const N_q   = dims[1], T = dims[2];
  const isBig = codes_a instanceof BigInt64Array;
  const out   = new (codes_a.constructor)(codes_a.length);

  for (let k = 0; k < N_q; k++) {
    const alpha_k = levelAlphas[k] ?? 0.5;
    const kBase   = k * vocab_size;

    for (let t = 0; t < T; t++) {
      const pos = k * T + t;
      const cA  = isBig ? Number(codes_a[pos]) : codes_a[pos];
      const cB  = isBig ? Number(codes_b[pos]) : codes_b[pos];

      if (cA === cB) { out[pos] = codes_a[pos]; continue; }

      const rankA      = globalRank[kBase + cA];
      const rankB      = globalRank[kBase + cB];
      const targetRank = Math.round(rankA + alpha_k * (rankB - rankA));
      out[pos] = isBig ? BigInt(rankToCode[kBase + targetRank]) : rankToCode[kBase + targetRank];
    }
  }
  return out;
}

function mixCodesIntLerpByLevel(codes_a, codes_b, dims, levelAlphas) {
  const N_q   = dims[1], T = dims[2];
  const out   = new (codes_a.constructor)(codes_a.length);
  const isBig = codes_a instanceof BigInt64Array;
  for (let k = 0; k < N_q; k++) {
    const alpha_k = levelAlphas[k] ?? 0.5;
    const w = 1 - alpha_k;
    for (let t = 0; t < T; t++) {
      const idx = k * T + t;
      if (isBig) {
        out[idx] = BigInt(Math.round(Number(codes_a[idx]) * w + Number(codes_b[idx]) * alpha_k));
      } else {
        out[idx] = Math.round(codes_a[idx] * w + codes_b[idx] * alpha_k);
      }
    }
  }
  return out;
}

// Run stages 1–3 and return all intermediate tensors needed for any
// interpolation point, plus the updated encoder LSTM state.
// channels = 2: chunk is already planar stereo [2*T] (L plane, R plane) and is
// fed to the 48k encoder as real L/R. channels = 1 on 48k: dual-mono duplicate.
async function encodeCapture(chunk, modelHz, hEnc, cEnc, leftGuardFrames = 0, channels = 1) {
  let audioTensor;
  if (modelHz === '48k') {
    if (channels === 2) {
      const T = chunk.length / 2;
      audioTensor = new ort.Tensor('float32', chunk.slice(), [1, 2, T]);
    } else {
      const T = chunk.length;
      const stereo = new Float32Array(2 * T);
      stereo.set(chunk, 0); stereo.set(chunk, T);
      audioTensor = new ort.Tensor('float32', stereo, [1, 2, T]);
    }
  } else {
    const T = chunk.length;
    audioTensor = new ort.Tensor('float32', chunk.slice(), [1, 1, T]);
  }

  const encOut = await sessions.encSeg.run({
    audio: audioTensor,
    h_in:  stateTensor(hEnc, 1),
    c_in:  stateTensor(cEnc, 1),
  });

  const quantOut     = await sessions.quantEnc.run({ emb: encOut.emb });
  const decCodesOut  = await sessions.decCodes.run({ codes: quantOut.codes });

  let embEncData   = new Float32Array(encOut.emb.data);
  let embEncDims   = encOut.emb.dims.slice();
  let codesData    = quantOut.codes.data.slice();
  let codesDims    = quantOut.codes.dims.slice();
  let embQuantData = new Float32Array(decCodesOut.emb.data);
  let embQuantDims = decCodesOut.emb.dims.slice();

  if (leftGuardFrames > 0) {
    const e = sliceFrames(embEncData,   embEncDims,   leftGuardFrames);
    embEncData = e.data; embEncDims = e.dims;
    const c = sliceFrames(codesData,    codesDims,    leftGuardFrames);
    codesData = c.data; codesDims = c.dims;
    const q = sliceFrames(embQuantData, embQuantDims, leftGuardFrames);
    embQuantData = q.data; embQuantDims = q.dims;
  }

  return {
    emb_enc:      embEncData,
    embEncDims,
    codes:        codesData,
    codesType:    quantOut.codes.type,
    codesDims,
    emb_quant:    embQuantData,
    embQuantDims,
    scale:        encOut.scale ? new Float32Array(encOut.scale.data) : null,
    scaleDims:    encOut.scale ? encOut.scale.dims.slice() : null,
    hEncNew: new Float32Array(encOut.h_out.data),
    cEncNew: new Float32Array(encOut.c_out.data),
  };
}

// Run stage 4 (decode_audio) from a continuous embedding tensor.
// stereoOut (48k only): return the decoder's planar stereo output [2*T]
// (L plane, R plane) instead of averaging the channels to mono.
async function decodeFromLatents(emb, embDims, scale, scaleDims, modelHz, hDec, cDec, stereoOut = false) {
  const inputs = {
    emb:  new ort.Tensor('float32', emb, embDims),
    h_in: stateTensor(hDec, 1),
    c_in: stateTensor(cDec, 1),
  };
  if (modelHz === '48k' && scale) {
    inputs.scale = new ort.Tensor('float32', scale, scaleDims);
  }
  const decOut = await sessions.decAudio.run(inputs);

  let outAudio;
  if (modelHz === '48k') {
    const raw = decOut.audio.data;
    if (stereoOut) {
      outAudio = new Float32Array(raw); // already planar [2*T]
    } else {
      const tOut = raw.length >> 1;
      outAudio   = new Float32Array(tOut);
      for (let i = 0; i < tOut; i++) outAudio[i] = (raw[i] + raw[tOut + i]) * 0.5;
    }
  } else {
    outAudio = new Float32Array(decOut.audio.data);
  }
  return {
    outAudio,
    hDecNew: new Float32Array(decOut.h_out.data),
    cDecNew: new Float32Array(decOut.c_out.data),
  };
}

// Apply interpolation at the chosen point then decode to audio.
// scaleAlpha independently controls the 48k RMS scale mix (A=0, B=1).
async function interpDecodeOnce(capA, capB, alpha, interpPoint, modelHz, hDec, cDec,
                                vqMode = 'flat_swap', levelAlphas = null, scaleAlpha = 0.5,
                                deterministic = true) {
  if (interpPoint === 'encoder_latents') {
    const n         = Math.min(capA.emb_enc.length, capB.emb_enc.length);
    const interpEmb = lerpF32(capA.emb_enc.subarray(0, n), capB.emb_enc.subarray(0, n), alpha);
    const [B, C]    = capA.embEncDims;
    const dims      = [B, C, n / (B * C)];
    const quantOut    = await sessions.quantEnc.run({
      emb: new ort.Tensor('float32', interpEmb, dims),
    });
    const decCodesOut = await sessions.decCodes.run({ codes: quantOut.codes });
    const qEmb  = new Float32Array(decCodesOut.emb.data);
    const qDims = decCodesOut.emb.dims.slice();
    const scale = (capA.scale && capB.scale) ? lerpF32(capA.scale, capB.scale, scaleAlpha) : null;
    return decodeFromLatents(qEmb, qDims, scale, capA.scaleDims, modelHz, hDec, cDec);
  }

  if (interpPoint === 'vq_codes') {
    let mixedCodes;
    switch (vqMode) {
      case 'frame_swap':
        mixedCodes = mixCodesByFrame(capA.codes, capB.codes, capA.codesDims, alpha, deterministic);
        break;
      case 'level_swap':
        mixedCodes = mixCodesByLevel(capA.codes, capB.codes, capA.codesDims, levelAlphas || [], deterministic);
        break;
      case 'int_lerp':
        mixedCodes = mixCodesIntLerpByLevel(capA.codes, capB.codes, capA.codesDims,
                       levelAlphas || new Array(capA.codesDims[1]).fill(alpha));
        break;
      case 'sim_interp':
        await ensureCodebooks(modelHz);
        mixedCodes = mixCodesSimInterp(capA.codes, capB.codes, capA.codesDims,
                       levelAlphas || new Array(capA.codesDims[1]).fill(alpha),
                       codebookCache[modelHz]);
        break;
      default: // flat_swap
        mixedCodes = mixCodesElementWise(capA.codes, capB.codes, alpha, deterministic);
    }
    const codesTensor = new ort.Tensor(capA.codesType, mixedCodes, capA.codesDims);
    const decCodesOut = await sessions.decCodes.run({ codes: codesTensor });
    const emb   = new Float32Array(decCodesOut.emb.data);
    const dims  = decCodesOut.emb.dims.slice();
    const scale = (capA.scale && capB.scale) ? lerpF32(capA.scale, capB.scale, scaleAlpha) : null;
    return decodeFromLatents(emb, dims, scale, capA.scaleDims, modelHz, hDec, cDec);
  }

  // quantized_embeddings
  const n         = Math.min(capA.emb_quant.length, capB.emb_quant.length);
  const interpEmb = lerpF32(capA.emb_quant.subarray(0, n), capB.emb_quant.subarray(0, n), alpha);
  const [B, C]    = capA.embQuantDims;
  const dims      = [B, C, n / (B * C)];
  const scale     = (capA.scale && capB.scale) ? lerpF32(capA.scale, capB.scale, scaleAlpha) : null;
  return decodeFromLatents(interpEmb, dims, scale, capA.scaleDims, modelHz, hDec, cDec);
}

// ---------------------------------------------------------------------------
// Capture cache  (experiment2 only)
// ---------------------------------------------------------------------------
//
// Structure stored per source:
//   Non-streaming 24k:
//     { mode: 'single', cap: <encodeCapture result>, totalLen }
//   Streaming 24k OLA:
//     { mode: 'ola24', stride, seg, win, totalLen,
//       chunks: [ { cap, srcS, srcE, off }, … ] }
//   48k (always stateless OLA):
//     { mode: 'ola48', totalLen,
//       segments: [ { cap, offset, validEnd }, … ] }
//
// Interpolation reads these without re-encoding.
// ---------------------------------------------------------------------------

const captureCache = { A: null, B: null };

// Encode one source, store captures, and return the decoded audio for display.
async function runEncode(msg) {
  const { jobId, source, audio, modelHz, bwKbps, streaming, frameSize, channels = 1,
          skipPreview = false, statefulOLA = true } = msg;

  try {
    self.postMessage({ type: 'progress', jobId, value: 0, status: 'Loading models…' });
    await ensureSessions(modelHz, bwKbps);
    if (currentJobId !== jobId) return;

    // skipPreview: callers that only need the captures (e.g. experiment 6, which
    // discards the decoded preview) skip the per-segment decode_audio pass —
    // roughly halves 48k encode time at zero cost since the output is unused.
    let decoded, captures;
    if (modelHz === '48k') {
      [decoded, captures] = await encodeAndCapture48k(audio, jobId, channels, skipPreview, statefulOLA);
    } else {
      [decoded, captures] = await encodeAndCapture24k(audio, streaming, frameSize, jobId, skipPreview);
    }

    if (decoded === null) return;   // cancelled
    captureCache[source] = captures;
    self.postMessage({ type: 'result', jobId, decoded }, [decoded.buffer]);

  } catch (err) {
    self.postMessage({ type: 'error', jobId, message: err.message });
  }
}

// Encode full audio, capture intermediate tensors, decode for display.
// Non-streaming: single pass.  Streaming: OLA with encoder state carried.
async function encodeAndCapture24k(audio, streaming, frameSize, jobId, skipPreview = false) {
  const totalLen = audio.length;

  if (!streaming) {
    self.postMessage({ type: 'progress', jobId, value: 0.3, status: 'Encoding…' });
    const cap = await encodeCapture(audio.slice(), '24k', zeroState(1), zeroState(1));
    if (currentJobId !== jobId) return [null, null];

    if (skipPreview) {
      self.postMessage({ type: 'progress', jobId, value: 1.0, status: 'Done.' });
      return [new Float32Array(0), { mode: 'single', cap, totalLen }];
    }

    self.postMessage({ type: 'progress', jobId, value: 0.7, status: 'Decoding…' });
    const dec = await decodeFromLatents(
      cap.emb_quant, cap.embQuantDims, cap.scale, cap.scaleDims,
      '24k', zeroState(1), zeroState(1));
    if (currentJobId !== jobId) return [null, null];

    self.postMessage({ type: 'progress', jobId, value: 1.0, status: 'Done.' });
    return [
      dec.outAudio.slice(0, totalLen),
      { mode: 'single', cap, totalLen },
    ];
  }

  // Streaming OLA — carry encoder state, thread decoder state for display quality
  const stride  = frameSize;
  const seg     = stride + 2 * HALF_OV_24;
  const win     = new Float32Array(seg);
  for (let i = 0; i < seg; i++) {
    const t = (i + 1) / (seg + 1);
    win[i] = 0.5 - Math.abs(t - 0.5);
  }

  const outBuf  = skipPreview ? null : new Float32Array(totalLen);
  const wgtBuf  = skipPreview ? null : new Float32Array(totalLen);
  const numSegs = Math.ceil(totalLen / stride);
  const chunks  = [];

  let hEnc = zeroState(1), cEnc = zeroState(1);
  let hDec = zeroState(1), cDec = zeroState(1);

  for (let s = 0; s < numSegs; s++) {
    if (currentJobId !== jobId) return [null, null];

    const off  = s * stride - HALF_OV_24;
    const srcS = Math.max(off, 0);
    const srcE = Math.min(off + seg, totalLen);
    const chunk = new Float32Array(seg);
    if (srcE > srcS) chunk.set(audio.subarray(srcS, srcE), srcS - off);

    const cap = await encodeCapture(chunk, '24k', hEnc, cEnc);
    hEnc = cap.hEncNew; cEnc = cap.cEncNew;

    chunks.push({ cap, srcS, srcE, off });

    if (!skipPreview) {
      if (currentJobId !== jobId) return [null, null];
      const dec = await decodeFromLatents(
        cap.emb_quant, cap.embQuantDims, cap.scale, cap.scaleDims,
        '24k', hDec, cDec);
      hDec = dec.hDecNew; cDec = dec.cDecNew;
      for (let i = srcS - off; i < srcE - off; i++) {
        const pos = off + i;
        outBuf[pos] += win[i] * dec.outAudio[i];
        wgtBuf[pos] += win[i];
      }
    }

    self.postMessage({
      type: 'progress', jobId,
      value: Math.min((s + 1) / numSegs, 1),
      status: `Processing… ${Math.round((s + 1) / numSegs * 100)} %`,
    });
  }

  if (skipPreview) return [new Float32Array(0), { mode: 'ola24', stride, seg, win, totalLen, chunks }];
  for (let i = 0; i < totalLen; i++) {
    if (wgtBuf[i] > 0) outBuf[i] /= wgtBuf[i];
  }
  return [outBuf, { mode: 'ola24', stride, seg, win, totalLen, chunks }];
}

// 48k uses stateful encoder and decoder so that LSTM context is continuous across
// segment boundaries, eliminating embedding jumps at every STRIDE_48 boundary.
// channels = 2: audio is planar stereo [2*totalLen] (L plane, R plane) and each
// segment chunk is sliced per plane so the encoder sees real L/R. The decoded
// preview (outBuf) stays mono either way — it's display-only.
// statefulOLA = false: reset LSTM state per segment (encoder/decoder both start
// from zero each chunk) — segments become independent (parallelisable), at the
// cost of the cross-boundary continuity. Guard frames still warm the conv stack.
async function encodeAndCapture48k(audio, jobId, channels = 1, skipPreview = false, statefulOLA = true) {
  const totalLen = audio.length / channels;
  const outBuf   = skipPreview ? null : new Float32Array(totalLen);
  const wgtBuf   = skipPreview ? null : new Float32Array(totalLen);
  const numSegs  = Math.ceil(totalLen / STRIDE_48);
  const segments = [];

  let hEnc = zeroState(1), cEnc = zeroState(1);
  let hDec = zeroState(1), cDec = zeroState(1);

  for (let s = 0; s < numSegs; s++) {
    if (currentJobId !== jobId) return [null, null];

    const offset          = s * STRIDE_48;
    const leftGuard       = Math.min(offset, GUARD_48);
    const leftGuardFrames = leftGuard / HOP_48;

    const extLen = leftGuard + SEG_48;
    const chunk  = new Float32Array(channels * extLen);
    for (let ch = 0; ch < channels; ch++) {
      const srcOff = ch * totalLen + offset - leftGuard;
      const n      = Math.min(extLen, totalLen - (offset - leftGuard));
      chunk.set(audio.subarray(srcOff, srcOff + n), ch * extLen);
    }

    // Stateless mode keeps hEnc/cEnc at their initial zero state every chunk.
    const cap = await encodeCapture(chunk, '48k', hEnc, cEnc, leftGuardFrames, channels);
    if (statefulOLA) { hEnc = cap.hEncNew; cEnc = cap.cEncNew; }

    const validEnd = Math.min(offset + SEG_48, totalLen);
    segments.push({ cap, offset, validEnd });

    // Preview decode — display-only, skipped when the caller discards it.
    // Carries decoder LSTM state, so it's a second sequential chain we avoid entirely.
    if (!skipPreview) {
      if (currentJobId !== jobId) return [null, null];
      const dec = await decodeFromLatents(
        cap.emb_quant, cap.embQuantDims, cap.scale, cap.scaleDims,
        '48k', hDec, cDec);
      if (statefulOLA) { hDec = dec.hDecNew; cDec = dec.cDecNew; }
      for (let i = 0; i < validEnd - offset; i++) {
        outBuf[offset + i] += TRI_WIN_48[i] * dec.outAudio[i];
        wgtBuf[offset + i] += TRI_WIN_48[i];
      }
    }

    self.postMessage({
      type: 'progress', jobId,
      value: Math.min((s + 1) / numSegs, 1),
      status: `Processing… ${Math.round((s + 1) / numSegs * 100)} %`,
    });
  }

  if (skipPreview) return [new Float32Array(0), { mode: 'ola48', totalLen, segments }];
  for (let i = 0; i < totalLen; i++) {
    if (wgtBuf[i] > 0) outBuf[i] /= wgtBuf[i];
  }
  return [outBuf, { mode: 'ola48', totalLen, segments }];
}

// ---------------------------------------------------------------------------
// Interpolation using cached captures  (no re-encoding)
// ---------------------------------------------------------------------------

async function runInterpolation(msg) {
  const { jobId, modelHz, bwKbps, streaming, frameSize,
          alpha, interpPoint,
          vqMode = 'flat_swap', levelAlphas = null,
          scaleAlpha = 0.5, deterministic = true } = msg;

  try {
    self.postMessage({ type: 'progress', jobId, value: 0, status: 'Loading models…' });
    await ensureSessions(modelHz, bwKbps);
    if (currentJobId !== jobId) return;

    const capsA = captureCache.A;
    const capsB = captureCache.B;
    if (!capsA || !capsB) {
      self.postMessage({ type: 'error', jobId, message: 'Sources not yet encoded.' });
      return;
    }

    const decoded = (capsA.mode === 'ola48')
      ? await interpFromCached48k(capsA, capsB, alpha, interpPoint, vqMode, levelAlphas, scaleAlpha, deterministic, jobId)
      : await interpFromCached24k(capsA, capsB, alpha, interpPoint, vqMode, levelAlphas, scaleAlpha, deterministic, jobId);

    if (decoded === null) return;
    self.postMessage({ type: 'result', jobId, decoded }, [decoded.buffer]);

  } catch (err) {
    self.postMessage({ type: 'error', jobId, message: err.message });
  }
}

async function interpFromCached24k(capsA, capsB, alpha, interpPoint, vqMode, levelAlphas, scaleAlpha, deterministic, jobId) {
  if (capsA.mode === 'single') {
    // Non-streaming: single interpDecodeOnce, zero decoder state
    self.postMessage({ type: 'progress', jobId, value: 0.5, status: 'Interpolating…' });
    const result = await interpDecodeOnce(
      capsA.cap, capsB.cap, alpha, interpPoint, '24k',
      zeroState(1), zeroState(1), vqMode, levelAlphas, scaleAlpha, deterministic);
    if (currentJobId !== jobId) return null;
    self.postMessage({ type: 'progress', jobId, value: 1.0, status: 'Done.' });
    return result.outAudio.slice(0, capsA.totalLen);
  }

  // Streaming OLA: iterate cached chunks, thread decoder state
  const { win, totalLen, chunks: chunksA } = capsA;
  const { chunks: chunksB } = capsB;
  const outBuf  = new Float32Array(totalLen);
  const wgtBuf  = new Float32Array(totalLen);
  const numSegs = chunksA.length;

  let hDec = zeroState(1), cDec = zeroState(1);

  for (let s = 0; s < numSegs; s++) {
    if (currentJobId !== jobId) return null;

    const { cap: capA, srcS, srcE, off } = chunksA[s];
    const { cap: capB }                  = chunksB[s];

    const dec = await interpDecodeOnce(
      capA, capB, alpha, interpPoint, '24k', hDec, cDec, vqMode, levelAlphas, scaleAlpha, deterministic);
    hDec = dec.hDecNew; cDec = dec.cDecNew;

    for (let i = srcS - off; i < srcE - off; i++) {
      const pos = off + i;
      outBuf[pos] += win[i] * dec.outAudio[i];
      wgtBuf[pos] += win[i];
    }

    self.postMessage({
      type: 'progress', jobId,
      value: Math.min((s + 1) / numSegs, 1),
      status: `Interpolating… ${Math.round((s + 1) / numSegs * 100)} %`,
    });
  }

  for (let i = 0; i < totalLen; i++) {
    if (wgtBuf[i] > 0) outBuf[i] /= wgtBuf[i];
  }
  return outBuf;
}

async function interpFromCached48k(capsA, capsB, alpha, interpPoint, vqMode, levelAlphas, scaleAlpha, deterministic, jobId) {
  const { totalLen, segments: segsA } = capsA;
  const { segments: segsB }           = capsB;
  const outBuf = new Float32Array(totalLen);
  const wgtBuf = new Float32Array(totalLen);

  for (let s = 0; s < segsA.length; s++) {
    if (currentJobId !== jobId) return null;

    const { cap: capA, offset, validEnd } = segsA[s];
    const { cap: capB }                   = segsB[s];

    const dec = await interpDecodeOnce(
      capA, capB, alpha, interpPoint, '48k',
      zeroState(1), zeroState(1), vqMode, levelAlphas, scaleAlpha, deterministic);

    for (let i = 0; i < validEnd - offset; i++) {
      outBuf[offset + i] += TRI_WIN_48[i] * dec.outAudio[i];
      wgtBuf[offset + i] += TRI_WIN_48[i];
    }

    self.postMessage({
      type: 'progress', jobId,
      value: Math.min((s + 1) / segsA.length, 1),
      status: `Interpolating… ${Math.round((s + 1) / segsA.length * 100)} %`,
    });
  }

  for (let i = 0; i < totalLen; i++) {
    if (wgtBuf[i] > 0) outBuf[i] /= wgtBuf[i];
  }
  return outBuf;
}

// ---------------------------------------------------------------------------
// Delta transfer  (experiment3)
// ---------------------------------------------------------------------------

// Apply dynamics transfer.
// accumState: Float32Array(C) — per-channel accumulator for cumulative mode.
// startFrame: first A-frame index where modification begins.
// anchorBFrame: Float32Array(C) — B embedding values at the onset frame,
//               used as the reference point for 'anchored' mode.
// anchorAFrame, anchorBFrame: Float32Array(C), pre-populated by caller for 'anchored' mode.
//   anchorAFrame[c] = A's embedding at the global onset frame.
//   anchorBFrame[c] = B's embedding at the global onset frame.
// filtDelta: optional Float32Array(C × TB) of pre-computed SVD-projected deltas.
// When provided, replaces per-frame B delta computation in all modes.
// For anchored mode with filtDelta, uses cumulative integration from onset rather than
// absolute B value difference (since filtDelta is already expressed as deltas, not absolutes).
function applyDeltaTransfer(embA, embB, C, TA, TB, mode, strength, startFrame, accumState, anchorAFrame, anchorBFrame, dimMask, filtDelta = null) {
  const out = new Float32Array(C * TA);
  for (let c = 0; c < C; c++) {
    // Inactive channels always pass A through unchanged (all modes).
    if (dimMask && !dimMask[c]) {
      for (let t = 0; t < TA; t++)
        out[c * TA + t] = embA[c * TA + t];
      continue;
    }
    for (let t = 0; t < TA; t++) {
      if (t < startFrame) {
        out[c * TA + t] = embA[c * TA + t];
        continue;
      }
      const tRelB = (t - startFrame) % TB;
      const delta = filtDelta ? filtDelta[c * TB + tRelB]
                  : (tRelB === 0 ? 0 : embB[c * TB + tRelB] - embB[c * TB + (tRelB - 1)]);
      if (mode === 'per_frame') {
        out[c * TA + t] = embA[c * TA + t] + strength * delta;
      } else if (mode === 'cumulative') {
        accumState[c] += delta;
        out[c * TA + t] = embA[c * TA + t] + strength * accumState[c];
      } else {
        // anchored: freeze A at its onset embedding, then follow B's trajectory from
        // B's onset value.
        // out[t] = A[onset] + strength × (B[t] − B[onset])
        // A's motion is completely replaced by B's dynamics starting from onset.
        // When B is silent (B[t] ≈ B[onset]) the output is a static A-onset embedding —
        // this is expected behaviour; use per_frame if you want A to continue when B is quiet.
        if (filtDelta) {
          accumState[c] += delta;
          out[c * TA + t] = anchorAFrame[c] + strength * accumState[c];
        } else {
          out[c * TA + t] = anchorAFrame[c] + strength * (embB[c * TB + tRelB] - anchorBFrame[c]);
        }
      }
    }
  }
  return out;
}

async function decodeWithDelta(embA, dimsA, embB, dimsB,
                               mode, strength, startFrame, accumState, anchorAFrame, anchorBFrame,
                               dimMask, filtDelta,
                               applyPoint, modelHz, scale, scaleDims, hDec, cDec) {
  const [, C, TA] = dimsA;
  const [, , TB]  = dimsB;
  const outEmb = applyDeltaTransfer(embA, embB, C, TA, TB, mode, strength, startFrame, accumState, anchorAFrame, anchorBFrame, dimMask, filtDelta);

  let finalEmb = outEmb, finalDims = dimsA.slice();
  if (applyPoint === 'encoder_latents') {
    const quantOut    = await sessions.quantEnc.run({ emb: new ort.Tensor('float32', outEmb, dimsA) });
    const decCodesOut = await sessions.decCodes.run({ codes: quantOut.codes });
    finalEmb  = new Float32Array(decCodesOut.emb.data);
    finalDims = decCodesOut.emb.dims.slice();
  }
  return decodeFromLatents(finalEmb, finalDims, scale, scaleDims, modelHz, hDec, cDec);
}

function getCapEmb(cap, applyPoint) {
  return applyPoint === 'encoder_latents'
    ? { emb: cap.emb_enc,   dims: cap.embEncDims }
    : { emb: cap.emb_quant, dims: cap.embQuantDims };
}

async function runDeltaTransfer(msg) {
  const { jobId, modelHz, bwKbps, streaming, frameSize,
          applyPoint, mode, strength, startFrac,
          dimMask = null, svdMode = false, nComponents = 8, svdCompMask = null } = msg;

  try {
    self.postMessage({ type: 'progress', jobId, value: 0, status: 'Loading models…' });
    await ensureSessions(modelHz, bwKbps);
    if (currentJobId !== jobId) return;

    const capsA = captureCache.A, capsB = captureCache.B;
    if (!capsA || !capsB) {
      self.postMessage({ type: 'error', jobId, message: 'Sources not yet encoded.' });
      return;
    }

    const svd = capsB.svd;
    let filtDeltaFull = null;
    if (svdMode && svd) {
      // Convert component mask to sorted index list
      const activeComps = [];
      if (svdCompMask) {
        for (let k = 0; k < svdCompMask.length; k++)
          if (svdCompMask[k]) activeComps.push(k);
      } else {
        for (let k = 0; k < Math.min(nComponents, svd.C); k++) activeComps.push(k);
      }
      // Always set filtDeltaFull in SVD mode so applyDeltaTransfer never falls back
      // to raw B deltas. Zero array = no B influence when nothing is selected.
      filtDeltaFull = activeComps.length > 0
        ? computeFiltDeltaFull(svd.U, svd.deltas, svd.C, svd.totalT, activeComps, svd.meanD, svd.stdD)
        : new Float32Array(svd.C * svd.totalT);
    }
    const svdTotalT = svd ? svd.totalT : 0;
    // SVD mode handles all channels via projection — per-channel mask is irrelevant
    const activeMask = svdMode ? null : dimMask;

    const decoded = capsA.mode === 'ola48'
      ? await deltaTransfer48k(capsA, capsB, applyPoint, mode, strength, startFrac, activeMask, filtDeltaFull, svdTotalT, jobId)
      : await deltaTransfer24k(capsA, capsB, applyPoint, mode, strength, startFrac, activeMask, filtDeltaFull, svdTotalT, jobId);

    if (decoded === null) return;
    self.postMessage({ type: 'result', jobId, decoded }, [decoded.buffer]);

  } catch (err) {
    self.postMessage({ type: 'error', jobId, message: err.message });
  }
}

// Both 24k (75 fr/s) and 48k (150 fr/s) encode one frame per 320 audio samples.
const SAMPLES_PER_EMB_FRAME = 320;

async function deltaTransfer24k(capsA, capsB, applyPoint, mode, strength, startFrac, dimMask, filtDeltaFull, svdTotalT, jobId) {
  if (capsA.mode === 'single') {
    const { emb: embA, dims: dimsA } = getCapEmb(capsA.cap, applyPoint);
    const { emb: embB, dims: dimsB } = getCapEmb(capsB.cap, applyPoint);
    const [, C, TA] = dimsA;
    const [, , TB] = dimsB;
    const startFrame = Math.round(startFrac * TA);
    const accum = new Float32Array(C);
    const anchorAFrame = new Float32Array(C);
    const anchorBFrame = new Float32Array(C);
    if (mode === 'anchored') {
      const sf = Math.min(startFrame, TA - 1);
      for (let c = 0; c < C; c++) {
        anchorAFrame[c] = embA[c * TA + sf];
        anchorBFrame[c] = embB[c * TB + Math.min(sf, TB - 1)];
      }
    }

    self.postMessage({ type: 'progress', jobId, value: 0.5, status: 'Applying delta…' });
    // In single mode filtDeltaFull spans the full B sequence (same as TB), use directly
    const dec = await decodeWithDelta(
      embA, dimsA, embB, dimsB, mode, strength, startFrame, accum, anchorAFrame, anchorBFrame,
      dimMask, filtDeltaFull,
      applyPoint, '24k', capsA.cap.scale, capsA.cap.scaleDims, zeroState(1), zeroState(1));
    if (currentJobId !== jobId) return null;

    self.postMessage({ type: 'progress', jobId, value: 1.0, status: 'Done.' });
    return dec.outAudio.slice(0, capsA.totalLen);
  }

  // Streaming OLA — carry decoder state, accumulator, and anchorBFrame across chunks.
  const { win, totalLen, chunks: chunksA } = capsA;
  const { chunks: chunksB } = capsB;
  const outBuf  = new Float32Array(totalLen);
  const wgtBuf  = new Float32Array(totalLen);
  const numSegs = chunksA.length;

  const C = getCapEmb(chunksA[0].cap, applyPoint).dims[1];
  const accum        = new Float32Array(C);
  const anchorAFrame = new Float32Array(C);
  const anchorBFrame = new Float32Array(C);
  const totalFramesA = Math.round(totalLen / SAMPLES_PER_EMB_FRAME);
  let globalFrameA = 0;
  let bGlobalOff   = 0;
  let anchorSet = false;
  let hDec = zeroState(1), cDec = zeroState(1);

  for (let s = 0; s < numSegs; s++) {
    if (currentJobId !== jobId) return null;

    const { cap: capA, srcS, srcE, off } = chunksA[s];
    const { cap: capB } = chunksB[s % chunksB.length];

    const { emb: embA, dims: dimsA } = getCapEmb(capA, applyPoint);
    const { emb: embB, dims: dimsB } = getCapEmb(capB, applyPoint);
    const [, , TA] = dimsA;
    const [, , TB] = dimsB;

    const startFrameGlobal  = Math.round(startFrac * totalFramesA);
    const chunkStartFrame   = Math.max(0, startFrameGlobal - globalFrameA);
    globalFrameA += TA;

    // Capture onset anchors once, on the chunk where onset falls.
    if (mode === 'anchored' && !anchorSet && chunkStartFrame < TA) {
      const sf = Math.min(chunkStartFrame, TA - 1);
      for (let c = 0; c < C; c++) {
        anchorAFrame[c] = embA[c * TA + sf];
        anchorBFrame[c] = embB[c * TB + Math.min(sf, TB - 1)];
      }
      anchorSet = true;
    }

    const filtDeltaChunk = filtDeltaFull
      ? sliceFiltDelta(filtDeltaFull, C, svdTotalT, bGlobalOff, TB)
      : null;
    bGlobalOff = svdTotalT > 0 ? (bGlobalOff + TB) % svdTotalT : 0;

    const dec = await decodeWithDelta(
      embA, dimsA, embB, dimsB, mode, strength, chunkStartFrame, accum, anchorAFrame, anchorBFrame,
      dimMask, filtDeltaChunk,
      applyPoint, '24k', capA.scale, capA.scaleDims, hDec, cDec);
    hDec = dec.hDecNew; cDec = dec.cDecNew;

    for (let i = srcS - off; i < srcE - off; i++) {
      const pos = off + i;
      outBuf[pos] += win[i] * dec.outAudio[i];
      wgtBuf[pos] += win[i];
    }

    self.postMessage({
      type: 'progress', jobId,
      value: Math.min((s + 1) / numSegs, 1),
      status: `Applying… ${Math.round((s + 1) / numSegs * 100)} %`,
    });
  }

  for (let i = 0; i < totalLen; i++) {
    if (wgtBuf[i] > 0) outBuf[i] /= wgtBuf[i];
  }
  return outBuf;
}

async function deltaTransfer48k(capsA, capsB, applyPoint, mode, strength, startFrac, dimMask, filtDeltaFull, svdTotalT, jobId) {
  const { totalLen, segments: segsA } = capsA;
  const { segments: segsB } = capsB;
  const outBuf = new Float32Array(totalLen);
  const wgtBuf = new Float32Array(totalLen);

  const C = getCapEmb(segsA[0].cap, applyPoint).dims[1];
  const accum        = new Float32Array(C);
  const anchorAFrame = new Float32Array(C);
  const anchorBFrame = new Float32Array(C);
  const totalFramesA = Math.round(totalLen / SAMPLES_PER_EMB_FRAME);
  let globalFrameA = 0;
  let bGlobalOff   = 0;
  let anchorSet = false;
  // Carry decoder LSTM state across segments (matches deltaTransfer24k behaviour).
  // Without this, each segment cold-starts, causing periodic transient artefacts
  // at every segment boundary (every STRIDE_48 = 43200 samples ≈ 0.9 s).
  let hDec = zeroState(1), cDec = zeroState(1);

  for (let s = 0; s < segsA.length; s++) {
    if (currentJobId !== jobId) return null;

    const { cap: capA, offset, validEnd } = segsA[s];
    const { cap: capB } = segsB[s % segsB.length];

    const { emb: embA, dims: dimsA } = getCapEmb(capA, applyPoint);
    const { emb: embB, dims: dimsB } = getCapEmb(capB, applyPoint);
    const [, , TA] = dimsA;
    const [, , TB] = dimsB;

    const startFrameGlobal = Math.round(startFrac * totalFramesA);
    const chunkStartFrame  = Math.max(0, startFrameGlobal - globalFrameA);
    globalFrameA += TA;

    // Capture onset anchors once, on the chunk where onset falls.
    if (mode === 'anchored' && !anchorSet && chunkStartFrame < TA) {
      const sf = Math.min(chunkStartFrame, TA - 1);
      for (let c = 0; c < C; c++) {
        anchorAFrame[c] = embA[c * TA + sf];
        anchorBFrame[c] = embB[c * TB + Math.min(sf, TB - 1)];
      }
      anchorSet = true;
    }

    const filtDeltaChunk = filtDeltaFull
      ? sliceFiltDelta(filtDeltaFull, C, svdTotalT, bGlobalOff, TB)
      : null;
    bGlobalOff = svdTotalT > 0 ? (bGlobalOff + TB) % svdTotalT : 0;

    const dec = await decodeWithDelta(
      embA, dimsA, embB, dimsB, mode, strength, chunkStartFrame, accum, anchorAFrame, anchorBFrame,
      dimMask, filtDeltaChunk,
      applyPoint, '48k', capA.scale, capA.scaleDims, hDec, cDec);
    hDec = dec.hDecNew; cDec = dec.cDecNew;

    for (let i = 0; i < validEnd - offset; i++) {
      outBuf[offset + i] += TRI_WIN_48[i] * dec.outAudio[i];
      wgtBuf[offset + i] += TRI_WIN_48[i];
    }

    self.postMessage({
      type: 'progress', jobId,
      value: Math.min((s + 1) / segsA.length, 1),
      status: `Applying… ${Math.round((s + 1) / segsA.length * 100)} %`,
    });
  }

  for (let i = 0; i < totalLen; i++) {
    if (wgtBuf[i] > 0) outBuf[i] /= wgtBuf[i];
  }
  return outBuf;
}

// ---------------------------------------------------------------------------
// SVD helpers (experiment3)
// ---------------------------------------------------------------------------

// Jacobi cyclic eigendecomposition for a real symmetric n×n matrix.
// Returns { eigenvalues: Float32Array(n), eigenvectors: Float32Array(n×n) }
// sorted by descending eigenvalue. eigenvectors[:,k] is the k-th eigenvector.
function symmetricEigen(flatA, n, maxSweeps = 30, onSweep = null) {
  const A = flatA.slice();
  const V = new Float32Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1.0;

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let offNorm = 0;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++)
        offNorm += A[p*n+q] * A[p*n+q];
    if (offNorm < 1e-20) break;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const Apq = A[p*n+q];
        if (Math.abs(Apq) < 1e-12) continue;
        const App = A[p*n+p], Aqq = A[q*n+q];
        const tau = (Aqq - App) / (2 * Apq);
        const t   = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau*tau));
        const c   = 1 / Math.sqrt(1 + t*t), s = t * c;
        A[p*n+p] = App - t*Apq; A[q*n+q] = Aqq + t*Apq;
        A[p*n+q] = A[q*n+p] = 0;
        for (let r = 0; r < n; r++) {
          if (r === p || r === q) continue;
          const Arp = A[r*n+p], Arq = A[r*n+q];
          A[r*n+p] = A[p*n+r] = c*Arp - s*Arq;
          A[r*n+q] = A[q*n+r] = s*Arp + c*Arq;
        }
        for (let r = 0; r < n; r++) {
          const Vrp = V[r*n+p], Vrq = V[r*n+q];
          V[r*n+p] = c*Vrp - s*Vrq;
          V[r*n+q] = s*Vrp + c*Vrq;
        }
      }
    }
    if (onSweep) onSweep(sweep + 1, maxSweeps);
  }

  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[b*n+b] - A[a*n+a]);
  const eigenvalues  = new Float32Array(n);
  const eigenvectors = new Float32Array(n * n);
  for (let k = 0; k < n; k++) {
    eigenvalues[k] = Math.max(0, A[order[k]*n+order[k]]);
    for (let r = 0; r < n; r++) eigenvectors[r*n+k] = V[r*n+order[k]];
  }
  return { eigenvalues, eigenvectors };
}

// Project delta sequence onto selected eigenvectors and reconstruct.
// U: Float32Array(C×C) eigenvectors column-major (each column = one direction)
// deltas: Float32Array(C×totalT) — raw (un-normalised) deltas
// activeComps: number[] — sorted list of component indices to include
// meanD: Float32Array(C)|null — per-channel delta mean (subtract before projection if set)
// stdD:  Float32Array(C)|null — per-channel delta std  (divide before projection,
//                               multiply after reconstruction, if set)
// Returns filtered Float32Array(C×totalT) in the ORIGINAL (un-normalised) delta space.
// The mean is intentionally NOT added back — centering strips DC drift from the transfer.
function computeFiltDeltaFull(U, deltas, C, totalT, activeComps, meanD = null, stdD = null) {
  const filt = new Float32Array(C * totalT);
  for (const k of activeComps) {
    if (k < 0 || k >= C) continue;
    for (let t = 0; t < totalT; t++) {
      let coeff = 0;
      for (let c = 0; c < C; c++) {
        let d = deltas[c * totalT + t];
        if (meanD) d -= meanD[c];
        if (stdD)  d /= stdD[c];
        coeff += U[c * C + k] * d;
      }
      for (let c = 0; c < C; c++) {
        let contrib = U[c * C + k] * coeff;
        if (stdD) contrib *= stdD[c];   // rescale back to original units
        filt[c * totalT + t] += contrib;
      }
    }
  }
  return filt;
}

// Extract a TB-wide chunk of filtDeltaFull starting at bOff (wrapping at svdTotalT).
function sliceFiltDelta(filtDeltaFull, C, svdTotalT, bOff, TB) {
  const chunk = new Float32Array(C * TB);
  for (let c = 0; c < C; c++)
    for (let t = 0; t < TB; t++)
      chunk[c * TB + t] = filtDeltaFull[c * svdTotalT + (bOff + t) % svdTotalT];
  return chunk;
}

function runGetEmbDeltas(msg) {
  const { applyPoint = 'encoder_latents' } = msg;
  const capsB = captureCache.B;
  if (!capsB) return;

  // Collect all caps across modes
  let caps;
  if (capsB.mode === 'single')  caps = [capsB.cap];
  else if (capsB.mode === 'ola48') caps = capsB.segments.map(s => s.cap);
  else                             caps = capsB.chunks.map(c => c.cap);

  const C  = getCapEmb(caps[0], applyPoint).dims[1];
  const Ts = caps.map(cap => getCapEmb(cap, applyPoint).dims[2]);
  const totalT = Ts.reduce((a, b) => a + b, 0);

  // Concatenate embeddings along T
  const full = new Float32Array(C * totalT);
  let tOff = 0;
  for (let i = 0; i < caps.length; i++) {
    const { emb, dims } = getCapEmb(caps[i], applyPoint);
    const T = dims[2];
    for (let c = 0; c < C; c++)
      for (let t = 0; t < T; t++)
        full[c * totalT + tOff + t] = emb[c * T + t];
    tOff += T;
  }

  // Compute frame deltas: delta[c,0] = 0, delta[c,t] = full[c,t] - full[c,t-1]
  const deltas = new Float32Array(C * totalT);
  for (let c = 0; c < C; c++)
    for (let t = 1; t < totalT; t++)
      deltas[c * totalT + t] = full[c * totalT + t] - full[c * totalT + t - 1];

  // For OLA modes, each segment is encoded independently (stateless), so the
  // embedding value at t=T_segment_end and t=T_segment_start+1 are unrelated.
  // The resulting cross-segment delta would be a large artificial jump that
  // corrupts the covariance/SVD and shows up as periodic banding artefacts.
  // Zero them out so SVD components only capture within-segment dynamics.
  if (capsB.mode === 'ola48' || capsB.mode === 'ola24') {
    let tBoundary = 0;
    for (let i = 0; i < caps.length - 1; i++) {
      tBoundary += Ts[i];
      for (let c = 0; c < C; c++)
        deltas[c * totalT + tBoundary] = 0;
    }
  }

  // Compute per-channel variance of the raw embeddings.
  const variance = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    let sum = 0;
    for (let t = 0; t < totalT; t++) sum += full[c * totalT + t];
    const mean = sum / totalT;
    let sq = 0;
    for (let t = 0; t < totalT; t++) { const d = full[c * totalT + t] - mean; sq += d * d; }
    variance[c] = sq / totalT;
  }

  // Optional pre-processing before covariance / SVD.
  // centerDeltas  — subtract per-channel mean of deltas (removes DC drift, standard PCA centering).
  // normalizeDeltas — divide by per-channel std so all channels contribute equally.
  const { centerDeltas = false, normalizeDeltas = false } = msg;
  let meanD = null, stdD = null;
  if (centerDeltas || normalizeDeltas) {
    meanD = new Float32Array(C);
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let t = 0; t < totalT; t++) s += deltas[c * totalT + t];
      meanD[c] = s / totalT;
    }
    if (normalizeDeltas) {
      stdD = new Float32Array(C);
      for (let c = 0; c < C; c++) {
        let sq = 0;
        for (let t = 0; t < totalT; t++) {
          const d = deltas[c * totalT + t] - meanD[c];
          sq += d * d;
        }
        stdD[c] = Math.max(Math.sqrt(sq / totalT), 1e-8);
      }
    }
    if (!centerDeltas) meanD = null; // not needed unless centering
  }

  // Covariance of (possibly normalised) deltas: cov[i,j] = Σ_t d̃[i,t]·d̃[j,t] / totalT
  const cov = new Float32Array(C * C);
  for (let i = 0; i < C; i++) {
    for (let j = i; j < C; j++) {
      let s = 0;
      for (let t = 0; t < totalT; t++) {
        let di = deltas[i * totalT + t], dj = deltas[j * totalT + t];
        if (meanD) { di -= meanD[i]; dj -= meanD[j]; }
        if (stdD)  { di /= stdD[i];  dj /= stdD[j]; }
        s += di * dj;
      }
      cov[i*C+j] = cov[j*C+i] = s / totalT;
    }
  }
  const { eigenvalues, eigenvectors } = symmetricEigen(cov, C);
  const svdS = new Float32Array(C);
  for (let k = 0; k < C; k++) svdS[k] = Math.sqrt(eigenvalues[k]);

  // Cache for SVD-mode transfer (deltas NOT transferred so worker retains the reference).
  // meanD / stdD are stored so computeFiltDeltaFull can apply the same normalisation.
  captureCache.B.svd = { U: eigenvectors, deltas, C, totalT, meanD, stdD };

  self.postMessage(
    { type: 'emb_deltas', data: deltas, variance, C, T: totalT, svdS },
    [variance.buffer, svdS.buffer]
  );
}

// ---------------------------------------------------------------------------
// SVD4 — per-source SVD for experiment4
// ---------------------------------------------------------------------------

// Full SVD of a source's raw embeddings via covariance eigendecomposition.
// Returns { U[C×C], S[C], Vt[C×totalT], Ec[C×totalT], mean[C], C, totalT }.
function computeFullSVD(caps, applyPoint, totalT, onSweep = null) {
  const C  = getCapEmb(caps[0], applyPoint).dims[1];

  // Concatenate embeddings along T, up to totalT frames.
  const E = new Float32Array(C * totalT);
  let tOff = 0;
  for (let i = 0; i < caps.length && tOff < totalT; i++) {
    const { emb, dims } = getCapEmb(caps[i], applyPoint);
    const T = Math.min(dims[2], totalT - tOff);
    for (let c = 0; c < C; c++)
      for (let t = 0; t < T; t++)
        E[c * totalT + tOff + t] = emb[c * dims[2] + t];
    tOff += T;
  }

  // Second-moment matrix cov[i,j] = Σ_t E[i,t]*E[j,t] / totalT (no centering).
  // Component 1 will capture the dominant embedding direction including DC.
  const cov = new Float32Array(C * C);
  for (let i = 0; i < C; i++)
    for (let j = i; j < C; j++) {
      let s = 0;
      for (let t = 0; t < totalT; t++) s += E[i * totalT + t] * E[j * totalT + t];
      cov[i*C+j] = cov[j*C+i] = s / totalT;
    }
  // 8 sweeps is sufficient for approximating the principal directions.
  const { eigenvalues, eigenvectors } = symmetricEigen(cov, C, 8, onSweep);

  const S = new Float32Array(C);
  for (let k = 0; k < C; k++) S[k] = Math.sqrt(Math.max(0, eigenvalues[k]));

  // Vt[k,t] = (Σ_c U[c,k] * E[c,t]) / S[k]
  const Vt = new Float32Array(C * totalT);
  for (let k = 0; k < C; k++) {
    const sk = Math.max(S[k], 1e-10);
    for (let t = 0; t < totalT; t++) {
      let v = 0;
      for (let c = 0; c < C; c++) v += eigenvectors[c * C + k] * E[c * totalT + t];
      Vt[k * totalT + t] = v / sk;
    }
  }

  return { U: eigenvectors, S, Vt, E, C, totalT };
}

// Compute SVD for both sources and store in captureCache.{A,B}.svd4.
// Posts { type: 'svd4_ready', sA, sB, C, totalT } when done.
function runComputeSVD4(msg) {
  try {
    const { applyPoint = 'encoder_latents' } = msg;
    const capsA = captureCache.A, capsB = captureCache.B;
    if (!capsA || !capsB) { self.postMessage({ type: 'svd4_error', message: 'Encode both sources first.' }); return; }

    const getCapsArr = caps => {
      if (caps.mode === 'single') return [caps.cap];
      if (caps.mode === 'ola48') return caps.segments.map(s => s.cap);
      return caps.chunks.map(c => c.cap);
    };

    const cA = getCapsArr(capsA), cB = getCapsArr(capsB);
    const TA = cA.reduce((s, c) => s + getCapEmb(c, applyPoint).dims[2], 0);
    const TB = cB.reduce((s, c) => s + getCapEmb(c, applyPoint).dims[2], 0);
    const totalT = Math.min(TA, TB);

    self.postMessage({ type: 'svd4_progress', message: 'SVD A — sweep 0/8', value: 0 });
    const svdA = computeFullSVD(cA, applyPoint, totalT, (sweep, total) =>
      self.postMessage({ type: 'svd4_progress',
        message: `SVD A — sweep ${sweep}/${total}`,
        value: sweep / total * 0.5 }));

    self.postMessage({ type: 'svd4_progress', message: 'SVD B — sweep 0/8', value: 0.5 });
    const svdB = computeFullSVD(cB, applyPoint, totalT, (sweep, total) =>
      self.postMessage({ type: 'svd4_progress',
        message: `SVD B — sweep ${sweep}/${total}`,
        value: 0.5 + sweep / total * 0.5 }));

    self.postMessage({ type: 'svd4_progress', message: 'Building caches…' });

    const buildCache = (svd, caps, capsArr) => {
      const rawTs = capsArr.map(c => getCapEmb(c, applyPoint).dims[2]);
      const segTs = [];
      let rem = totalT;
      for (const t of rawTs) { if (rem <= 0) break; segTs.push(Math.min(t, rem)); rem -= t; }
      const n = segTs.length;

      let segsInfo = null, segScales = null;
      if (caps.mode === 'ola48') {
        segsInfo  = caps.segments.slice(0, n).map(s => ({ offset: s.offset, validEnd: s.validEnd }));
        segScales = caps.segments.slice(0, n).map(s => ({ scale: s.cap.scale, scaleDims: s.cap.scaleDims }));
      } else if (caps.mode === 'ola24') {
        segsInfo = caps.chunks.slice(0, n).map(c => ({ srcS: c.srcS, srcE: c.srcE, off: c.off }));
      }
      const totalLen = caps.mode === 'ola48' ? segsInfo[n - 1].validEnd : caps.totalLen;

      return { ...svd, segTs, mode: caps.mode, segsInfo, segScales, win: caps.win || null, totalLen };
    };

    captureCache.A.svd4 = buildCache(svdA, capsA, cA);
    captureCache.B.svd4 = buildCache(svdB, capsB, cB);

    const sA = svdA.S.slice(), sB = svdB.S.slice();
    self.postMessage({ type: 'svd4_ready', sA, sB, C: svdA.C, totalT }, [sA.buffer, sB.buffer]);
  } catch (err) {
    self.postMessage({ type: 'svd4_error', message: 'SVD failed: ' + err.message });
  }
}

// Decode a modified [C, totalT] embedding matrix back to audio using the
// recipient source's OLA layout and segment scales.
async function decodeSVD4(Enew, C, totalT, svd4, applyPoint, modelHz, jobId, overrideScales = null) {
  const { segTs, mode, segsInfo, win, totalLen } = svd4;
  const segScales = overrideScales ?? svd4.segScales;
  const is48k = modelHz === '48k';

  if (mode === 'single') {
    const dims = [1, C, totalT];
    let finalEmb = Enew, finalDims = dims;
    if (applyPoint === 'encoder_latents') {
      const qOut = await sessions.quantEnc.run({ emb: new ort.Tensor('float32', Enew, dims) });
      const dOut = await sessions.decCodes.run({ codes: qOut.codes });
      finalEmb = new Float32Array(dOut.emb.data); finalDims = dOut.emb.dims.slice();
    }
    const dec = await decodeFromLatents(finalEmb, finalDims, null, null, modelHz, zeroState(1), zeroState(1));
    return dec.outAudio.slice(0, totalLen);
  }

  const outBuf = new Float32Array(totalLen);
  const wgtBuf = new Float32Array(totalLen);
  let hDec = zeroState(1), cDec = zeroState(1);
  let tOff = 0;

  for (let s = 0; s < segTs.length; s++) {
    if (currentJobId !== jobId) return null;
    const T = segTs[s];

    const segEmb = new Float32Array(C * T);
    for (let c = 0; c < C; c++)
      for (let t = 0; t < T; t++)
        segEmb[c * T + t] = Enew[c * totalT + tOff + t];
    const dims = [1, C, T];

    let finalEmb = segEmb, finalDims = dims;
    if (applyPoint === 'encoder_latents') {
      const qOut = await sessions.quantEnc.run({ emb: new ort.Tensor('float32', segEmb, dims) });
      const dOut = await sessions.decCodes.run({ codes: qOut.codes });
      finalEmb = new Float32Array(dOut.emb.data); finalDims = dOut.emb.dims.slice();
    }

    const sc = segScales ? segScales[s] : null;
    const dec = await decodeFromLatents(finalEmb, finalDims, sc?.scale || null, sc?.scaleDims || null, modelHz, hDec, cDec);
    hDec = dec.hDecNew; cDec = dec.cDecNew;

    if (is48k) {
      const { offset, validEnd } = segsInfo[s];
      for (let i = 0; i < validEnd - offset; i++) {
        outBuf[offset + i] += TRI_WIN_48[i] * dec.outAudio[i];
        wgtBuf[offset + i] += TRI_WIN_48[i];
      }
    } else {
      const { srcS, srcE, off } = segsInfo[s];
      for (let i = srcS - off; i < srcE - off; i++) {
        outBuf[off + i] += win[i] * dec.outAudio[i];
        wgtBuf[off + i] += win[i];
      }
    }

    tOff += T;
    self.postMessage({ type: 'progress', jobId, value: 0.3 + 0.7 * (s + 1) / segTs.length,
                       status: `Decoding… ${Math.round((s + 1) / segTs.length * 100)} %` });
  }

  for (let i = 0; i < totalLen; i++) if (wgtBuf[i] > 0) outBuf[i] /= wgtBuf[i];
  return outBuf;
}

// Apply SVD4 component swap and decode.
async function runApplySVD4(msg) {
  const { jobId, modelHz, bwKbps, swapMode, direction, strength, compMask, applyPoint } = msg;
  try {
    await ensureSessions(modelHz, bwKbps);
    if (currentJobId !== jobId) return;

    const sv4A = captureCache.A?.svd4, sv4B = captureCache.B?.svd4;
    if (!sv4A || !sv4B) { self.postMessage({ type: 'error', jobId, message: 'Compute SVD first.' }); return; }

    // A is always recipient, B is always donor.
    const recip = sv4A;
    const donor = sv4B;
    const { U: Ur, S: Sr, Vt: Vtr, E: ER, C, totalT } = recip;
    const { U: Ud, S: Sd, Vt: Vtd, E: ED, segScales: scalesD } = donor;
    const { mode = 'usv', alphaU = 0, alphaS = 0, alphaV = 0, alphaUS = 0 } = msg;
    const isUSV = mode === 'us_v';

    // Detect pure-endpoint cases — use E directly to avoid SVD round-trip error.
    const allActive = compMask.every(v => v > 0);
    const pureA = allActive && (isUSV ? alphaUS === 0 && alphaV === 0
                                      : alphaU  === 0 && alphaS === 0 && alphaV === 0);
    const pureB = allActive && (isUSV ? alphaUS === 1 && alphaV === 1
                                      : alphaU  === 1 && alphaS === 1 && alphaV === 1);

    self.postMessage({ type: 'progress', jobId, value: 0.05, status: 'Building modified embedding…' });

    const Enew = new Float32Array(C * totalT);

    if (pureA) {
      Enew.set(ER);
    } else if (pureB) {
      Enew.set(ED);
    } else if (isUSV) {
      for (let t = 0; t < totalT; t++) {
        for (let c = 0; c < C; c++) {
          let val = 0;
          for (let k = 0; k < C; k++) {
            if (compMask[k]) {
              const usA  = Ur[c*C+k] * Sr[k];
              const usB  = Ud[c*C+k] * Sd[k];
              const usEff = usA + (usB - usA) * alphaUS;
              const vEff  = Vtr[k*totalT+t] + (Vtd[k*totalT+t] - Vtr[k*totalT+t]) * alphaV;
              val += usEff * vEff;
            }
          }
          Enew[c*totalT+t] = val;
        }
      }
    } else {
      for (let t = 0; t < totalT; t++) {
        for (let c = 0; c < C; c++) {
          let val = 0;
          for (let k = 0; k < C; k++) {
            if (compMask[k]) {
              const uEff = Ur[c*C+k] + (Ud[c*C+k] - Ur[c*C+k]) * alphaU;
              const sEff = Sr[k]     + (Sd[k]     - Sr[k])     * alphaS;
              const vEff = Vtr[k*totalT+t] + (Vtd[k*totalT+t] - Vtr[k*totalT+t]) * alphaV;
              val += uEff * sEff * vEff;
            }
          }
          Enew[c*totalT+t] = val;
        }
      }
    }

    // Blend per-segment scales proportional to how far the sliders lean toward B.
    const scaleAlpha = pureB ? 1 : (isUSV ? (alphaUS + alphaV) / 2
                                           : (alphaU + alphaS + alphaV) / 3);
    let effectiveScales = recip.segScales;
    if (scaleAlpha > 0 && recip.segScales && scalesD) {
      effectiveScales = recip.segScales.map((rsc, s) => {
        const dsc = scalesD[s];
        if (!rsc?.scale || !dsc?.scale) return rsc;
        const blended = new Float32Array(rsc.scale.length);
        for (let i = 0; i < blended.length; i++)
          blended[i] = rsc.scale[i] + (dsc.scale[i] - rsc.scale[i]) * scaleAlpha;
        return { scale: blended, scaleDims: rsc.scaleDims };
      });
    }

    self.postMessage({ type: 'progress', jobId, value: 0.3, status: 'Decoding…' });

    const decoded = await decodeSVD4(Enew, C, totalT, recip, applyPoint, modelHz, jobId, effectiveScales);
    if (decoded === null) return;
    self.postMessage({ type: 'result', jobId, decoded }, [decoded.buffer]);
  } catch (err) {
    self.postMessage({ type: 'error', jobId, message: err.message });
  }
}

// ---------------------------------------------------------------------------
// Joint SVD (Experiment 5)
// ---------------------------------------------------------------------------

// Compute a single SVD on the horizontally-concatenated embedding matrix [E_A | E_B].
// This yields a shared basis U and per-source coordinate matrices VtA, VtB.
function computeJointSVD(cA, cB, applyPoint, totalT, onSweep = null) {
  const C = getCapEmb(cA[0], applyPoint).dims[1];

  // Build E_A and E_B each of shape [C, totalT].
  function buildE(caps) {
    const E = new Float32Array(C * totalT);
    let tOff = 0;
    for (let i = 0; i < caps.length && tOff < totalT; i++) {
      const { emb, dims } = getCapEmb(caps[i], applyPoint);
      const T = Math.min(dims[2], totalT - tOff);
      for (let c = 0; c < C; c++)
        for (let t = 0; t < T; t++)
          E[c * totalT + tOff + t] = emb[c * dims[2] + t];
      tOff += T;
    }
    return E;
  }

  const EA = buildE(cA);
  const EB = buildE(cB);
  const T2 = totalT * 2; // joint time dimension

  // Second-moment matrix on E_joint = [EA | EB] (shape [C, T2]).
  // cov[i,j] = (Σ_t EA[i,t]*EA[j,t] + Σ_t EB[i,t]*EB[j,t]) / T2
  const cov = new Float32Array(C * C);
  for (let i = 0; i < C; i++) {
    for (let j = i; j < C; j++) {
      let s = 0;
      for (let t = 0; t < totalT; t++) s += EA[i * totalT + t] * EA[j * totalT + t];
      for (let t = 0; t < totalT; t++) s += EB[i * totalT + t] * EB[j * totalT + t];
      cov[i*C+j] = cov[j*C+i] = s / T2;
    }
  }

  const { eigenvalues, eigenvectors } = symmetricEigen(cov, C, 8, onSweep);
  const U = eigenvectors; // [C, C] — shared basis

  const S = new Float32Array(C);
  for (let k = 0; k < C; k++) S[k] = Math.sqrt(Math.max(0, eigenvalues[k]));

  // Project each source onto the shared basis: Vt[k,t] = (U[:,k] · E[:,t]) / S[k]
  function projectVt(E) {
    const Vt = new Float32Array(C * totalT);
    for (let k = 0; k < C; k++) {
      const sk = Math.max(S[k], 1e-10);
      for (let t = 0; t < totalT; t++) {
        let v = 0;
        for (let c = 0; c < C; c++) v += U[c * C + k] * E[c * totalT + t];
        Vt[k * totalT + t] = v / sk;
      }
    }
    return Vt;
  }

  const VtA = projectVt(EA);
  const VtB = projectVt(EB);

  return { U, S, VtA, VtB, EA, EB, C, totalT };
}

// Compute joint SVD and store result; posts svd5_ready when done.
function runComputeSVD5(msg) {
  try {
    const { applyPoint = 'encoder_latents' } = msg;
    const capsA = captureCache.A, capsB = captureCache.B;
    if (!capsA || !capsB) {
      self.postMessage({ type: 'svd5_error', message: 'Encode both sources first.' });
      return;
    }

    const getCapsArr = caps => {
      if (caps.mode === 'single') return [caps.cap];
      if (caps.mode === 'ola48') return caps.segments.map(s => s.cap);
      return caps.chunks.map(c => c.cap);
    };

    const cA = getCapsArr(capsA), cB = getCapsArr(capsB);
    const TA = cA.reduce((s, c) => s + getCapEmb(c, applyPoint).dims[2], 0);
    const TB = cB.reduce((s, c) => s + getCapEmb(c, applyPoint).dims[2], 0);
    const totalT = Math.min(TA, TB);

    self.postMessage({ type: 'svd5_progress', message: 'Joint SVD — sweep 0/8', value: 0 });
    const joint = computeJointSVD(cA, cB, applyPoint, totalT, (sweep, total) =>
      self.postMessage({ type: 'svd5_progress',
        message: `Joint SVD — sweep ${sweep}/${total}`,
        value: sweep / total }));

    self.postMessage({ type: 'svd5_progress', message: 'Building caches…' });

    // Segment layout taken from capsA (A is always the recipient for decoding).
    const rawTs = cA.map(c => getCapEmb(c, applyPoint).dims[2]);
    const segTs = [];
    let rem = totalT;
    for (const t of rawTs) { if (rem <= 0) break; segTs.push(Math.min(t, rem)); rem -= t; }
    const n = segTs.length;

    let segsInfo = null, segScales = null;
    if (capsA.mode === 'ola48') {
      segsInfo  = capsA.segments.slice(0, n).map(s => ({ offset: s.offset, validEnd: s.validEnd }));
      segScales = capsA.segments.slice(0, n).map(s => ({ scale: s.cap.scale, scaleDims: s.cap.scaleDims }));
    } else if (capsA.mode === 'ola24') {
      segsInfo = capsA.chunks.slice(0, n).map(c => ({ srcS: c.srcS, srcE: c.srcE, off: c.off }));
    }
    const totalLen = capsA.mode === 'ola48'
      ? segsInfo[n - 1].validEnd
      : capsA.totalLen;

    captureCache.svd5 = {
      ...joint,
      segTs, mode: capsA.mode, segsInfo, segScales,
      win: capsA.win || null, totalLen,
    };

    const sJoint = joint.S.slice();
    self.postMessage({ type: 'svd5_ready', sJoint, C: joint.C, totalT },
      [sJoint.buffer]);
  } catch (err) {
    self.postMessage({ type: 'svd5_error', message: 'Joint SVD failed: ' + err.message });
  }
}

// Apply interpolation in the shared V space and decode.
async function runApplySVD5(msg) {
  const { jobId, modelHz, bwKbps, alphaV = 0, compMask, applyPoint,
          sMode = 'none', sScale = 1.0, sTilt = 0.0, softWeights = null } = msg;
  try {
    await ensureSessions(modelHz, bwKbps);
    if (currentJobId !== jobId) return;

    const sv5 = captureCache.svd5;
    if (!sv5) { self.postMessage({ type: 'error', jobId, message: 'Compute joint SVD first.' }); return; }

    const { U, S, VtA, VtB, EA, EB, C, totalT } = sv5;

    // Compute effective S weight per component.
    const Seff = new Float32Array(C);
    for (let k = 0; k < C; k++) {
      let w;
      if (sMode === 'soft') {
        w = softWeights ? softWeights[k] : (compMask[k] ? 1 : 0);
      } else {
        w = compMask[k] ? 1 : 0;
        if (sMode === 'scale') w *= sScale;
        else if (sMode === 'tilt') w *= Math.exp(-sTilt * k / Math.max(C - 1, 1));
      }
      Seff[k] = S[k] * w;
    }

    const allActive = compMask.every(v => v > 0);
    const pureA = sMode === 'none' && allActive && alphaV === 0;
    const pureB = sMode === 'none' && allActive && alphaV === 1;

    self.postMessage({ type: 'progress', jobId, value: 0.05, status: 'Building modified embedding…' });

    const Enew = new Float32Array(C * totalT);

    if (pureA) {
      Enew.set(EA);
    } else if (pureB) {
      Enew.set(EB);
    } else {
      for (let t = 0; t < totalT; t++) {
        for (let c = 0; c < C; c++) {
          let val = 0;
          for (let k = 0; k < C; k++) {
            if (Seff[k] !== 0) {
              const vEff = VtA[k * totalT + t] + (VtB[k * totalT + t] - VtA[k * totalT + t]) * alphaV;
              val += U[c * C + k] * Seff[k] * vEff;
            }
          }
          Enew[c * totalT + t] = val;
        }
      }
    }

    // Blend per-segment scales proportional to alphaV.
    const scalesA = sv5.segScales;
    const scalesB = captureCache.B?.mode === 'ola48'
      ? captureCache.B.segments?.map(s => ({ scale: s.cap.scale, scaleDims: s.cap.scaleDims }))
      : null;
    let effectiveScales = scalesA;
    if (alphaV > 0 && scalesA && scalesB) {
      effectiveScales = scalesA.map((rsc, s) => {
        const dsc = scalesB[s];
        if (!rsc?.scale || !dsc?.scale) return rsc;
        const blended = new Float32Array(rsc.scale.length);
        for (let i = 0; i < blended.length; i++)
          blended[i] = rsc.scale[i] + (dsc.scale[i] - rsc.scale[i]) * alphaV;
        return { scale: blended, scaleDims: rsc.scaleDims };
      });
    }

    self.postMessage({ type: 'progress', jobId, value: 0.3, status: 'Decoding…' });

    const decoded = await decodeSVD4(Enew, C, totalT, sv5, applyPoint, modelHz, jobId, effectiveScales);
    if (decoded === null) return;
    self.postMessage({ type: 'result', jobId, decoded }, [decoded.buffer]);
  } catch (err) {
    self.postMessage({ type: 'error', jobId, message: err.message });
  }
}

// ---------------------------------------------------------------------------
// Experiment6 helpers — shared by get_embeddings_exp6 and runPlayFrameExp6
// ---------------------------------------------------------------------------

// Flatten all segment caps into one concatenated cap along T.
function flattenCaps(caps) {
  const toN = d => Number(d);
  let capList;
  if (caps.mode === 'single')     capList = [caps.cap];
  else if (caps.mode === 'ola24') capList = caps.chunks.map(c => c.cap);
  else                            capList = caps.segments.map(s => s.cap); // ola48

  if (capList.length === 1) return capList[0];

  const first   = capList[0];
  const D_enc   = toN(first.embEncDims[1]);
  const D_quant = toN(first.embQuantDims[1]);
  const K       = toN(first.codesDims[1]);
  const totalT  = capList.reduce((s, c) => s + toN(c.codesDims[2]), 0);

  const emb_enc   = new Float32Array(D_enc   * totalT);
  const emb_quant = new Float32Array(D_quant * totalT);
  const codes = first.codesType === 'int64'
    ? new BigInt64Array(K * totalT) : new Int32Array(K * totalT);

  let tOff = 0;
  for (const cap of capList) {
    const T_i = toN(cap.codesDims[2]);
    for (let d = 0; d < D_enc; d++)
      for (let t = 0; t < T_i; t++)
        emb_enc[d * totalT + tOff + t] = cap.emb_enc[d * T_i + t];
    for (let d = 0; d < D_quant; d++)
      for (let t = 0; t < T_i; t++)
        emb_quant[d * totalT + tOff + t] = cap.emb_quant[d * T_i + t];
    for (let k = 0; k < K; k++)
      for (let t = 0; t < T_i; t++)
        codes[k * totalT + tOff + t] = cap.codes[k * T_i + t];
    tOff += T_i;
  }
  return { emb_enc, embEncDims: [1, D_enc, totalT],
           emb_quant, embQuantDims: [1, D_quant, totalT],
           codes, codesType: first.codesType, codesDims: [1, K, totalT] };
}

// Return a Float32Array(T) of per-frame scale scalars for 48k sources,
// or null for 24k (where scale is not used).
function flattenScales(caps) {
  let capList;
  if (caps.mode === 'single')     capList = [caps.cap];
  else if (caps.mode === 'ola24') capList = caps.chunks.map(c => c.cap);
  else                            capList = caps.segments.map(s => s.cap);
  if (!capList[0].scale) return null; // 24k has no scale
  const totalT = capList.reduce((s, c) => s + Number(c.codesDims[2]), 0);
  const out = new Float32Array(totalT);
  let tOff = 0;
  for (const cap of capList) {
    const T_i = Number(cap.codesDims[2]);
    // cap.scale is [1,1,1] — treat the first element as the segment scalar
    out.fill(cap.scale[0], tOff, tOff + T_i);
    tOff += T_i;
  }
  return out;
}

// Return the per-segment cap that contains flattened frame index `frameIdx`.
// Used to retrieve the 48k scale for the relevant segment.
function getCapForFrame(caps, frameIdx) {
  if (caps.mode === 'single') return caps.cap;
  const chunks = caps.mode === 'ola24' ? caps.chunks : caps.segments;
  let cum = 0;
  for (const chunk of chunks) {
    const T_i = Number(chunk.cap.codesDims[2]);
    if (frameIdx < cum + T_i) return chunk.cap;
    cum += T_i;
  }
  return chunks[chunks.length - 1].cap;
}

// ---------------------------------------------------------------------------
// Experiment6: decode a context window around a clicked frame
// ---------------------------------------------------------------------------

async function runPlayFrameExp6(msg) {
  const { jobId, source, startFrame, endFrame, bestIdx, modelHz, bwKbps } = msg;
  try {
    await ensureSessions(modelHz, bwKbps);

    const caps = captureCache[source];
    if (!caps) throw new Error('No capture for source ' + source);

    const flat    = flattenCaps(caps);
    const K       = Number(flat.codesDims[1]);
    const T       = Number(flat.codesDims[2]);
    const nFrames = endFrame - startFrame;

    // Extract window and convert to BigInt64 (decode_codes expects int64)
    const codes64 = new BigInt64Array(K * nFrames);
    for (let k = 0; k < K; k++)
      for (let t = 0; t < nFrames; t++) {
        const v = flat.codes[k * T + startFrame + t];
        codes64[k * nFrames + t] = typeof v === 'bigint' ? v : BigInt(v);
      }

    const codesTensor = new ort.Tensor('int64', codes64, [1, K, nFrames]);
    const decOut  = await sessions.decCodes.run({ codes: codesTensor });
    const emb     = new Float32Array(decOut.emb.data);
    const dims    = [...decOut.emb.dims].map(Number);

    // For 48k, retrieve scale from the segment containing the clicked frame.
    let scale = null, scaleDims = null;
    if (modelHz === '48k') {
      const segCap = getCapForFrame(caps, bestIdx);
      scale     = segCap.scale;
      scaleDims = segCap.scaleDims ? [...segCap.scaleDims].map(Number) : null;
    }

    const stereoOut = modelHz === '48k';
    const result  = await decodeFromLatents(emb, dims, scale, scaleDims, modelHz, zeroState(1), zeroState(1), stereoOut);
    const decoded = result.outAudio;
    self.postMessage({ type: 'frame_audio_exp6', jobId, decoded, channels: stereoOut ? 2 : 1 }, [decoded.buffer]);
  } catch (err) {
    self.postMessage({ type: 'error', jobId, message: 'Frame decode: ' + err.message });
  }
}

// ---------------------------------------------------------------------------
// Experiment6: decode a blended trajectory embedding sequence
// ---------------------------------------------------------------------------

async function runDecodeTrajectoryExp6(msg) {
  const { jobId, embQuant, dims, frameScales, modelHz, bwKbps } = msg;
  try {
    await ensureSessions(modelHz, bwKbps);

    const [, D, N] = dims;
    let decoded;

    if (modelHz === '48k') {
      // 48k decoder must be called in ≤150-frame chunks (matches OLA path).
      // Loudness: when per-frame scales are supplied, decode every chunk at
      // scale = 1 and apply a per-SAMPLE gain envelope (linear interp between
      // frame scales) after OLA — scale is a linear post-multiply in the
      // decoder graph, so this is exact, and it removes the loudness steps a
      // per-chunk mean produces when loud and quiet grains mix within a chunk.
      // Fall back to captureCache first-segment scale if frameScales wasn't supplied.
      const caps    = captureCache.A || captureCache.B;
      const refCap  = !caps ? null
                    : caps.mode === 'single' ? caps.cap
                    : caps.mode === 'ola24'  ? caps.chunks[0].cap
                    :                          caps.segments[0].cap;
      const fallbackScale     = refCap?.scale     ? refCap.scale.slice()            : new Float32Array([1.0]);
      const fallbackScaleDims = refCap?.scaleDims ? [...refCap.scaleDims].map(Number) : [1, 1];

      const CHUNK_F  = SEG_48 / HOP_48;    // 150 frames per chunk
      const STRIDE_F = STRIDE_48 / HOP_48; // 135 frames per stride (10% overlap)

      // Stereo OLA: planar [2*totalSamples] (L plane, R plane); the triangular
      // weights are channel-independent so wgtBuf stays mono.
      const totalSamples = N * HOP_48;
      const outBuf = new Float32Array(2 * totalSamples);
      const wgtBuf = new Float32Array(totalSamples);
      let hDec = zeroState(1), cDec = zeroState(1);

      const numChunks = Math.ceil(N / STRIDE_F);
      for (let seg = 0; seg < numChunks; seg++) {
        const tStart  = seg * STRIDE_F;
        const tEnd    = Math.min(tStart + CHUNK_F, N);
        const chunk_T = tEnd - tStart;

        const chunkEmb = new Float32Array(D * chunk_T);
        for (let d = 0; d < D; d++)
          for (let i = 0; i < chunk_T; i++)
            chunkEmb[d * chunk_T + i] = embQuant[d * N + tStart + i];

        // Always use fallbackScaleDims (real shape from the ONNX encoder output) —
        // never hardcode [1,1,1]; a shape mismatch silently hangs the 48k decoder.
        // With frameScales: unit scale here, per-sample envelope applied after OLA.
        const scale = frameScales
          ? new Float32Array(fallbackScale.length).fill(1.0)
          : fallbackScale;

        const res = await decodeFromLatents(chunkEmb, [1, D, chunk_T], scale, fallbackScaleDims, modelHz, hDec, cDec, true);
        hDec = res.hDecNew; cDec = res.cDecNew;

        const sampleOffset = tStart * HOP_48;
        const numSamples   = chunk_T * HOP_48;
        // res.outAudio is planar [2*numSamples]
        for (let i = 0; i < numSamples; i++) {
          const w = TRI_WIN_48[i];
          outBuf[sampleOffset + i]                += w * res.outAudio[i];
          outBuf[totalSamples + sampleOffset + i] += w * res.outAudio[numSamples + i];
          wgtBuf[sampleOffset + i]                += w;
        }
      }

      for (let i = 0; i < totalSamples; i++) {
        if (wgtBuf[i] > 0) {
          outBuf[i]                /= wgtBuf[i];
          outBuf[totalSamples + i] /= wgtBuf[i];
        }
      }

      // Per-sample gain envelope from the per-frame scales (linear interp),
      // applied identically to both planes.
      if (frameScales) {
        for (let s = 0; s < totalSamples; s++) {
          const t  = s / HOP_48;
          const t0 = Math.min(N - 1, Math.floor(t));
          const t1 = Math.min(N - 1, t0 + 1);
          const fr = t - t0;
          const g  = frameScales[t0] * (1 - fr) + frameScales[t1] * fr;
          outBuf[s]                *= g;
          outBuf[totalSamples + s] *= g;
        }
      }
      decoded = outBuf;
    } else {
      const res = await decodeFromLatents(embQuant, dims, null, null, modelHz, zeroState(1), zeroState(1));
      decoded = new Float32Array(res.outAudio);
    }

    const outChannels = modelHz === '48k' ? 2 : 1;
    self.postMessage({ type: 'trajectory_audio_exp6', jobId, decoded, channels: outChannels }, [decoded.buffer]);
  } catch (err) {
    self.postMessage({ type: 'error', jobId, message: 'Trajectory decode: ' + err.message });
  }
}

// ---------------------------------------------------------------------------
// Message dispatcher
// ---------------------------------------------------------------------------

self.onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case 'init':
      modelsBaseUrl = msg.modelsBaseUrl;
      self.postMessage({ type: 'ready' });
      break;
    case 'process':
      // Experiment1 full pipeline (encode + decode in one pass)
      currentJobId = msg.jobId;
      runJob(msg);
      break;
    case 'encode':
      // Experiment2/3/4: encode one source, cache captures, return decoded audio
      currentJobId = msg.jobId;
      runEncode(msg);
      break;
    case 'interpolate':
      // Experiment2: interpolate using cached captures — no audio needed
      currentJobId = msg.jobId;
      runInterpolation(msg);
      break;
    case 'delta_transfer':
      // Experiment3: dynamics transfer using cached captures
      currentJobId = msg.jobId;
      runDeltaTransfer(msg);
      break;
    case 'get_emb_deltas':
      // Experiment3: return B embedding frame deltas for visualization (does not cancel jobs)
      runGetEmbDeltas(msg);
      break;
    case 'compute_svd4':
      // Experiment4: compute per-source SVD and store in cache (synchronous)
      runComputeSVD4(msg);
      break;
    case 'apply_svd4':
      // Experiment4: apply SVD component swap and decode
      currentJobId = msg.jobId;
      runApplySVD4(msg);
      break;
    case 'compute_svd5':
      // Experiment5: compute joint SVD on concatenated A+B embeddings
      runComputeSVD5(msg);
      break;
    case 'apply_svd5':
      // Experiment5: interpolate in shared V space and decode
      currentJobId = msg.jobId;
      runApplySVD5(msg);
      break;
    case 'get_embeddings_exp6': {
      // Experiment6: return quantized embeddings + codes for both sources.
      // captureCache entries are wrapped structs (mode: single/ola24/ola48);
      // flatten across all segments before sending.
      const capsA = captureCache.A, capsB = captureCache.B;
      if (!capsA || !capsB) {
        self.postMessage({ type: 'error', jobId: msg.jobId, message: 'Encode both sources first.' });
        break;
      }

      try {
        const flatA = flattenCaps(capsA);
        const flatB = flattenCaps(capsB);

        function toNumDims(dims) { return [...dims].map(d => Number(d)); }
        function codesToInt32(codes, type) {
          if (type === 'int64') {
            const out = new Int32Array(codes.length);
            for (let i = 0; i < codes.length; i++) out[i] = Number(codes[i]);
            return out;
          }
          return new Int32Array(codes);
        }

        const embEncA   = flatA.emb_enc.slice();
        const embEncB   = flatB.emb_enc.slice();
        const embQuantA = flatA.emb_quant.slice();
        const embQuantB = flatB.emb_quant.slice();
        const codesA    = codesToInt32(flatA.codes, flatA.codesType);
        const codesB    = codesToInt32(flatB.codes, flatB.codesType);
        const scalesA   = flattenScales(capsA); // Float32Array(T_A) or null
        const scalesB   = flattenScales(capsB); // Float32Array(T_B) or null

        const transfers = [embEncA.buffer, embEncB.buffer,
                           embQuantA.buffer, embQuantB.buffer,
                           codesA.buffer, codesB.buffer];
        if (scalesA) transfers.push(scalesA.buffer);
        if (scalesB) transfers.push(scalesB.buffer);

        self.postMessage({
          type:      'embeddings_exp6',
          jobId:     msg.jobId,
          embEncA,   embEncDimsA:  toNumDims(flatA.embEncDims),
          embEncB,   embEncDimsB:  toNumDims(flatB.embEncDims),
          embQuantA, embQuantDimsA: toNumDims(flatA.embQuantDims),
          embQuantB, embQuantDimsB: toNumDims(flatB.embQuantDims),
          codesA,    codesDimsA:   toNumDims(flatA.codesDims),
          codesB,    codesDimsB:   toNumDims(flatB.codesDims),
          scalesA,   scalesB,
        }, transfers);
      } catch (err) {
        self.postMessage({ type: 'error', jobId: msg.jobId, message: 'Embedding extract: ' + err.message });
      }
      break;
    }
    case 'play_frame_exp6':
      runPlayFrameExp6(msg);
      break;
    case 'decode_trajectory_exp6':
      runDecodeTrajectoryExp6(msg);
      break;
    case 'cancel':
      currentJobId = -1;
      break;
  }
};
