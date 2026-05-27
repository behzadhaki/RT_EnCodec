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
 *   { type: 'cancel',  jobId }
 *
 * Messages OUT
 * ------------
 *   { type: 'ready' }
 *   { type: 'progress', jobId, value, status }
 *   { type: 'result',   jobId, decoded: Float32Array }
 *   { type: 'error',    jobId, message }
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

// ---------------------------------------------------------------------------
// Single-chunk pipeline
// ---------------------------------------------------------------------------

async function processChunk(chunk, modelHz, hEnc, cEnc, hDec, cDec) {
  const T = chunk.length;

  // Build audio tensor — 24k: [1,1,T]  |  48k: [1,2,T] (mono duped to stereo)
  let audioTensor;
  if (modelHz === '48k') {
    const stereo = new Float32Array(2 * T);
    stereo.set(chunk, 0);
    stereo.set(chunk, T);
    audioTensor = new ort.Tensor('float32', stereo, [1, 2, T]);
  } else {
    audioTensor = new ort.Tensor('float32', chunk.slice(), [1, 1, T]);
  }

  // Stage 1 — encode_audio_segment
  const encOut = await sessions.encSeg.run({
    audio: audioTensor,
    h_in: stateTensor(hEnc, 1),
    c_in: stateTensor(cEnc, 1),
  });

  // Stage 2 — quantize_encodings  (emb output from stage 1 passed straight in)
  const quantOut = await sessions.quantEnc.run({ emb: encOut.emb });

  // Stage 3 — decode_codes
  const decCodesOut = await sessions.decCodes.run({ codes: quantOut.codes });

  // Stage 4 — decode_audio
  const decAudioInputs = { emb: decCodesOut.emb, h_in: stateTensor(hDec, 1), c_in: stateTensor(cDec, 1) };
  if (modelHz === '48k') decAudioInputs.scale = encOut.scale;   // 48k carries RMS scale
  const decOut = await sessions.decAudio.run(decAudioInputs);

  // Extract mono output
  let outAudio;
  if (modelHz === '48k') {
    // Shape [1,2,T_out] — average L and R channels
    const raw  = decOut.audio.data;
    const tOut = raw.length >> 1;
    outAudio   = new Float32Array(tOut);
    for (let i = 0; i < tOut; i++) outAudio[i] = (raw[i] + raw[tOut + i]) * 0.5;
  } else {
    outAudio = new Float32Array(decOut.audio.data);   // [1,1,T] → flat T
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

// 24 kHz streaming OLA
// The causal encoder's Conv1d stack has a receptive field of ~600-1200 input
// samples; without context from the previous chunk the first encoder frames are
// corrupted, producing an audible click at every boundary. Extending each
// segment by HALF_OV_24 samples on both sides gives the encoder the causal
// context it needs; the triangle crossfade then blends the overlap region.
// HALF_OV_24 = 640 = 2 × 320 (two encoder frames, ≈ 27 ms).
const HALF_OV_24 = 640;

// 48 kHz OLA  (mirrors _linear_overlap_add in rt_encodec)
const SEG_48    = 48000;   // 1.0 s segment
const STRIDE_48 = 43200;   // 0.9 s stride → 4800-sample (100 ms) overlap
                           // The original Python model uses 47520 (10 ms), but
                           // in the browser the decoder LSTM starts from zero state
                           // each segment and needs ~15 frames to warm up; 100 ms
                           // gives the triangle crossfade enough room to hide it.

// Triangle window: weight[i] = 0.5 − |t[i] − 0.5|, t = linspace(0,1,n+2)[1:−1]
const TRI_WIN_48 = new Float32Array(SEG_48);
for (let i = 0; i < SEG_48; i++) {
  const t = (i + 1) / (SEG_48 + 1);
  TRI_WIN_48[i] = 0.5 - Math.abs(t - 0.5);
}

// ---------------------------------------------------------------------------
// Job runner
// ---------------------------------------------------------------------------

let currentJobId = -1;

async function runJob(msg) {
  const { jobId, audio, modelHz, bwKbps, streaming, frameSize } = msg;

  try {
    self.postMessage({ type: 'progress', jobId, value: 0, status: 'Loading models…' });
    await ensureSessions(modelHz, bwKbps);
    if (currentJobId !== jobId) return;

    let decoded;

    if (modelHz === '48k') {
      decoded = await run48k(audio, streaming, jobId);
    } else {
      decoded = await run24k(audio, streaming, frameSize, jobId);
    }

    if (decoded === null) return;   // cancelled
    self.postMessage({ type: 'result', jobId, decoded }, [decoded.buffer]);

  } catch (err) {
    self.postMessage({ type: 'error', jobId, message: err.message });
  }
}

// ---------------------------------------------------------------------------
// 24 kHz runner
//   Non-streaming — single pass, zero LSTM state throughout.
//   Streaming     — triangle-windowed overlap-add (stride = frameSize,
//                   window = frameSize + 2×HALF_OV_24), LSTM state carried.
//                   The ±HALF_OV_24 extension gives the causal encoder the
//                   cross-boundary context it needs to avoid click artifacts.
// ---------------------------------------------------------------------------

async function run24k(audio, streaming, frameSize, jobId) {
  if (!streaming) {
    // Single-pass, stateless
    if (currentJobId !== jobId) return null;
    self.postMessage({ type: 'progress', jobId, value: 0.5, status: 'Processing… 50 %' });
    const result = await processChunk(
      audio.slice(), '24k', zeroState(1), zeroState(1), zeroState(1), zeroState(1));
    if (currentJobId !== jobId) return null;
    self.postMessage({ type: 'progress', jobId, value: 1,   status: 'Processing… 100 %' });
    return result.outAudio.slice(0, audio.length);
  }

  // --- Streaming: OLA ---
  const stride = frameSize;
  const seg    = stride + 2 * HALF_OV_24;   // window length; multiple of 320 ✓

  // Triangle window for this segment length
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

    const off = s * stride - HALF_OV_24;   // window start in audio coords (may be < 0)

    // Build input: zero-padded where out of range
    const chunk  = new Float32Array(seg);  // zero-initialised
    const srcS   = Math.max(off, 0);
    const srcE   = Math.min(off + seg, totalLen);
    if (srcE > srcS) chunk.set(audio.subarray(srcS, srcE), srcS - off);

    const result = await processChunk(chunk, '24k', hEnc, cEnc, hDec, cDec);
    hEnc = result.hEncNew; cEnc = result.cEncNew;
    hDec = result.hDecNew; cDec = result.cDecNew;

    // Weighted accumulate — only within valid (non-padded) output range
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

  // Normalise by accumulated window weights
  for (let i = 0; i < totalLen; i++) {
    if (wgtBuf[i] > 0) outBuf[i] /= wgtBuf[i];
  }

  return outBuf;
}

// ---------------------------------------------------------------------------
// 48 kHz runner  (triangle-windowed overlap-add, stride = 0.99 s)
// Mirrors rt_encodec._linear_overlap_add. Streaming flag controls whether
// LSTM state is carried across segment boundaries.
// ---------------------------------------------------------------------------

async function run48k(audio, streaming, jobId) {
  const totalLen = audio.length;
  const outBuf   = new Float32Array(totalLen);
  const wgtBuf   = new Float32Array(totalLen);

  let hEnc = zeroState(1), cEnc = zeroState(1);
  let hDec = zeroState(1), cDec = zeroState(1);

  // Number of segments: same formula as the Python model
  const numSegs = Math.ceil(totalLen / STRIDE_48);

  for (let seg = 0; seg < numSegs; seg++) {
    if (currentJobId !== jobId) return null;

    const offset = seg * STRIDE_48;
    const chunk  = new Float32Array(SEG_48);           // zero-padded
    chunk.set(audio.subarray(offset, offset + SEG_48));

    const result = await processChunk(chunk, '48k', hEnc, cEnc, hDec, cDec);

    if (streaming) {
      hEnc = result.hEncNew; cEnc = result.cEncNew;
      hDec = result.hDecNew; cDec = result.cDecNew;
    } else {
      hEnc = zeroState(1); cEnc = zeroState(1);
      hDec = zeroState(1); cDec = zeroState(1);
    }

    // Weighted accumulate — only within the valid (non-padded) output range
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

  // Normalise by accumulated window weights
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
// All four modes use a deterministic golden-ratio hash so results are
// reproducible across slider moves at the same alpha value.

// Mode 1 — element-wise: each scalar code position independently picks A or B.
function mixCodesElementWise(codes_a, codes_b, alpha) {
  const out = new (codes_a.constructor)(codes_a.length);
  for (let i = 0; i < out.length; i++) {
    const frac = ((Math.imul(i + 1, 0x9E3779B9) >>> 0) / 0xFFFFFFFF);
    out[i] = frac < alpha ? codes_b[i] : codes_a[i];
  }
  return out;
}

// Mode 2 — per-frame: all N_q levels at a given time frame come from the
// same source (A or B), preserving intra-frame RVQ coherence.
function mixCodesByFrame(codes_a, codes_b, dims, alpha) {
  const N_q = dims[1], T = dims[2];
  const out = new (codes_a.constructor)(codes_a.length);
  for (let t = 0; t < T; t++) {
    const frac = ((Math.imul(t + 1, 0x9E3779B9) >>> 0) / 0xFFFFFFFF);
    const useB = frac < alpha;
    for (let k = 0; k < N_q; k++) {
      const idx = k * T + t;
      out[idx] = useB ? codes_b[idx] : codes_a[idx];
    }
  }
  return out;
}

// Mode 3 — per-level: each RVQ level k has its own alpha (levelAlphas[k]).
// Within a level, each time frame independently picks A or B via a hash
// seeded on (k, t) so levels don't correlate with each other.
function mixCodesByLevel(codes_a, codes_b, dims, levelAlphas) {
  const N_q = dims[1], T = dims[2];
  const out = new (codes_a.constructor)(codes_a.length);
  for (let k = 0; k < N_q; k++) {
    const alpha_k = levelAlphas[k] ?? 0.5;
    for (let t = 0; t < T; t++) {
      const seed = (Math.imul(k + 1, 49999) + t + 1) | 0;
      const frac = ((Math.imul(seed, 0x9E3779B9) >>> 0) / 0xFFFFFFFF);
      const idx  = k * T + t;
      out[idx] = frac < alpha_k ? codes_b[idx] : codes_a[idx];
    }
  }
  return out;
}

// Mode 4 — integer lerp per level: each RVQ level k uses its own alpha_k.
// round( (1-α_k)·code_a[k,t] + α_k·code_b[k,t] ) for every time frame t.
// Codebooks are not acoustically ordered — this is purely exploratory.
function mixCodesIntLerpByLevel(codes_a, codes_b, dims, levelAlphas) {
  const N_q    = dims[1], T = dims[2];
  const out    = new (codes_a.constructor)(codes_a.length);
  const isBig  = codes_a instanceof BigInt64Array;
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

// Run stages 1–3 (encode → quantize → decode_codes) and capture all
// intermediate tensors needed for any of the three interpolation points.
async function encodeCapture(chunk, modelHz, hEnc, cEnc) {
  const T = chunk.length;
  let audioTensor;
  if (modelHz === '48k') {
    const stereo = new Float32Array(2 * T);
    stereo.set(chunk, 0); stereo.set(chunk, T);
    audioTensor = new ort.Tensor('float32', stereo, [1, 2, T]);
  } else {
    audioTensor = new ort.Tensor('float32', chunk.slice(), [1, 1, T]);
  }

  // Stage 1 — encoder
  const encOut = await sessions.encSeg.run({
    audio: audioTensor,
    h_in:  stateTensor(hEnc, 1),
    c_in:  stateTensor(cEnc, 1),
  });

  // Stage 2 — vector quantizer
  const quantOut = await sessions.quantEnc.run({ emb: encOut.emb });

  // Stage 3 — code decoder (reconstructs continuous embedding from codes)
  const decCodesOut = await sessions.decCodes.run({ codes: quantOut.codes });

  return {
    // Point 1 — encoder latents (pre-VQ continuous embedding)
    emb_enc:     new Float32Array(encOut.emb.data),
    embEncDims:  encOut.emb.dims.slice(),
    // Point 2 — VQ codes (discrete)
    codes:       quantOut.codes.data.slice(),   // Int32Array or BigInt64Array
    codesType:   quantOut.codes.type,
    codesDims:   quantOut.codes.dims.slice(),
    // Point 3 — quantized embeddings (post-VQ continuous embedding)
    emb_quant:   new Float32Array(decCodesOut.emb.data),
    embQuantDims: decCodesOut.emb.dims.slice(),
    // RMS scale (48k only, needed by decode_audio)
    scale:       encOut.scale ? new Float32Array(encOut.scale.data) : null,
    scaleDims:   encOut.scale ? encOut.scale.dims.slice() : null,
    // Updated encoder LSTM state
    hEncNew: new Float32Array(encOut.h_out.data),
    cEncNew: new Float32Array(encOut.c_out.data),
  };
}

// Run stage 4 (decode_audio) from a continuous embedding tensor.
async function decodeFromLatents(emb, embDims, scale, scaleDims, modelHz, hDec, cDec) {
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
    const raw  = decOut.audio.data;
    const tOut = raw.length >> 1;
    outAudio   = new Float32Array(tOut);
    for (let i = 0; i < tOut; i++) outAudio[i] = (raw[i] + raw[tOut + i]) * 0.5;
  } else {
    outAudio = new Float32Array(decOut.audio.data);
  }
  return {
    outAudio,
    hDecNew: new Float32Array(decOut.h_out.data),
    cDecNew: new Float32Array(decOut.c_out.data),
  };
}

// ---------------------------------------------------------------------------
// Interpolation runners
// ---------------------------------------------------------------------------

async function runInterpolation(msg) {
  const { jobId, audio_a, audio_b, modelHz, bwKbps, streaming, frameSize,
          alpha, interpPoint,
          vqMode = 'element_wise', levelAlphas = null } = msg;
  try {
    self.postMessage({ type: 'progress', jobId, value: 0, status: 'Loading models…' });
    await ensureSessions(modelHz, bwKbps);
    if (currentJobId !== jobId) return;

    const decoded = modelHz === '48k'
      ? await interpRun48k(audio_a, audio_b, alpha, interpPoint, vqMode, levelAlphas, jobId)
      : await interpRun24k(audio_a, audio_b, alpha, interpPoint, streaming, frameSize, vqMode, levelAlphas, jobId);

    if (decoded === null) return;
    self.postMessage({ type: 'result', jobId, decoded }, [decoded.buffer]);
  } catch (err) {
    self.postMessage({ type: 'error', jobId, message: err.message });
  }
}

async function interpRun24k(audio_a, audio_b, alpha, interpPoint, streaming, frameSize, vqMode, levelAlphas, jobId) {
  const totalLen = Math.min(audio_a.length, audio_b.length);
  const a = audio_a.subarray(0, totalLen);
  const b = audio_b.subarray(0, totalLen);

  if (!streaming) {
    // ── Non-streaming single pass ──────────────────────────────────────────
    self.postMessage({ type: 'progress', jobId, value: 0.2, status: 'Encoding A…' });
    const capA = await encodeCapture(a.slice(), '24k', zeroState(1), zeroState(1));
    if (currentJobId !== jobId) return null;

    self.postMessage({ type: 'progress', jobId, value: 0.4, status: 'Encoding B…' });
    const capB = await encodeCapture(b.slice(), '24k', zeroState(1), zeroState(1));
    if (currentJobId !== jobId) return null;

    self.postMessage({ type: 'progress', jobId, value: 0.7, status: 'Interpolating…' });
    const result = await interpDecodeOnce(capA, capB, alpha, interpPoint, '24k',
                                          zeroState(1), zeroState(1), vqMode, levelAlphas);
    if (currentJobId !== jobId) return null;

    self.postMessage({ type: 'progress', jobId, value: 1.0, status: 'Done.' });
    return result.outAudio.slice(0, totalLen);
  }

  // ── Streaming OLA ──────────────────────────────────────────────────────
  const stride  = frameSize;
  const seg     = stride + 2 * HALF_OV_24;
  const win     = new Float32Array(seg);
  for (let i = 0; i < seg; i++) {
    const t = (i + 1) / (seg + 1);
    win[i] = 0.5 - Math.abs(t - 0.5);
  }

  const outBuf  = new Float32Array(totalLen);
  const wgtBuf  = new Float32Array(totalLen);
  const numSegs = Math.ceil(totalLen / stride);

  let hEncA = zeroState(1), cEncA = zeroState(1);
  let hEncB = zeroState(1), cEncB = zeroState(1);
  let hDec  = zeroState(1), cDec  = zeroState(1);

  for (let s = 0; s < numSegs; s++) {
    if (currentJobId !== jobId) return null;

    const off  = s * stride - HALF_OV_24;
    const srcS = Math.max(off, 0);
    const srcE = Math.min(off + seg, totalLen);

    const chunkA = new Float32Array(seg);
    const chunkB = new Float32Array(seg);
    if (srcE > srcS) {
      chunkA.set(a.subarray(srcS, srcE), srcS - off);
      chunkB.set(b.subarray(srcS, srcE), srcS - off);
    }

    const capA = await encodeCapture(chunkA, '24k', hEncA, cEncA);
    hEncA = capA.hEncNew; cEncA = capA.cEncNew;

    const capB = await encodeCapture(chunkB, '24k', hEncB, cEncB);
    hEncB = capB.hEncNew; cEncB = capB.cEncNew;

    const dec = await interpDecodeOnce(capA, capB, alpha, interpPoint, '24k', hDec, cDec, vqMode, levelAlphas);
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

async function interpRun48k(audio_a, audio_b, alpha, interpPoint, vqMode, levelAlphas, jobId) {
  const totalLen = Math.min(audio_a.length, audio_b.length);
  const a = audio_a.subarray(0, totalLen);
  const b = audio_b.subarray(0, totalLen);

  const outBuf  = new Float32Array(totalLen);
  const wgtBuf  = new Float32Array(totalLen);
  const numSegs = Math.ceil(totalLen / STRIDE_48);

  for (let seg = 0; seg < numSegs; seg++) {
    if (currentJobId !== jobId) return null;

    const offset = seg * STRIDE_48;
    const chunkA = new Float32Array(SEG_48);
    const chunkB = new Float32Array(SEG_48);
    chunkA.set(a.subarray(offset, offset + SEG_48));
    chunkB.set(b.subarray(offset, offset + SEG_48));

    // 48k is always stateless per segment
    const capA = await encodeCapture(chunkA, '48k', zeroState(1), zeroState(1));
    const capB = await encodeCapture(chunkB, '48k', zeroState(1), zeroState(1));
    const dec  = await interpDecodeOnce(capA, capB, alpha, interpPoint, '48k',
                                        zeroState(1), zeroState(1), vqMode, levelAlphas);

    const validEnd = Math.min(offset + SEG_48, totalLen);
    for (let i = 0; i < validEnd - offset; i++) {
      outBuf[offset + i] += TRI_WIN_48[i] * dec.outAudio[i];
      wgtBuf[offset + i] += TRI_WIN_48[i];
    }

    self.postMessage({
      type: 'progress', jobId,
      value: Math.min((seg + 1) / numSegs, 1),
      status: `Interpolating… ${Math.round((seg + 1) / numSegs * 100)} %`,
    });
  }

  for (let i = 0; i < totalLen; i++) {
    if (wgtBuf[i] > 0) outBuf[i] /= wgtBuf[i];
  }
  return outBuf;
}

// Apply interpolation at the chosen point then decode to audio.
// capA / capB are encodeCapture() results for the two sources.
async function interpDecodeOnce(capA, capB, alpha, interpPoint, modelHz, hDec, cDec,
                                vqMode = 'element_wise', levelAlphas = null) {
  if (interpPoint === 'encoder_latents') {
    // Lerp continuous encoder embeddings, then re-quantize so the decoder
    // stays in-distribution (it only saw quantized embeddings during training).
    const n         = Math.min(capA.emb_enc.length, capB.emb_enc.length);
    const interpEmb = lerpF32(capA.emb_enc.subarray(0, n), capB.emb_enc.subarray(0, n), alpha);
    const [B, C]    = capA.embEncDims;
    const dims      = [B, C, n / (B * C)];
    // Re-quantize: snap interpolated embedding to nearest codebook entries.
    const quantOut    = await sessions.quantEnc.run({
      emb: new ort.Tensor('float32', interpEmb, dims),
    });
    const decCodesOut = await sessions.decCodes.run({ codes: quantOut.codes });
    const qEmb  = new Float32Array(decCodesOut.emb.data);
    const qDims = decCodesOut.emb.dims.slice();
    const scale = (capA.scale && capB.scale) ? lerpF32(capA.scale, capB.scale, alpha) : null;
    return decodeFromLatents(qEmb, qDims, scale, capA.scaleDims, modelHz, hDec, cDec);
  }

  if (interpPoint === 'vq_codes') {
    // Mix discrete VQ codes according to the chosen sub-mode, then
    // re-embed via decode_codes and decode to audio.
    let mixedCodes;
    switch (vqMode) {
      case 'per_frame':
        mixedCodes = mixCodesByFrame(capA.codes, capB.codes, capA.codesDims, alpha);
        break;
      case 'per_level':
        mixedCodes = mixCodesByLevel(capA.codes, capB.codes, capA.codesDims, levelAlphas || []);
        break;
      case 'int_lerp':
        mixedCodes = mixCodesIntLerpByLevel(capA.codes, capB.codes, capA.codesDims,
                       levelAlphas || new Array(capA.codesDims[1]).fill(alpha));
        break;
      default: // element_wise
        mixedCodes = mixCodesElementWise(capA.codes, capB.codes, alpha);
    }
    const codesTensor = new ort.Tensor(capA.codesType, mixedCodes, capA.codesDims);
    const decCodesOut = await sessions.decCodes.run({ codes: codesTensor });
    const emb   = new Float32Array(decCodesOut.emb.data);
    const dims  = decCodesOut.emb.dims.slice();
    const scale = (capA.scale && capB.scale) ? lerpF32(capA.scale, capB.scale, alpha) : null;
    return decodeFromLatents(emb, dims, scale, capA.scaleDims, modelHz, hDec, cDec);
  }

  // quantized_embeddings — lerp the decode_codes output embeddings.
  const n        = Math.min(capA.emb_quant.length, capB.emb_quant.length);
  const interpEmb = lerpF32(capA.emb_quant.subarray(0, n), capB.emb_quant.subarray(0, n), alpha);
  const [B, C]   = capA.embQuantDims;
  const dims     = [B, C, n / (B * C)];
  const scale    = (capA.scale && capB.scale) ? lerpF32(capA.scale, capB.scale, alpha) : null;
  return decodeFromLatents(interpEmb, dims, scale, capA.scaleDims, modelHz, hDec, cDec);
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
      currentJobId = msg.jobId;
      runJob(msg);
      break;
    case 'interpolate':
      currentJobId = msg.jobId;
      runInterpolation(msg);
      break;
    case 'cancel':
      currentJobId = -1;
      break;
  }
};
