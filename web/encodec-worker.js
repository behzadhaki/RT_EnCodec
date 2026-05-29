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
// Single-chunk pipeline  (used by experiment1 'process' handler only)
// ---------------------------------------------------------------------------

async function processChunk(chunk, modelHz, hEnc, cEnc, hDec, cDec) {
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

  const quantOut = await sessions.quantEnc.run({ emb: encOut.emb });
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

const SEG_48    = 48000;
const STRIDE_48 = 43200;

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

    const offset = seg * STRIDE_48;
    const chunk  = new Float32Array(SEG_48);
    chunk.set(audio.subarray(offset, offset + SEG_48));

    const result = await processChunk(chunk, '48k', hEnc, cEnc, hDec, cDec);

    if (streaming) {
      hEnc = result.hEncNew; cEnc = result.cEncNew;
      hDec = result.hDecNew; cDec = result.cDecNew;
    } else {
      hEnc = zeroState(1); cEnc = zeroState(1);
      hDec = zeroState(1); cDec = zeroState(1);
    }

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

  const encOut = await sessions.encSeg.run({
    audio: audioTensor,
    h_in:  stateTensor(hEnc, 1),
    c_in:  stateTensor(cEnc, 1),
  });

  const quantOut     = await sessions.quantEnc.run({ emb: encOut.emb });
  const decCodesOut  = await sessions.decCodes.run({ codes: quantOut.codes });

  return {
    emb_enc:     new Float32Array(encOut.emb.data),
    embEncDims:  encOut.emb.dims.slice(),
    codes:       quantOut.codes.data.slice(),
    codesType:   quantOut.codes.type,
    codesDims:   quantOut.codes.dims.slice(),
    emb_quant:   new Float32Array(decCodesOut.emb.data),
    embQuantDims: decCodesOut.emb.dims.slice(),
    scale:       encOut.scale ? new Float32Array(encOut.scale.data) : null,
    scaleDims:   encOut.scale ? encOut.scale.dims.slice() : null,
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
  const { jobId, source, audio, modelHz, bwKbps, streaming, frameSize } = msg;

  try {
    self.postMessage({ type: 'progress', jobId, value: 0, status: 'Loading models…' });
    await ensureSessions(modelHz, bwKbps);
    if (currentJobId !== jobId) return;

    let decoded, captures;
    if (modelHz === '48k') {
      [decoded, captures] = await encodeAndCapture48k(audio, jobId);
    } else {
      [decoded, captures] = await encodeAndCapture24k(audio, streaming, frameSize, jobId);
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
async function encodeAndCapture24k(audio, streaming, frameSize, jobId) {
  const totalLen = audio.length;

  if (!streaming) {
    self.postMessage({ type: 'progress', jobId, value: 0.3, status: 'Encoding…' });
    const cap = await encodeCapture(audio.slice(), '24k', zeroState(1), zeroState(1));
    if (currentJobId !== jobId) return [null, null];

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

  const outBuf  = new Float32Array(totalLen);
  const wgtBuf  = new Float32Array(totalLen);
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

    const dec = await decodeFromLatents(
      cap.emb_quant, cap.embQuantDims, cap.scale, cap.scaleDims,
      '24k', hDec, cDec);
    hDec = dec.hDecNew; cDec = dec.cDecNew;

    chunks.push({ cap, srcS, srcE, off });

    for (let i = srcS - off; i < srcE - off; i++) {
      const pos = off + i;
      outBuf[pos] += win[i] * dec.outAudio[i];
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
  return [outBuf, { mode: 'ola24', stride, seg, win, totalLen, chunks }];
}

// 48k uses stateful encoder and decoder so that LSTM context is continuous across
// segment boundaries, eliminating embedding jumps at every STRIDE_48 boundary.
async function encodeAndCapture48k(audio, jobId) {
  const totalLen = audio.length;
  const outBuf   = new Float32Array(totalLen);
  const wgtBuf   = new Float32Array(totalLen);
  const numSegs  = Math.ceil(totalLen / STRIDE_48);
  const segments = [];

  let hEnc = zeroState(1), cEnc = zeroState(1);
  let hDec = zeroState(1), cDec = zeroState(1);

  for (let s = 0; s < numSegs; s++) {
    if (currentJobId !== jobId) return [null, null];

    const offset = s * STRIDE_48;
    const chunk  = new Float32Array(SEG_48);
    chunk.set(audio.subarray(offset, offset + SEG_48));

    const cap = await encodeCapture(chunk, '48k', hEnc, cEnc);
    hEnc = cap.hEncNew; cEnc = cap.cEncNew;

    const dec = await decodeFromLatents(
      cap.emb_quant, cap.embQuantDims, cap.scale, cap.scaleDims,
      '48k', hDec, cDec);
    hDec = dec.hDecNew; cDec = dec.cDecNew;

    const validEnd = Math.min(offset + SEG_48, totalLen);
    segments.push({ cap, offset, validEnd });

    for (let i = 0; i < validEnd - offset; i++) {
      outBuf[offset + i] += TRI_WIN_48[i] * dec.outAudio[i];
      wgtBuf[offset + i] += TRI_WIN_48[i];
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
        ? computeFiltDeltaFull(svd.U, svd.deltas, svd.C, svd.totalT, activeComps)
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
function symmetricEigen(flatA, n) {
  const A = flatA.slice();
  const V = new Float32Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1.0;

  for (let sweep = 0; sweep < 30; sweep++) {
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
// deltas: Float32Array(C×totalT)
// activeComps: number[] — sorted list of component indices to include
// Returns filtered Float32Array(C×totalT).
function computeFiltDeltaFull(U, deltas, C, totalT, activeComps) {
  const filt = new Float32Array(C * totalT);
  for (const k of activeComps) {
    if (k < 0 || k >= C) continue;
    for (let t = 0; t < totalT; t++) {
      let coeff = 0;
      for (let c = 0; c < C; c++) coeff += U[c*C+k] * deltas[c*totalT+t];
      for (let c = 0; c < C; c++) filt[c*totalT+t] += U[c*C+k] * coeff;
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

  // Covariance of deltas: cov[i,j] = Σ_t delta[i,t]·delta[j,t] / totalT
  const cov = new Float32Array(C * C);
  for (let i = 0; i < C; i++) {
    for (let j = i; j < C; j++) {
      let s = 0;
      for (let t = 0; t < totalT; t++) s += deltas[i*totalT+t] * deltas[j*totalT+t];
      cov[i*C+j] = cov[j*C+i] = s / totalT;
    }
  }
  const { eigenvalues, eigenvectors } = symmetricEigen(cov, C);
  const svdS = new Float32Array(C);
  for (let k = 0; k < C; k++) svdS[k] = Math.sqrt(eigenvalues[k]);

  // Cache for SVD-mode transfer (deltas NOT transferred so worker retains the reference)
  captureCache.B.svd = { U: eigenvectors, deltas, C, totalT };

  self.postMessage(
    { type: 'emb_deltas', data: deltas, variance, C, T: totalT, svdS },
    [variance.buffer, svdS.buffer]
  );
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
      // Experiment2/3: encode one source, cache captures, return decoded audio
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
    case 'cancel':
      currentJobId = -1;
      break;
  }
};
