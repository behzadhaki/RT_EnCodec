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

function mixCodesElementWise(codes_a, codes_b, alpha) {
  const out = new (codes_a.constructor)(codes_a.length);
  for (let i = 0; i < out.length; i++) {
    const frac = ((Math.imul(i + 1, 0x9E3779B9) >>> 0) / 0xFFFFFFFF);
    out[i] = frac < alpha ? codes_b[i] : codes_a[i];
  }
  return out;
}

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
async function interpDecodeOnce(capA, capB, alpha, interpPoint, modelHz, hDec, cDec,
                                vqMode = 'flat_swap', levelAlphas = null) {
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
    const scale = (capA.scale && capB.scale) ? lerpF32(capA.scale, capB.scale, alpha) : null;
    return decodeFromLatents(qEmb, qDims, scale, capA.scaleDims, modelHz, hDec, cDec);
  }

  if (interpPoint === 'vq_codes') {
    let mixedCodes;
    switch (vqMode) {
      case 'frame_swap':
        mixedCodes = mixCodesByFrame(capA.codes, capB.codes, capA.codesDims, alpha);
        break;
      case 'level_swap':
        mixedCodes = mixCodesByLevel(capA.codes, capB.codes, capA.codesDims, levelAlphas || []);
        break;
      case 'int_lerp':
        mixedCodes = mixCodesIntLerpByLevel(capA.codes, capB.codes, capA.codesDims,
                       levelAlphas || new Array(capA.codesDims[1]).fill(alpha));
        break;
      default: // flat_swap
        mixedCodes = mixCodesElementWise(capA.codes, capB.codes, alpha);
    }
    const codesTensor = new ort.Tensor(capA.codesType, mixedCodes, capA.codesDims);
    const decCodesOut = await sessions.decCodes.run({ codes: codesTensor });
    const emb   = new Float32Array(decCodesOut.emb.data);
    const dims  = decCodesOut.emb.dims.slice();
    const scale = (capA.scale && capB.scale) ? lerpF32(capA.scale, capB.scale, alpha) : null;
    return decodeFromLatents(emb, dims, scale, capA.scaleDims, modelHz, hDec, cDec);
  }

  // quantized_embeddings
  const n         = Math.min(capA.emb_quant.length, capB.emb_quant.length);
  const interpEmb = lerpF32(capA.emb_quant.subarray(0, n), capB.emb_quant.subarray(0, n), alpha);
  const [B, C]    = capA.embQuantDims;
  const dims      = [B, C, n / (B * C)];
  const scale     = (capA.scale && capB.scale) ? lerpF32(capA.scale, capB.scale, alpha) : null;
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

// 48k is always stateless per segment (matches interpRun48k behaviour).
async function encodeAndCapture48k(audio, jobId) {
  const totalLen = audio.length;
  const outBuf   = new Float32Array(totalLen);
  const wgtBuf   = new Float32Array(totalLen);
  const numSegs  = Math.ceil(totalLen / STRIDE_48);
  const segments = [];

  for (let s = 0; s < numSegs; s++) {
    if (currentJobId !== jobId) return [null, null];

    const offset = s * STRIDE_48;
    const chunk  = new Float32Array(SEG_48);
    chunk.set(audio.subarray(offset, offset + SEG_48));

    const cap = await encodeCapture(chunk, '48k', zeroState(1), zeroState(1));
    const dec = await decodeFromLatents(
      cap.emb_quant, cap.embQuantDims, cap.scale, cap.scaleDims,
      '48k', zeroState(1), zeroState(1));

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
          vqMode = 'flat_swap', levelAlphas = null } = msg;

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
      ? await interpFromCached48k(capsA, capsB, alpha, interpPoint, vqMode, levelAlphas, jobId)
      : await interpFromCached24k(capsA, capsB, alpha, interpPoint, vqMode, levelAlphas, jobId);

    if (decoded === null) return;
    self.postMessage({ type: 'result', jobId, decoded }, [decoded.buffer]);

  } catch (err) {
    self.postMessage({ type: 'error', jobId, message: err.message });
  }
}

async function interpFromCached24k(capsA, capsB, alpha, interpPoint, vqMode, levelAlphas, jobId) {
  if (capsA.mode === 'single') {
    // Non-streaming: single interpDecodeOnce, zero decoder state
    self.postMessage({ type: 'progress', jobId, value: 0.5, status: 'Interpolating…' });
    const result = await interpDecodeOnce(
      capsA.cap, capsB.cap, alpha, interpPoint, '24k',
      zeroState(1), zeroState(1), vqMode, levelAlphas);
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
      capA, capB, alpha, interpPoint, '24k', hDec, cDec, vqMode, levelAlphas);
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

async function interpFromCached48k(capsA, capsB, alpha, interpPoint, vqMode, levelAlphas, jobId) {
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
      zeroState(1), zeroState(1), vqMode, levelAlphas);

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
      // Experiment2: encode one source, cache captures, return decoded audio
      currentJobId = msg.jobId;
      runEncode(msg);
      break;
    case 'interpolate':
      // Experiment2: interpolate using cached captures — no audio needed
      currentJobId = msg.jobId;
      runInterpolation(msg);
      break;
    case 'cancel':
      currentJobId = -1;
      break;
  }
};
