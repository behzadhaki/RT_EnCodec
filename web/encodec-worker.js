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
    case 'cancel':
      currentJobId = -1;
      break;
  }
};
