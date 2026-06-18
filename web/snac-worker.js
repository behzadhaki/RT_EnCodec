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

async function ensureSessions(model, jobId) {
  if (model === cachedModel && sessions) return;
  const base = `${modelsBaseUrl}/${model}`;
  progress(jobId, 0.03, 'Loading model layout…');
  meta = await (await fetch(`${base}/model.json`)).json();
  progress(jobId, 0.06, 'Loading ONNX graphs…');
  sessions = {
    encSeg:   await ort.InferenceSession.create(`${base}/encode_audio_segment.onnx`),
    quantEnc: await ort.InferenceSession.create(`${base}/quantize_encodings.onnx`),
    decCodes: await ort.InferenceSession.create(`${base}/decode_codes.onnx`),
    decAudio: await ort.InferenceSession.create(`${base}/decode_audio.onnx`),
  };
  cachedModel = model;
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
        const info = await encode(msg.audio, msg.model, msg.chunk, jobId);
        progress(jobId, 1, 'Encoded.');
        self.postMessage({ type: 'encoded', jobId, ...info });
        break;
      }

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
    self.postMessage({ type: 'error', jobId, message: err.message });
  }
};
