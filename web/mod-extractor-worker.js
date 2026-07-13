importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js');

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
ort.env.wasm.numThreads = 1;

let encodecBaseUrl = '../serialization/encodec_onnx_exports';
let snacBaseUrl = '../serialization/snac_onnx_exports';
let sessions = null;
let codebook = null;
let cbMeta = null;

function setProgress(value, status) {
  postMessage({ type: 'progress', value, status });
}

function parseModelId(modelId) {
  const parts = modelId.split(':');
  const family = parts[0];
  const variant = parts[1];
  const bwKbps = parts[2] || null;
  return { family, variant, bwKbps };
}

async function ensureModel(modelId) {
  if (sessions && sessions._key === modelId) return;
  const { family, variant, bwKbps } = parseModelId(modelId);
  if (family === 'encodec') {
    const base = `${encodecBaseUrl}/${variant}/${bwKbps}kbps`;
    sessions = {
      encSeg:   await ort.InferenceSession.create(`${base}/encode_audio_segment.onnx`),
      quantEnc: await ort.InferenceSession.create(`${base}/quantize_encodings.onnx`),
      _key: modelId, _hopSize: 320, _sr: parseInt(variant) * 1000,
      _strides: null,
    };
  } else {
    const base = `${snacBaseUrl}/${variant}`;
    const m = await (await fetch(`${base}/model.json`)).json();
    sessions = {
      encSeg:   await ort.InferenceSession.create(`${base}/encode_audio_segment.onnx`),
      quantEnc: await ort.InferenceSession.create(`${base}/quantize_encodings.onnx`),
      _key: modelId, _hopSize: m.hop_length, _sr: m.sampling_rate,
      _strides: m.vq_strides, _preprocessPad: m.preprocess_pad,
    };
  }
}

async function ensureCodebook(modelId) {
  const { family, variant } = parseModelId(modelId);
  const baseUrl = family === 'encodec' ? encodecBaseUrl : snacBaseUrl;
  const key = `${family}:${variant}`;
  if (codebook && cbMeta?._key === key) return;
  const meta = await (await fetch(`${baseUrl}/${variant}/codebooks.json`)).json();
  meta._key = key;
  const buf = await (await fetch(`${baseUrl}/${variant}/codebooks.bin`)).arrayBuffer();
  codebook = new Float32Array(buf);
  cbMeta = meta;
}

function zeroState() {
  return {
    h: new Float32Array(2 * 512),
    c: new Float32Array(2 * 512),
  };
}

function encTensors(s) {
  return {
    h_in: new ort.Tensor('float32', s.h, [2, 1, 512]),
    c_in: new ort.Tensor('float32', s.c, [2, 1, 512]),
  };
}

function deltaL2(eOff, prev, dim) {
  let d2 = 0;
  for (let d = 0; d < dim; d++) {
    const diff = codebook[eOff + d] - prev[d];
    d2 += diff * diff;
  }
  return Math.sqrt(d2);
}

function deltaCosine(eOff, prev, dim) {
  let dot = 0, nCur = 0, nPrev = 0;
  for (let d = 0; d < dim; d++) {
    const cur = codebook[eOff + d];
    dot += cur * prev[d];
    nCur += cur * cur;
    nPrev += prev[d] * prev[d];
  }
  const denom = Math.sqrt(nCur) * Math.sqrt(nPrev);
  const sim = denom > 1e-10 ? dot / denom : 1;
  return 1 - Math.max(-1, Math.min(1, sim));
}

function encodecDeltas(codes, nLevels, nFrames, metric) {
  const { vocab_size, dim } = cbMeta;
  const deltaFn = metric === 'cosine' ? deltaCosine : deltaL2;
  const curves = new Float32Array(nLevels * nFrames);
  for (let k = 0; k < nLevels; k++) {
    const kOff = k * vocab_size * dim;
    const prev = new Float32Array(dim);
    for (let t = 0; t < nFrames; t++) {
      const idx = codes[k * nFrames + t];
      if (idx < 0 || idx >= vocab_size) {
        if (t > 0) curves[k * nFrames + t] = curves[k * nFrames + t - 1];
        continue;
      }
      const eOff = kOff + idx * dim;
      if (t === 0) {
        curves[k * nFrames] = 0;
      } else {
        curves[k * nFrames + t] = deltaFn(eOff, prev, dim);
      }
      for (let d = 0; d < dim; d++) prev[d] = codebook[eOff + d];
    }
  }
  return curves;
}

function snacDeltas(quantOut, strides, nLevels, metric, upsample) {
  const { vocab_size, dim } = cbMeta;
  const deltaFn = metric === 'cosine' ? deltaCosine : deltaL2;
  const codeNames = Array.from({ length: nLevels }, (_, i) => `codes_${i}`);
  const stride1Idx = nLevels - 1;
  const nFrames = quantOut[codeNames[stride1Idx]].data.length;
  const codes = new Int32Array(nLevels * nFrames);
  const curves = new Float32Array(nLevels * nFrames);

  for (let k = 0; k < nLevels; k++) {
    const stride = strides[k];
    const rawCodes = quantOut[codeNames[k]].data;
    const Tk = rawCodes.length;
    const kOff = k * vocab_size * dim;
    const prev = new Float32Array(dim);

    if (upsample === 'step') {
      // Repeat codes to base rate, then compute deltas at base rate
      for (let t = 0; t < nFrames; t++) {
        const codeIdx = Math.min(Math.floor(t / stride), Tk - 1);
        const idx = Number(rawCodes[codeIdx]);
        codes[k * nFrames + t] = idx;
        if (idx < 0 || idx >= vocab_size) {
          if (t > 0) curves[k * nFrames + t] = curves[k * nFrames + t - 1];
          continue;
        }
        const eOff = kOff + idx * dim;
        if (t === 0) {
          curves[k * nFrames] = 0;
        } else {
          curves[k * nFrames + t] = deltaFn(eOff, prev, dim);
        }
        for (let d = 0; d < dim; d++) prev[d] = codebook[eOff + d];
      }
    } else {
      // Compute deltas at native rate, then lerp to base frame rate
      const native = new Float32Array(Tk);
      for (let t = 0; t < Tk; t++) {
        const idx = Number(rawCodes[t]);
        for (let s = 0; s < stride; s++) {
          const p = t * stride + s;
          if (p < nFrames) codes[k * nFrames + p] = idx;
        }
        if (idx < 0 || idx >= vocab_size) {
          if (t > 0) native[t] = native[t - 1];
          continue;
        }
        const eOff = kOff + idx * dim;
        if (t === 0) {
          native[t] = 0;
        } else {
          native[t] = deltaFn(eOff, prev, dim);
        }
        for (let d = 0; d < dim; d++) prev[d] = codebook[eOff + d];
      }
      for (let t = 0; t < nFrames; t++) {
        const srcPos = t / stride;
        const lo = Math.min(Math.floor(srcPos), Tk - 1);
        const hi = Math.min(lo + 1, Tk - 1);
        const frac = srcPos - lo;
        curves[k * nFrames + t] = native[lo] + frac * (native[hi] - native[lo]);
      }
    }
  }
  return { curves, codes, nFrames };
}

async function runEncode(msg) {
  const { jobId, audio, modelId } = msg;
  try {
    setProgress(0, 'Loading model…');
    await ensureModel(modelId);
    await ensureCodebook(modelId);

    let T = audio.length;
    const { family, variant } = parseModelId(modelId);
    const hasStereoInput = (family === 'encodec' && variant === '48k');
    const hasState = (family === 'encodec');
    // Pad SNAC audio to preprocess_pad multiple for attention alignment
    let audioData = audio;
    if (family !== 'encodec' && sessions._preprocessPad && T % sessions._preprocessPad !== 0) {
      const newLen = Math.ceil(T / sessions._preprocessPad) * sessions._preprocessPad;
      const padded = new Float32Array(newLen);
      padded.set(audio);
      audioData = padded;
      T = newLen;
    }
    setProgress(0.15, 'Encoding…');
    let audioTensor;
    if (hasStereoInput) {
      const stereo = new Float32Array(2 * T);
      stereo.set(audioData, 0); stereo.set(audioData, T);
      audioTensor = new ort.Tensor('float32', stereo, [1, 2, T]);
    } else {
      audioTensor = new ort.Tensor('float32', audioData, [1, 1, T]);
    }
    const st = hasState ? zeroState() : null;
    const encOut = await sessions.encSeg.run({
      audio: audioTensor,
      ...(hasState ? encTensors(st) : {}),
    });

    setProgress(0.4, 'Quantizing…');

    const metric = msg.metric || 'l2';
    let curves, codes, nLevels, nFrames;

    if (family === 'encodec') {
      const quantOut = await sessions.quantEnc.run({ emb: encOut.emb });
      const codesRaw = quantOut.codes.data;
      const dims = [...quantOut.codes.dims].map(Number);
      nFrames = dims[2];
      nLevels = Math.min(dims[1], 12);
      codes = new Int32Array(codesRaw.length);
      for (let i = 0; i < codesRaw.length; i++) codes[i] = Number(codesRaw[i]);
      setProgress(0.6, 'Computing deltas…');
      curves = encodecDeltas(codes, nLevels, nFrames, metric);
    } else {
      const strides = sessions._strides;
      nLevels = strides.length;
      const codeNames = Array.from({ length: nLevels }, (_, i) => `codes_${i}`);
      const inputs = { z: encOut.z };
      const quantOut = await sessions.quantEnc.run(inputs);
      setProgress(0.6, 'Computing deltas…');
      const upsample = msg.upsample || 'lerp';
      const result = snacDeltas(quantOut, strides, nLevels, metric, upsample);
      curves = result.curves;
      codes = result.codes;
      nFrames = result.nFrames;
    }

    postMessage({
      type: 'result',
      jobId,
      curves,
      nLevels,
      nFrames,
      codes,
      codesDims: [1, nLevels, nFrames],
      sampleRate: sessions._sr,
      hopSize: sessions._hopSize,
    }, [curves.buffer, codes.buffer]);

  } catch (err) {
    postMessage({ type: 'error', jobId, message: err?.message || String(err) });
  }
}

onmessage = ({ data }) => {
  switch (data.type) {
    case 'init':
      if (data.encodecBaseUrl) encodecBaseUrl = data.encodecBaseUrl;
      if (data.snacBaseUrl) snacBaseUrl = data.snacBaseUrl;
      postMessage({ type: 'ready' });
      break;
    case 'encode':
      runEncode(data);
      break;
  }
};
