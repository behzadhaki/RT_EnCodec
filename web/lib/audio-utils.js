export const MAX_S = 10;

// Returns a raw (un-normalized) Float32Array for the given synth type.
export function generateSource(type, freq, dur, sr) {
  const n   = Math.round(dur * sr);
  const buf = new Float32Array(n);
  const τ   = 2 * Math.PI;

  switch (type) {
    case 'sine':
      for (let i = 0; i < n; i++) buf[i] = Math.sin(τ * freq * i / sr);
      break;
    case 'triangle':
      for (let i = 0; i < n; i++) buf[i] = (2 / Math.PI) * Math.asin(Math.sin(τ * freq * i / sr));
      break;
    case 'sawtooth':
      for (let i = 0; i < n; i++) buf[i] = 2 * ((freq * i / sr) % 1) - 1;
      break;
    case 'square':
      for (let i = 0; i < n; i++) buf[i] = Math.sign(Math.sin(τ * freq * i / sr));
      break;
    case 'sweep': {
      const f0 = 20, f1 = sr / 2 * 0.9;
      for (let i = 0; i < n; i++) {
        const t = i / sr;
        buf[i] = Math.sin(τ * (f0 * t + (f1 - f0) * t * t / (2 * dur)));
      }
      break;
    }
    case 'silence':
      break;
  }
  return buf;
}

// Applies a repeating exponential-decay amplitude gate.
export function applyPulseGate(audio, sr, freq, decayMs) {
  const period = Math.max(1, Math.round(sr / Math.max(0.01, freq)));
  const out    = new Float32Array(audio.length);
  if (decayMs <= 0) {
    for (let i = 0; i < audio.length; i++)
      out[i] = (i % period === 0) ? audio[i] : 0;
  } else {
    const tau = (decayMs / 1000) * sr / 5;
    for (let i = 0; i < audio.length; i++)
      out[i] = audio[i] * Math.exp(-(i % period) / tau);
  }
  return out;
}

// Decodes and resamples an audio File to mono Float32Array at the given sr.
// Trims to maxS seconds at the source sample rate before resampling.
export async function loadFile(file, sr, maxS = MAX_S) {
  const arrBuf  = await file.arrayBuffer();
  const tmpCtx  = new OfflineAudioContext(1, 1, sr);
  const decoded = await tmpCtx.decodeAudioData(arrBuf);
  const trimLen = Math.min(decoded.length, Math.round(maxS * decoded.sampleRate));
  const outLen  = Math.round(trimLen / decoded.sampleRate * sr);
  const offCtx  = new OfflineAudioContext(1, outLen, sr);
  const trimBuf = new AudioBuffer({
    numberOfChannels: decoded.numberOfChannels,
    length: trimLen, sampleRate: decoded.sampleRate,
  });
  for (let ch = 0; ch < decoded.numberOfChannels; ch++)
    trimBuf.copyToChannel(decoded.getChannelData(ch).slice(0, trimLen), ch);
  const src = offCtx.createBufferSource();
  src.buffer = trimBuf;
  src.connect(offCtx.destination);
  src.start(0);
  const rendered = await offCtx.startRendering();
  const mono = new Float32Array(rendered.length);
  for (let ch = 0; ch < rendered.numberOfChannels; ch++) {
    const d = rendered.getChannelData(ch);
    for (let i = 0; i < rendered.length; i++) mono[i] += d[i] / rendered.numberOfChannels;
  }
  return mono;
}

// Returns the duration of an audio File in seconds, capped at MAX_S.
export async function getAudioFileDuration(file) {
  const arrBuf  = await file.arrayBuffer();
  const tmpCtx  = new OfflineAudioContext(1, 1, 44100);
  const decoded = await tmpCtx.decodeAudioData(arrBuf);
  return Math.min(decoded.duration, MAX_S);
}

// Scales audio so its peak magnitude equals `target`. Returns a new array.
export function normalizePeak(audio, target = 0.8) {
  let peak = 0;
  for (let i = 0; i < audio.length; i++) { const a = Math.abs(audio[i]); if (a > peak) peak = a; }
  if (peak < 1e-9) return audio;
  const scale = target / peak;
  const out   = new Float32Array(audio.length);
  for (let i = 0; i < audio.length; i++) out[i] = audio[i] * scale;
  return out;
}

// Loops `audio` to exactly `targetLen` samples (wraps around if shorter).
export function loopToLength(audio, targetLen) {
  if (audio.length >= targetLen) return audio.slice(0, targetLen);
  const out = new Float32Array(targetLen);
  for (let i = 0; i < targetLen; i++) out[i] = audio[i % audio.length];
  return out;
}
