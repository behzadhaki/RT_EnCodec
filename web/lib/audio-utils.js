export const MAX_S = 10;
export const PEAK_TARGET = 0.75; // loaded sources are peak-normalised to this

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
// Planar multi-channel aware: the gate phase restarts per channel plane so
// L and R stay sample-aligned.
export function applyPulseGate(audio, sr, freq, decayMs, channels = 1) {
  const T      = audio.length / channels;
  const period = Math.max(1, Math.round(sr / Math.max(0.01, freq)));
  const out    = new Float32Array(audio.length);
  for (let ch = 0; ch < channels; ch++) {
    const off = ch * T;
    if (decayMs <= 0) {
      for (let i = 0; i < T; i++)
        out[off + i] = (i % period === 0) ? audio[off + i] : 0;
    } else {
      const tau = (decayMs / 1000) * sr / 5;
      for (let i = 0; i < T; i++)
        out[off + i] = audio[off + i] * Math.exp(-(i % period) / tau);
    }
  }
  return out;
}

// Decodes and resamples an audio File at the given sr.
// Trims to maxS seconds at the source sample rate before resampling.
// channels = 1 → mono Float32Array (all source channels averaged).
// channels = 2 → planar stereo Float32Array(2*T), L plane then R plane
//                (mono sources are upmixed to both planes by Web Audio).
export async function loadFile(file, sr, maxS = MAX_S, channels = 1) {
  const arrBuf  = await file.arrayBuffer();
  const tmpCtx  = new OfflineAudioContext(1, 1, sr);
  const decoded = await tmpCtx.decodeAudioData(arrBuf);
  const trimLen = Math.min(decoded.length, Math.round(maxS * decoded.sampleRate));
  const outLen  = Math.round(trimLen / decoded.sampleRate * sr);
  const offCtx  = new OfflineAudioContext(channels, outLen, sr);
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

  if (channels === 2) {
    const T   = rendered.length;
    const out = new Float32Array(2 * T);
    out.set(rendered.getChannelData(0), 0);
    out.set(rendered.getChannelData(Math.min(1, rendered.numberOfChannels - 1)), T);
    return normalizePeak(out, PEAK_TARGET); // match loaded sources to a common peak
  }

  const mono = new Float32Array(rendered.length);
  for (let ch = 0; ch < rendered.numberOfChannels; ch++) {
    const d = rendered.getChannelData(ch);
    for (let i = 0; i < rendered.length; i++) mono[i] += d[i] / rendered.numberOfChannels;
  }
  return normalizePeak(mono, PEAK_TARGET);
}

export const MAX_FOLDER_FILES = 20;

// Loads every file in `files`, decodes/resamples each to `sr` via loadFile,
// and concatenates them into one source (name-sorted, natural order). At most
// `maxFiles` files are used, and each file is capped at MAX_S seconds (so a
// long clip contributes only its first MAX_S). channels = 2 keeps the planar
// layout [L…|R…], concatenated per-plane so the joined stereo stays valid.
// Unreadable files are skipped with a warning rather than aborting the whole set.
export async function loadFolder(files, sr, maxS = MAX_S, channels = 1, maxFiles = MAX_FOLDER_FILES) {
  if (!files || !files.length) throw new Error('No files in folder.');
  const sorted = [...files].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' })).slice(0, maxFiles);

  const perFileMaxS = Math.min(maxS, MAX_S); // never more than MAX_S per clip
  const bufs = [];
  for (const f of sorted) {
    try { bufs.push(await loadFile(f, sr, perFileMaxS, channels)); }
    catch (e) { console.warn('loadFolder: skipping unreadable file', f.name, e); }
  }
  if (!bufs.length) throw new Error('No decodable audio in folder.');

  if (channels === 2) {
    const Ts     = bufs.map(b => b.length / 2);
    const totalT = Ts.reduce((s, t) => s + t, 0);
    const out    = new Float32Array(2 * totalT);
    let off = 0;
    for (let i = 0; i < bufs.length; i++) {
      const T = Ts[i];
      out.set(bufs[i].subarray(0, T),      off);            // L plane
      out.set(bufs[i].subarray(T, 2 * T),  totalT + off);   // R plane
      off += T;
    }
    return out;
  }

  const totalT = bufs.reduce((s, b) => s + b.length, 0);
  const out = new Float32Array(totalT);
  let off = 0;
  for (const b of bufs) { out.set(b, off); off += b.length; }
  return out;
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
