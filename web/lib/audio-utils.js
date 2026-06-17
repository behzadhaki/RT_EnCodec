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
// Takes the window [startS, startS + maxS) seconds at the source sample rate
// before resampling (startS = 0 → from the beginning; maxS = Infinity → to end).
// channels = 1 → mono Float32Array (all source channels averaged).
// channels = 2 → planar stereo Float32Array(2*T), L plane then R plane
//                (mono sources are upmixed to both planes by Web Audio).
export async function loadFile(file, sr, maxS = MAX_S, channels = 1, startS = 0) {
  const arrBuf  = await file.arrayBuffer();
  const tmpCtx  = new OfflineAudioContext(1, 1, sr);
  const decoded = await tmpCtx.decodeAudioData(arrBuf);
  const start   = Math.min(decoded.length, Math.max(0, Math.round(startS * decoded.sampleRate)));
  const avail   = decoded.length - start;
  const trimLen = Math.max(0, Math.min(avail, Math.round(maxS * decoded.sampleRate)));
  const outLen  = Math.max(1, Math.round(trimLen / decoded.sampleRate * sr));
  const offCtx  = new OfflineAudioContext(channels, outLen, sr);
  const trimBuf = new AudioBuffer({
    numberOfChannels: decoded.numberOfChannels,
    length: Math.max(1, trimLen), sampleRate: decoded.sampleRate,
  });
  for (let ch = 0; ch < decoded.numberOfChannels; ch++)
    trimBuf.copyToChannel(decoded.getChannelData(ch).slice(start, start + trimLen), ch);
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

// Total seconds budget for a joined folder source. When the summed clip
// durations exceed this, callers pick a reduction strategy (see
// planFolderSelections) before concatenating.
export const MAX_FOLDER_TOTAL_S = 600; // 10 minutes

// Reads a WAV file's duration from its header alone — no decode. Walks the RIFF
// chunks within the first 64 KB to find `fmt ` (byteRate) and `data` (size);
// duration = dataBytes / byteRate. Returns null if it can't be determined this
// way (non-WAV, streamed size, or metadata pushing `data` past the probe window)
// so the caller can fall back to a full decode.
export async function wavDurationSeconds(file) {
  const PROBE = 1 << 16; // 64 KB — covers headers + typical LIST/INFO metadata
  let buf;
  try { buf = await file.slice(0, PROBE).arrayBuffer(); }
  catch { return null; }
  const dv = new DataView(buf);
  if (dv.byteLength < 12) return null;
  if (dv.getUint32(0, false) !== 0x52494646) return null; // 'RIFF'
  if (dv.getUint32(8, false) !== 0x57415645) return null; // 'WAVE'
  let off = 12, byteRate = 0, dataSize = 0;
  while (off + 8 <= dv.byteLength) {
    const id   = dv.getUint32(off, false);
    const size = dv.getUint32(off + 4, true);
    const body = off + 8;
    if (id === 0x666d7420 && body + 16 <= dv.byteLength) {       // 'fmt '
      byteRate = dv.getUint32(body + 8, true);
    } else if (id === 0x64617461) {                             // 'data'
      dataSize = size; break;
    }
    off = body + size + (size & 1); // chunks are word-aligned
  }
  if (byteRate > 0 && dataSize > 0 && dataSize !== 0xffffffff) return dataSize / byteRate;
  return null;
}

// Reads an audio File's duration in seconds from container metadata only (no full
// decode) via a media element — cheap for mp3/m4a/mp4/etc. Returns null if the
// duration can't be read this way (streamed/unknown).
function mediaElementDurationSeconds(file) {
  return new Promise((resolve) => {
    let url;
    try { url = URL.createObjectURL(file); } catch { resolve(null); return; }
    const a = new Audio();
    a.preload = 'metadata';
    const done = (v) => { try { URL.revokeObjectURL(url); } catch {} a.removeAttribute('src'); resolve(v); };
    a.onloadedmetadata = () => done(isFinite(a.duration) && a.duration > 0 ? a.duration : null);
    a.onerror = () => done(null);
    a.src = url;
  });
}

// Duration of an audio File in seconds (uncapped). Cheapest path first: WAV header
// → container metadata (mp3/m4a/…) → full decode as a last resort.
export async function probeAudioDurationSeconds(file) {
  const fromHeader = await wavDurationSeconds(file);
  if (fromHeader != null && isFinite(fromHeader) && fromHeader > 0) return fromHeader;
  const fromMeta = await mediaElementDurationSeconds(file);
  if (fromMeta != null) return fromMeta;
  try {
    const arrBuf  = await file.arrayBuffer();
    const tmpCtx  = new OfflineAudioContext(1, 1, 44100);
    const decoded = await tmpCtx.decodeAudioData(arrBuf);
    return decoded.duration;
  } catch { return 0; }
}

// Builds a list of per-file selections {file, startS, lenS} for a folder source,
// reducing the set so the joined total fits `budgetS`. `durations[i]` is the full
// length (s) of `files[i]` (from probeAudioDurationSeconds). Strategies:
//   'all'        — every clip whole (no reduction; over-budget loads anyway)
//   'first'      — keep clips whole in name order until the budget runs out
//   'trim'       — first  budgetS/N s of every clip (equal share)
//   'random'     — a random budgetS/N s window of every clip (equal share)
//   'random:K'   — a random K-second window of every clip (whole clip if shorter)
//   'truncate'   — concatenate in order, hard-cut at exactly budgetS
//
// Random offsets are deterministic per clip (seeded from salt+name+size+index), so
// the same folder reproduces the same windows across reloads — which lets persisted
// per-clip code sidecars be reused instead of re-encoded. The `salt` distinguishes
// sources (e.g. 'A'/'B') so the SAME folder loaded into both gets different random
// windows rather than identical ones. Pass `rand` to override (e.g. in tests).
function clipSeedFrac(file, i, salt = '') {
  const str = `${salt}:${file?.name ?? ''}:${file?.size ?? 0}:${i}`;
  let h = 2166136261 >>> 0;                 // FNV-1a
  for (let k = 0; k < str.length; k++) { h ^= str.charCodeAt(k); h = Math.imul(h, 16777619) >>> 0; }
  return h / 4294967296;                     // [0, 1)
}

export function planFolderSelections(files, durations, strategy, budgetS = MAX_FOLDER_TOTAL_S, salt = '', rand = null) {
  const N = files.length;
  const sel = [];
  if (strategy === 'all') {
    for (let i = 0; i < N; i++) sel.push({ file: files[i], startS: 0, lenS: Infinity });
  } else if (strategy === 'first') {
    let acc = 0;
    for (let i = 0; i < N; i++) {
      if (acc + durations[i] > budgetS) break;
      sel.push({ file: files[i], startS: 0, lenS: Infinity });
      acc += durations[i];
    }
    if (!sel.length) sel.push({ file: files[0], startS: 0, lenS: budgetS }); // 1st clip alone over budget
  } else if (strategy === 'trim' || strategy === 'random' || strategy.startsWith('random:')) {
    // 'random:K' → fixed K-second window; 'trim'/'random' → equal budgetS/N share.
    const fixed   = strategy.startsWith('random:') ? parseFloat(strategy.slice(7)) : null;
    const k       = fixed != null ? fixed : budgetS / N;
    const isRandom = strategy === 'random' || fixed != null;
    for (let i = 0; i < N; i++) {
      const win   = Math.min(durations[i], k);
      const slack = Math.max(0, durations[i] - win);
      const frac  = rand ? rand(i) : clipSeedFrac(files[i], i, salt);
      const startS = (isRandom && slack > 0) ? frac * slack : 0;
      sel.push({ file: files[i], startS, lenS: win });
    }
  } else { // 'truncate'
    let acc = 0;
    for (let i = 0; i < N && acc < budgetS; i++) {
      const lenS = Math.min(durations[i], budgetS - acc);
      sel.push({ file: files[i], startS: 0, lenS });
      acc += lenS;
    }
  }
  return sel;
}

// Loads a folder source and concatenates it into one joined buffer (name order).
// `selections` is a [{file, startS, lenS}] list (see planFolderSelections); when
// omitted, every file is loaded whole. channels = 2 keeps the planar layout
// [L…|R…], concatenated per-plane so the joined stereo stays valid. Unreadable
// files are skipped with a warning rather than aborting the whole set.
export async function loadFolder(files, sr, channels = 1, selections = null) {
  const sels = selections
    || (files || []).map(f => ({ file: f, startS: 0, lenS: Infinity }));
  if (!sels.length) throw new Error('No files in folder.');

  const bufs = [];
  for (const s of sels) {
    try { bufs.push(await loadFile(s.file, sr, s.lenS ?? Infinity, channels, s.startS ?? 0)); }
    catch (e) { console.warn('loadFolder: skipping unreadable file', s.file?.name, e); }
  }
  if (!bufs.length) throw new Error('No decodable audio in folder.');

  // clipStarts: per-channel sample offset where each clip begins (so callers can
  // mark per-clip boundaries on the joined source, e.g. per-clip wheel time).
  if (channels === 2) {
    const Ts     = bufs.map(b => b.length / 2);
    const totalT = Ts.reduce((s, t) => s + t, 0);
    const out    = new Float32Array(2 * totalT);
    const clipStarts = [];
    let off = 0;
    for (let i = 0; i < bufs.length; i++) {
      const T = Ts[i];
      clipStarts.push(off);
      out.set(bufs[i].subarray(0, T),      off);            // L plane
      out.set(bufs[i].subarray(T, 2 * T),  totalT + off);   // R plane
      off += T;
    }
    out.clipStarts = clipStarts;
    return out;
  }

  const totalT = bufs.reduce((s, b) => s + b.length, 0);
  const out = new Float32Array(totalT);
  const clipStarts = [];
  let off = 0;
  for (const b of bufs) { clipStarts.push(off); out.set(b, off); off += b.length; }
  out.clipStarts = clipStarts;
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
