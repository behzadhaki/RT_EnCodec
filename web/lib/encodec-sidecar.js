/**
 * encodec-sidecar.js — portable, model-versioned EnCodec codes stored as a file
 * next to the source .wav (so a re-load skips re-encoding, and the codes can be
 * consumed by other tools, e.g. a Max/MSP patch).
 *
 * Only folder sources get sidecars: the directory picker yields a writable
 * FileSystemDirectoryHandle, so we can create a sibling file. A single-file
 * picker yields only a file handle (no parent access), so those fall back to the
 * IndexedDB cache.
 *
 * One sidecar per (wav × model × bandwidth):
 *     <wavName>.encodec.<hz>.<bw>      e.g.  drum.wav.encodec.48.24
 * where <hz> is the model's kHz (24 or 48) and <bw> the bandwidth in kbps.
 *
 * A sidecar holds one or more *window entries* — codes for a [startS, lenS)
 * slice of the clip — keyed by "<startMs>-<lenMs>". Picking a new random window
 * adds an entry; an already-seen window is read back instead of re-encoded.
 *
 * Binary layout (all integers little-endian):
 *     "ENCSIDE1"                       8-byte magic
 *     uint32 headerLen
 *     header JSON (UTF-8, headerLen bytes)   — see HeaderShape below
 *     payload                          concatenated per-entry buffers:
 *                                      [Int16 codes][Float32 scales?] …
 *
 * header = {
 *   v: 1,
 *   filename, fileSize, crc32,         // identity of the source wav (validated on read)
 *   modelHz, bwKbps, sampleRate, channels,
 *   entries: [{
 *     key,                             // "<startMs>-<lenMs>"
 *     codesDims,                       // e.g. [1, nCodebooks, nFrames]
 *     scales: bool,
 *     codesOff, codesLen,              // byte range in payload (Int16 count = codesLen/2)
 *     scalesOff, scalesLen,            // byte range in payload (Float32 count = scalesLen/4)
 *   }],
 * }
 *
 * Codes are 0–1023 (vocab 1024) → stored Int16, upcast to Int32 on read to match
 * the rest of the pipeline.
 */

const MAGIC = 'ENCSIDE1';

// ── CRC-32 (identity check for the source wav) ───────────────────────────────

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    t[i] = c;
  }
  return t;
})();

export function crc32(buf) {
  const bytes = new Uint8Array(buf instanceof ArrayBuffer ? buf : buf.buffer);
  let c = 0xFFFFFFFF;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}

// WeakMap so CRC is computed at most once per File object per page lifetime.
const _crcCache = new WeakMap();

export async function crc32FromFile(file) {
  if (_crcCache.has(file)) return _crcCache.get(file);
  const val = crc32(await file.arrayBuffer());
  _crcCache.set(file, val);
  return val;
}

/** kHz digits for a modelHz string ('48k' → 48, '24k' → 24). */
function hzDigits(modelHz) {
  const n = parseInt(modelHz, 10);
  return Number.isFinite(n) ? n : modelHz;
}

/** Sidecar filename for a source file under a given model + bandwidth. */
export function sidecarFileName(fileName, modelHz, bwKbps) {
  return `${fileName}.encodec.${hzDigits(modelHz)}.${bwKbps}`;
}

/** Stable key for a [startS, lenS) window (ms-rounded so floats don't drift). */
export function windowKey(startS, lenS) {
  const ms = s => Math.round((Number.isFinite(s) ? s : 0) * 1000);
  return `${ms(startS)}-${Number.isFinite(lenS) ? ms(lenS) : 'full'}`;
}

// ── Serialization ───────────────────────────────────────────────────────────

function utf8Encode(str) { return new TextEncoder().encode(str); }
function utf8Decode(buf) { return new TextDecoder().decode(buf); }

/**
 * Serialize a full sidecar from metadata + a map of window entries.
 * @param meta    { filename, fileSize, crc32, modelHz, bwKbps, sampleRate, channels }
 * @param entries Map|object  key → { codes (Int16Array|Int32Array), codesDims, scales (Float32Array|null) }
 * @returns ArrayBuffer
 */
export function serializeSidecar(meta, entries) {
  const list = entries instanceof Map ? [...entries.entries()] : Object.entries(entries);

  // Lay out payload buffers and record byte ranges.
  const chunks = [];          // ArrayBuffer pieces in payload order
  let off = 0;
  const headerEntries = list.map(([key, e]) => {
    const codesI16 = e.codes instanceof Int16Array
      ? e.codes
      : Int16Array.from(e.codes, v => v | 0);
    const codesBuf = codesI16.buffer.slice(codesI16.byteOffset, codesI16.byteOffset + codesI16.byteLength);
    const codesOff = off; chunks.push(codesBuf); off += codesBuf.byteLength;

    let scalesOff = 0, scalesLen = 0;
    if (e.scales && e.scales.length) {
      const f32 = e.scales instanceof Float32Array ? e.scales : Float32Array.from(e.scales);
      const sBuf = f32.buffer.slice(f32.byteOffset, f32.byteOffset + f32.byteLength);
      scalesOff = off; scalesLen = sBuf.byteLength; chunks.push(sBuf); off += sBuf.byteLength;
    }
    return {
      key,
      codesDims: e.codesDims,
      scales: scalesLen > 0,
      codesOff, codesLen: codesBuf.byteLength,
      scalesOff, scalesLen,
    };
  });

  const header = {
    v: 1,
    filename: meta.filename, fileSize: meta.fileSize, crc32: meta.crc32,
    modelHz: meta.modelHz, bwKbps: meta.bwKbps,
    sampleRate: meta.sampleRate, channels: meta.channels,
    ola: meta.ola ?? null,            // 48k OLA mode ('stateful'|'stateless') — null for 24k
    entries: headerEntries,
  };
  const headerBytes = utf8Encode(JSON.stringify(header));

  const magicBytes = utf8Encode(MAGIC);            // 8 bytes
  const total = magicBytes.length + 4 + headerBytes.length + off;
  const out = new Uint8Array(total);
  const dv  = new DataView(out.buffer);
  let p = 0;
  out.set(magicBytes, p); p += magicBytes.length;
  dv.setUint32(p, headerBytes.length, true); p += 4;
  out.set(headerBytes, p); p += headerBytes.length;
  for (const c of chunks) { out.set(new Uint8Array(c), p); p += c.byteLength; }
  return out.buffer;
}

/**
 * Parse a sidecar ArrayBuffer. Returns { header, entries } where entries is a
 * Map key → { codes (Int32Array), codesDims, scales (Float32Array|null) }, or
 * null if the buffer isn't a valid sidecar.
 */
export function parseSidecar(arrayBuffer) {
  try {
    const u8 = new Uint8Array(arrayBuffer);
    if (u8.length < 12) return null;
    if (utf8Decode(u8.subarray(0, 8)) !== MAGIC) return null;
    const dv = new DataView(arrayBuffer);
    const headerLen = dv.getUint32(8, true);
    const headerStart = 12;
    const payloadStart = headerStart + headerLen;
    if (payloadStart > u8.length) return null;
    const header = JSON.parse(utf8Decode(u8.subarray(headerStart, payloadStart)));

    const entries = new Map();
    for (const e of header.entries || []) {
      const cStart = payloadStart + e.codesOff;
      const codesI16 = new Int16Array(arrayBuffer.slice(cStart, cStart + e.codesLen));
      const codes = new Int32Array(codesI16.length);
      for (let i = 0; i < codesI16.length; i++) codes[i] = codesI16[i];
      let scales = null;
      if (e.scales && e.scalesLen > 0) {
        const sStart = payloadStart + e.scalesOff;
        scales = new Float32Array(arrayBuffer.slice(sStart, sStart + e.scalesLen));
      }
      entries.set(e.key, { codes, codesDims: e.codesDims, scales });
    }
    return { header, entries };
  } catch {
    return null;
  }
}

/** True if a parsed sidecar header matches the current file + model params. */
export function sidecarMatches(header, { fileSize, crc32, modelHz, bwKbps, sampleRate, channels, ola = null }) {
  return !!header
    && header.fileSize === fileSize
    && header.crc32 === crc32
    && header.modelHz === modelHz
    && header.bwKbps === bwKbps
    && header.sampleRate === sampleRate
    && header.channels === channels
    && (header.ola ?? null) === (ola ?? null);
}

// ── File System Access helpers (Chromium) ────────────────────────────────────

/** Request read/write permission on a handle; returns true if granted. */
export async function ensureRWPermission(handle) {
  if (!handle || !handle.queryPermission) return false;
  const opts = { mode: 'readwrite' };
  if (await handle.queryPermission(opts) === 'granted') return true;
  try { return await handle.requestPermission(opts) === 'granted'; }
  catch { return false; }
}

/** Read + parse a sidecar from a directory handle. Returns parsed result or null. */
export async function readSidecar(dirHandle, fileName, modelHz, bwKbps) {
  if (!dirHandle) return null;
  const name = sidecarFileName(fileName, modelHz, bwKbps);
  try {
    const fh = await dirHandle.getFileHandle(name);   // throws if missing
    const file = await fh.getFile();
    return parseSidecar(await file.arrayBuffer());
  } catch {
    return null;   // not found / unreadable
  }
}

/** Write a sidecar buffer next to its wav. Requires rw permission. Throws on failure. */
export async function writeSidecar(dirHandle, fileName, modelHz, bwKbps, arrayBuffer) {
  const name = sidecarFileName(fileName, modelHz, bwKbps);
  const fh = await dirHandle.getFileHandle(name, { create: true });
  const w  = await fh.createWritable();
  await w.write(arrayBuffer);
  await w.close();
}
