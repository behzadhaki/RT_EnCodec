/**
 * snac-sidecar.js — file cache for SNAC codes, the multi-scale analogue of
 * encodec-sidecar.js. Stored next to a folder source so a re-load with the same
 * model skips re-encoding.
 *
 * SNAC codes are N ragged levels (different rates), so unlike the EnCodec
 * sidecar (single [1,K,T]) this stores each level's codes with its own length.
 * The SNAC model variant is tracked in BOTH the filename and the header.
 *
 * One cache file per (source-audio × model), keyed by the CRC-32 of the exact
 * (folder-concatenated, window-sliced) audio buffer — so the same folder +
 * selections + model reproduces the key and hits the cache; a different model,
 * folder, or window misses.
 *
 *   filename:  .snaccache.<model>.<crc8>        e.g.  .snaccache.24khz.1a2b3c4d
 *
 * Binary layout (little-endian):
 *   "SNACSID1"            8-byte magic
 *   uint32 headerLen
 *   header JSON (UTF-8)
 *   payload              per-level Int16 codes, concatenated in level order
 *
 * header = { v, model, crc32, sampleRate, frameRate, vqStrides,
 *            levels: [{ len, off, bytes }] }
 */

import { crc32 } from './encodec-sidecar.js';

export { crc32 };

const MAGIC = 'SNACSID1';
const u8e = s => new TextEncoder().encode(s);
const u8d = b => new TextDecoder().decode(b);

/** CRC-32 of a typed array's exact bytes (robust to subarray views). */
export function crcOfAudio(f32) {
  return crc32(new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength).slice().buffer);
}

export function snacCacheName(model, crc) {
  return `.snaccache.${model}.${(crc >>> 0).toString(16).padStart(8, '0')}`;
}

/** meta: { model, crc32, sampleRate, frameRate, vqStrides }; levels: Int32Array[] */
export function serializeSnacCache(meta, levels) {
  const chunks = [];
  let off = 0;
  const levelInfo = levels.map(lv => {
    const i16 = Int16Array.from(lv, v => v | 0);
    const o = off; chunks.push(i16.buffer); off += i16.byteLength;
    return { len: lv.length, off: o, bytes: i16.byteLength };
  });
  const header = {
    v: 1, model: meta.model, crc32: meta.crc32,
    sampleRate: meta.sampleRate, frameRate: meta.frameRate,
    vqStrides: meta.vqStrides, levels: levelInfo,
  };
  const hb = u8e(JSON.stringify(header)), mb = u8e(MAGIC);
  const out = new Uint8Array(mb.length + 4 + hb.length + off);
  const dv = new DataView(out.buffer);
  let p = 0;
  out.set(mb, p); p += mb.length;
  dv.setUint32(p, hb.length, true); p += 4;
  out.set(hb, p); p += hb.length;
  for (const c of chunks) { out.set(new Uint8Array(c), p); p += c.byteLength; }
  return out.buffer;
}

/** Returns { header, levels: Int32Array[] } or null. */
export function parseSnacCache(ab) {
  try {
    const u8 = new Uint8Array(ab);
    if (u8.length < 12 || u8d(u8.subarray(0, 8)) !== MAGIC) return null;
    const dv = new DataView(ab);
    const hl = dv.getUint32(8, true), hs = 12, ps = hs + hl;
    if (ps > u8.length) return null;
    const header = JSON.parse(u8d(u8.subarray(hs, ps)));
    const levels = header.levels.map(li => {
      const s = ps + li.off;
      const i16 = new Int16Array(ab.slice(s, s + li.bytes));
      const out = new Int32Array(i16.length);
      for (let i = 0; i < i16.length; i++) out[i] = i16[i];
      return out;
    });
    return { header, levels };
  } catch { return null; }
}

export function snacCacheMatches(header, { model, crc32, sampleRate }) {
  return !!header && header.model === model
    && header.crc32 === crc32 && header.sampleRate === sampleRate;
}

export async function readSnacCache(dirHandle, model, crc) {
  if (!dirHandle) return null;
  try {
    const fh = await dirHandle.getFileHandle(snacCacheName(model, crc));
    return parseSnacCache(await (await fh.getFile()).arrayBuffer());
  } catch { return null; }
}

export async function writeSnacCache(dirHandle, model, crc, arrayBuffer) {
  const fh = await dirHandle.getFileHandle(snacCacheName(model, crc), { create: true });
  const w = await fh.createWritable();
  await w.write(arrayBuffer);
  await w.close();
}
