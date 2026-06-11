/**
 * encodec-cache.js — IndexedDB cache for experiment6 per-source analysis results.
 *
 * One record per (file identity + model params + segment).
 * File identity = filename + fileSize + CRC-32 of raw file bytes.
 * Model params  = modelHz + bwKbps + sampleRate.
 * Segment       = segStart (always 0) + segEnd (number of resampled samples loaded).
 *
 * Stored arrays per record (v2 format — all buffers gzip-compressed):
 *   codes   (Int16Array,  gzip'd)         — VQ codebook indices (max 1023 → fits Int16)
 *   scales  (Float32Array, gzip'd | null) — per-frame scales (24 kHz model)
 *
 * embEnc (pre-VQ) is NOT stored — it is only needed for the Pre-VQ UMAP tab and
 * can be re-computed on demand from rawAudioA/B when the user requests that view.
 *
 * embQuant is NOT stored.  At cache-load time the caller reconstructs it by
 * summing codebook vectors: embQuant[d,t] = Σ_q codebook[q, codes[q,t], d].
 * This is exact (same arithmetic as the quantiser) and fast (~2 ms for 10 s).
 *
 * DB_VERSION 2 — v1 store is dropped on upgrade (clean-slate migration).
 */

const DB_NAME    = 'encodec-cache';
const DB_VERSION = 2;
const STORE      = 'analyses';

// ── Compression helpers ────────────────────────────────────────────────────────

/** Gzip-compress an ArrayBuffer; returns a new ArrayBuffer. */
async function compress(arrayBuffer) {
  const cs     = new CompressionStream('gzip');
  const writer = cs.writable.getWriter();
  writer.write(new Uint8Array(arrayBuffer));
  writer.close();
  return new Response(cs.readable).arrayBuffer();
}

/** Gzip-decompress an ArrayBuffer; returns a new ArrayBuffer. */
async function decompress(arrayBuffer) {
  const ds     = new DecompressionStream('gzip');
  const writer = ds.writable.getWriter();
  writer.write(new Uint8Array(arrayBuffer));
  writer.close();
  return new Response(ds.readable).arrayBuffer();
}

// ── CRC-32 ────────────────────────────────────────────────────────────────────

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
  const buf = await file.arrayBuffer();
  const val = crc32(buf);
  _crcCache.set(file, val);
  return val;
}

// ── IndexedDB ─────────────────────────────────────────────────────────────────

export function openEncodecDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);

    req.onupgradeneeded = e => {
      const db = e.target.result;
      // v1 → v2: format changed (gzip + Int16 codes + embQuant dropped).
      // Existing entries are incompatible — drop the old store for a clean slate.
      if (e.oldVersion < 2 && db.objectStoreNames.contains(STORE)) {
        db.deleteObjectStore(STORE);
      }
      const store = db.createObjectStore(STORE, { autoIncrement: true });
      // Compound unique index — the natural lookup key.
      store.createIndex(
        'lookup',
        ['filename', 'fileSize', 'crc32', 'modelHz', 'bwKbps', 'sampleRate', 'segStart', 'segEnd'],
        { unique: true },
      );
    };

    req.onsuccess = e => resolve(e.target.result);
    req.onerror   = e => reject(e.target.error);
  });
}

/**
 * Build a cache key object.
 * @param {File}   file       — source File object
 * @param {number} crc        — CRC-32 of file bytes (from crc32FromFile)
 * @param {object} modelOpts  — { modelHz, bwKbps, sampleRate }
 * @param {number} segEnd     — number of samples actually loaded (at sampleRate)
 */
export function makeCacheKey(file, crc, { modelHz, bwKbps, sampleRate }, segEnd) {
  return {
    filename:   file.name,
    fileSize:   file.size,
    crc32:      crc,
    modelHz,
    bwKbps,
    sampleRate,
    segStart:   0,
    segEnd,
  };
}

/**
 * Look up a cached entry by key.
 * Returns the stored embeddings object or null if not found.
 *
 * NOTE: embEnc / embEncDims are intentionally absent — re-computed on demand.
 * NOTE: embQuant / embQuantDims are intentionally absent — reconstruct from codes.
 *
 * {
 *   codes   (Int32Array),   codesDims  (number[]),
 *   scales  (Float32Array | null),
 * }
 */
export function cacheGet(db, key) {
  return new Promise((resolve, reject) => {
    const tx  = db.transaction(STORE, 'readonly');
    const idx = tx.objectStore(STORE).index('lookup');
    const req = idx.get([
      key.filename, key.fileSize, key.crc32,
      key.modelHz, key.bwKbps, key.sampleRate,
      key.segStart, key.segEnd,
    ]);
    req.onsuccess = async e => {
      const rec = e.target.result;
      if (!rec) { resolve(null); return; }
      try {
        // Decompress all stored buffers in parallel.
        const [codesBuf, scalesBuf] = await Promise.all([
          decompress(rec.codesData),
          rec.scalesData ? decompress(rec.scalesData) : Promise.resolve(null),
        ]);
        // Codes were stored as Int16 (values 0–1023) — upcast to Int32 for use.
        const codesI16 = new Int16Array(codesBuf);
        const codes    = new Int32Array(codesI16.length);
        for (let i = 0; i < codesI16.length; i++) codes[i] = codesI16[i];
        resolve({
          codes,
          codesDims: rec.codesDims,
          scales:    scalesBuf ? new Float32Array(scalesBuf) : null,
        });
      } catch (err) {
        reject(err);
      }
    };
    req.onerror = e => reject(e.target.error);
  });
}

/**
 * Return a summary of all stored entries (no large buffers, just metadata).
 * Each item: { id, filename, fileSize, modelHz, bwKbps, sampleRate,
 *              segStart, segEnd, timestamp, approxBytes }
 */
export function cacheList(db) {
  return new Promise((resolve, reject) => {
    const tx    = db.transaction(STORE, 'readonly');
    const store = tx.objectStore(STORE);
    const req   = store.openCursor();
    const items = [];
    req.onsuccess = e => {
      const cursor = e.target.result;
      if (!cursor) { resolve(items); return; }
      const r = cursor.value;
      // approxBytes = compressed storage footprint (codes + scales only in v2).
      const approxBytes =
        (r.codesData?.byteLength  ?? 0) +
        (r.scalesData?.byteLength ?? 0);
      items.push({
        id: cursor.primaryKey,
        filename: r.filename, fileSize: r.fileSize, crc32: r.crc32,
        modelHz: r.modelHz, bwKbps: r.bwKbps, sampleRate: r.sampleRate,
        segStart: r.segStart, segEnd: r.segEnd,
        timestamp: r.timestamp,
        approxBytes,
      });
      cursor.continue();
    };
    req.onerror = e => reject(e.target.error);
  });
}

/**
 * Retrieve a cache entry directly by its auto-increment primary key.
 * Identical decompression / Int16→Int32 upcast as cacheGet.
 * Returns { codes, codesDims, scales } or null if not found.
 */
export function cacheGetById(db, id) {
  return new Promise((resolve, reject) => {
    const tx  = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).get(id);
    req.onsuccess = async e => {
      const rec = e.target.result;
      if (!rec) { resolve(null); return; }
      try {
        const [codesBuf, scalesBuf] = await Promise.all([
          decompress(rec.codesData),
          rec.scalesData ? decompress(rec.scalesData) : Promise.resolve(null),
        ]);
        const codesI16 = new Int16Array(codesBuf);
        const codes    = new Int32Array(codesI16.length);
        for (let i = 0; i < codesI16.length; i++) codes[i] = codesI16[i];
        resolve({
          codes,
          codesDims: rec.codesDims,
          scales:    scalesBuf ? new Float32Array(scalesBuf) : null,
        });
      } catch (err) {
        reject(err);
      }
    };
    req.onerror = e => reject(e.target.error);
  });
}

/**
 * Delete a single cache entry by its auto-increment primary key.
 */
export function cacheDelete(db, id) {
  return new Promise((resolve, reject) => {
    const tx  = db.transaction(STORE, 'readwrite');
    const req = tx.objectStore(STORE).delete(id);
    req.onsuccess = () => resolve();
    req.onerror   = e => reject(e.target.error);
  });
}

/**
 * Delete all cache entries.
 */
export function cacheClear(db) {
  return new Promise((resolve, reject) => {
    const tx  = db.transaction(STORE, 'readwrite');
    const req = tx.objectStore(STORE).clear();
    req.onsuccess = () => resolve();
    req.onerror   = e => reject(e.target.error);
  });
}

/**
 * Save or overwrite a cache entry.
 *
 * Compression and Int16 downcast happen before opening the IDB transaction so
 * no async work stalls mid-transaction (which would cause auto-commit before
 * the write lands).
 *
 * @param {IDBDatabase} db
 * @param {object}      key  — from makeCacheKey
 * @param {object}      emb  — {
 *   codes  (Int32Array), codesDims,
 *   scales (Float32Array | null),
 * }
 * NOTE: embEnc  is not stored — re-computed on demand for the pre-VQ tab.
 * NOTE: embQuant is not stored — reconstructed from codes + codebooks on load.
 */
export async function cachePut(db, key, emb) {
  // ── Step 1: downcast codes Int32 → Int16 ──────────────────────────────────
  // All codebook indices are 0–1023 (vocab_size 1024), safely fitting Int16.
  const codesI16 = new Int16Array(emb.codes.length);
  for (let i = 0; i < emb.codes.length; i++) codesI16[i] = emb.codes[i];

  // ── Step 2: gzip-compress buffers in parallel ─────────────────────────────
  const [codesComp, scalesComp] = await Promise.all([
    compress(codesI16.buffer),
    emb.scales ? compress(emb.scales.buffer) : Promise.resolve(null),
  ]);

  // ── Step 3: write to IDB — no async after transaction open ────────────────
  return new Promise((resolve, reject) => {
    const tx    = db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);

    // Delete any existing record with the same compound key first (put via
    // index isn't directly supported — find the auto-increment key then delete + add).
    const idx     = store.index('lookup');
    const findReq = idx.getKey([
      key.filename, key.fileSize, key.crc32,
      key.modelHz, key.bwKbps, key.sampleRate,
      key.segStart, key.segEnd,
    ]);

    findReq.onsuccess = e => {
      if (e.target.result !== undefined) store.delete(e.target.result);

      const record = {
        ...key,
        // embEnc omitted — re-computed on demand for pre-VQ tab
        // embQuant omitted — reconstructed from codes + codebooks on load
        codesData:  codesComp,
        codesDims:  emb.codesDims,
        scalesData: scalesComp,
        timestamp:  Date.now(),
      };

      const addReq = store.add(record);
      addReq.onsuccess = () => resolve();
      addReq.onerror   = e2 => reject(e2.target.error);
    };
    findReq.onerror = e => reject(e.target.error);
  });
}
