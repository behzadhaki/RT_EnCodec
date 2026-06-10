/**
 * encodec-cache.js — IndexedDB cache for experiment6 per-source analysis results.
 *
 * One record per (file identity + model params + segment).
 * File identity = filename + fileSize + CRC-32 of raw file bytes.
 * Model params  = modelHz + bwKbps + sampleRate.
 * Segment       = segStart (always 0) + segEnd (number of resampled samples loaded).
 *
 * Stored arrays per record:
 *   embEnc   (Float32Array) — pre-VQ encoder embeddings
 *   embQuant (Float32Array) — post-VQ quantised embeddings
 *   codes    (Int32Array)   — VQ codebook indices
 *   scales   (Float32Array | null) — per-frame scales (24 kHz model)
 *
 * Storing embQuant + codes alongside embEnc means cache hits require zero
 * additional inference — the full `currentEmbeds` object can be assembled
 * directly from cached data for both sources.
 */

const DB_NAME    = 'encodec-cache';
const DB_VERSION = 1;
const STORE      = 'analyses';

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
      const db    = e.target.result;
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
 * {
 *   embEnc   (Float32Array), embEncDims  (number[]),
 *   embQuant (Float32Array), embQuantDims (number[]),
 *   codes    (Int32Array),   codesDims   (number[]),
 *   scales   (Float32Array | null),
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
    req.onsuccess = e => {
      const rec = e.target.result;
      if (!rec) { resolve(null); return; }
      resolve({
        embEnc:    new Float32Array(rec.embEncData),
        embEncDims: rec.embEncDims,
        embQuant:  new Float32Array(rec.embQuantData),
        embQuantDims: rec.embQuantDims,
        codes:     new Int32Array(rec.codesData),
        codesDims: rec.codesDims,
        scales:    rec.scalesData ? new Float32Array(rec.scalesData) : null,
      });
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
      const approxBytes =
        (r.embEncData?.byteLength   ?? 0) +
        (r.embQuantData?.byteLength ?? 0) +
        (r.codesData?.byteLength    ?? 0) +
        (r.scalesData?.byteLength   ?? 0);
      items.push({
        id: cursor.primaryKey,
        filename: r.filename, fileSize: r.fileSize,
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
 * @param {IDBDatabase} db
 * @param {object}      key  — from makeCacheKey
 * @param {object}      emb  — {
 *   embEnc (Float32Array), embEncDims,
 *   embQuant (Float32Array), embQuantDims,
 *   codes (Int32Array), codesDims,
 *   scales (Float32Array | null),
 * }
 */
export function cachePut(db, key, emb) {
  return new Promise((resolve, reject) => {
    const tx    = db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);

    // Delete any existing record with same compound key first (put via index
    // isn't supported directly — use the index to find the auto-increment key
    // then delete + add).
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
        embEncData:   emb.embEnc.buffer.slice(0),
        embEncDims:   emb.embEncDims,
        embQuantData: emb.embQuant.buffer.slice(0),
        embQuantDims: emb.embQuantDims,
        codesData:    emb.codes.buffer.slice(0),
        codesDims:    emb.codesDims,
        scalesData:   emb.scales ? emb.scales.buffer.slice(0) : null,
        timestamp:    Date.now(),
      };

      const addReq = store.add(record);
      addReq.onsuccess = () => resolve();
      addReq.onerror   = e2 => reject(e2.target.error);
    };
    findReq.onerror = e => reject(e.target.error);
  });
}
