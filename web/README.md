# EnCodec Web Experiments

Browser-based demos using the exported ONNX models via `onnxruntime-web`.

---

## Running locally

Serve from the **repo root** (so the ONNX model paths resolve correctly):

```bash
python -m http.server 8000
# then open http://localhost:8000/web/experiment1.html
```

The models are loaded from `serialization/encodec_onnx_exports/`. Run
`python serialization/export_onnx.py` first if they don't exist yet.

---

## Experiments

### `experiment1.html` — Encode / Decode Explorer

Encode audio with EnCodec and compare the reconstruction spectrogram against
the original.

**Source**: sine, triangle, sawtooth, square, sweep (all with duration), or file upload (WAV / MP3 / etc., max 10 s).

**Model**: 24 kHz or 48 kHz, all exported bandwidths.

**Mode**:
- *Non-streaming* — encode the full audio as one pass, LSTM state reset to zero.
- *Streaming* — process in chunks with LSTM state carried across boundaries.
  For 24 kHz, chunk size is user-selectable (320 – 9600 samples).
  For 48 kHz, chunks are always 1 s (48 000 samples).

**Spectrogram**:
- *Source* — original audio in plasma colormap.
- *Decoded* — reconstructed audio in hot colormap.
- *Both* — plasma layer at full opacity, hot layer overlaid with screen blend.

---

## Files

```
web/
├── experiment1.html    Main widget (self-contained except worker + models)
├── encodec-worker.js   Web Worker: onnxruntime-web inference
└── README.md
```
