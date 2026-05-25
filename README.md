# EnCodec — ONNX Export Fork

This repository is a fork of [Meta's EnCodec](https://github.com/facebookresearch/encodec) extended to support ONNX serialization of the encoder and decoder. The original README has been preserved as [`encodec_readme.md`](encodec_readme.md).

---

## Repository Structure

```
.
├── encodec/              # Original Meta EnCodec source code (unmodified)
├── rt_encodec/           # Modified package with ONNX-compatible changes (see below)
├── serialization/        # ONNX export scripts and notebook
│   ├── serialization.ipynb          # Step-by-step ONNX export walkthrough
│   └── test_serialization_traced.py # JIT tracing / state-dict serialization tests
├── tests/                # Numerical equivalence tests (encodec vs rt_encodec)
│   └── test_equivalence.py          # Comprehensive bit-exact comparison across all configs
├── encodec_readme.md     # Original Meta README
└── README.md             # This file
```

`encodec` is kept intact as the upstream reference. All modifications live exclusively in `rt_encodec`. Import from `rt_encodec` wherever ONNX export is needed.

---

## Changes in `rt_encodec`

### `modules/conv.py`

**`get_extra_padding_for_conv1d`** — replaced `math.ceil()` on a Python float with pure integer arithmetic so the expression is ONNX-traceable:

```python
# Before (not ONNX-traceable)
n_frames = (length - kernel_size + padding_total) / stride + 1
ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)

# After (integer ceiling: ceil(a/stride) + 1 = (a + stride - 1) // stride + 1)
n_frames_ceil = (length - kernel_size + padding_total + stride - 1) // stride + 1
ideal_length = (n_frames_ceil - 1) * stride + (kernel_size - padding_total)
```

**`pad_for_conv1d`** — removed (was unused; `SConv1d` calls `get_extra_padding_for_conv1d` directly).

**`pad1d`** — decorated with `@torch.jit.script`. Replaced the conditional branch `if length <= max_pad` with an always-computed `extra_pad = max(0, max_pad - length + 1)` so the function traces cleanly. Removed asserts incompatible with TorchScript.

**`unpad1d`** — removed asserts that cannot be traced.

---

### `modules/lstm.py`

**`SLSTM.forward`** — initialised `h0` and `c0` explicitly as zero tensors instead of passing `None`. The ONNX exporter cannot trace `nn.LSTM` with implicit hidden-state initialisation:

```python
# Before
y, _ = self.lstm(x)

# After
h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, ...)
c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, ...)
y, _ = self.lstm(x, (h0, c0))
```

---

### `model.py`

Added an `exporting_to_onnx: bool = False` constructor argument. When `True`, Python-level asserts that depend on runtime values (input rank, channel count, segment checks) are skipped so the tracer can proceed with concrete dummy inputs:

```python
model.exporting_to_onnx = True   # set before torch.onnx.export(...)
```

No forward-pass logic is altered; only guard asserts are gated.

---

### `quantization/core_vq.py`

**`ResidualVectorQuantization.decode`** — replaced the scalar `torch.tensor(0.0)` accumulator (which causes shape-broadcast issues) with a `None`-initialised accumulator, and replaced direct iteration over the tensor with `torch.unbind(q_indices, dim=0)` for cleaner tracing:

```python
# Before
quantized_out = torch.tensor(0.0, device=q_indices.device)
for i, indices in enumerate(q_indices):
    quantized_out = quantized_out + layer.decode(indices)

# After
quantized_out = None
for i, indices in enumerate(torch.unbind(q_indices, dim=0)):
    quantized = layer.decode(indices)
    quantized_out = quantized if quantized_out is None else quantized_out + quantized
```

---

### `utils.py`

Commented out the `assert sum_weight.min() > 0` guard in `_linear_overlap_add` — it is a Python-level check that is irrelevant inside a traced graph.

---

## Serialization Usage

Scripts in `serialization/` add the repo root to `sys.path` automatically, so no install step is required.

**Notebook** (`serialization/serialization.ipynb`): walks through loading a pretrained model, running encode/decode, and exporting the encoder and decoder to ONNX via `torch.onnx.export`.

**Test script** (`serialization/test_serialization_traced.py`): verifies three serialization approaches — state-dict round-trip, full-model pickle, and JIT tracing.

Run from the repo root or from inside `serialization/`:

```bash
python serialization/test_serialization_traced.py
```

---

## Tests

`tests/test_equivalence.py` verifies that `rt_encodec` produces bit-exact output
compared to the original `encodec` for every combination of model, bandwidth, and
input length. It uses random weights (no pretrained download required) and covers:

| Test class | What is checked |
|---|---|
| `TestEncode` | `encode()` codebook indices and scale factors are identical |
| `TestDecode` | `decode()` reconstructed audio is identical given the same frames |
| `TestForward` | Full encode+decode roundtrip output is identical |
| `TestExportingFlagNeutral` | `exporting_to_onnx=True` does not alter any computed values |
| `TestBatchEquivalence` | Results agree at batch size > 1 |
| `TestPaddingRegression` | Targeted regression for inputs where `length % stride != 0` — the cases most sensitive to the ceiling-arithmetic fix in `get_extra_padding_for_conv1d` |

Both 24 kHz (bandwidths 1.5 / 3 / 6 / 12 / 24 kbps) and 48 kHz (3 / 6 / 12 / 24 kbps)
models are covered, with input lengths chosen to hit clean multiples, off-by-one edges,
and multi-segment inputs that exercise the overlap-add decoder path.

Run from the repo root:

```bash
pytest tests/test_equivalence.py -v
```

---

## Original EnCodec

See [`encodec_readme.md`](encodec_readme.md) for the full original documentation, model details, training setup, and citation.
