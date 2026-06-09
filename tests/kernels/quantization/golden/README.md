# Golden Baselines for Kernel Performance Regression Tests

This directory contains per-GPU JSON files with golden TFLOP/s baselines for
the hybrid W4A16 kernel.  The test in `test_hybrid_w4a16_perf.py` compares
measured performance against these values using a two-sided tolerance band.

## JSON Schema

```json
{
  "gpu": "<gcnArch prefix, e.g. gfx1151>",
  "shapes": [
    {
      "in_features": 2560,
      "out_features": 3840,
      "group_size": 128,
      "comment": "Qwen3-4B qkv_proj",
      "skip": "(optional) reason to skip entire shape",
      "providers": [
        {
          "provider": "hybrid-w4a16",
          "skip": "(optional) reason to skip this provider",
          "baselines": [
            {
              "batch_size": 1,
              "kernel": "wvsplitk_int4",
              "tflops": 5.12,
              "expected_failure": "(optional) reason this is expected to fail",
              "intermittent": false,
              "skip": "(optional) reason to skip this batch size"
            }
          ]
        }
      ]
    }
  ]
}
```

### Fields

| Field | Level | Description |
| --- | --- | --- |
| `gpu` | top | GCN architecture prefix (e.g. `gfx1151`). Matched against `_GCN_ARCH` at runtime. |
| `in_features` | shape | K dimension of the GEMM. |
| `out_features` | shape | N dimension of the GEMM. |
| `group_size` | shape | Quantization group size (typically 128). |
| `comment` | shape | Human-readable label (model name + layer). |
| `skip` | shape/provider/baseline | When present, the item is skipped. Value is the reason string. `tflops` is not required when `skip` is set. |
| `provider` | provider | Kernel variant. The base name is `hybrid-w4a16`; suffix `-zp` selects the asymmetric (per-group zero-point) dequant path, and suffix `-bf16` runs the kernel with bfloat16 activations/scales/zp instead of float16. Valid combinations: `hybrid-w4a16`, `hybrid-w4a16-zp`, `hybrid-w4a16-bf16`, `hybrid-w4a16-zp-bf16`. |
| `kernel` | baseline | Expected kernel label (e.g. `wvsplitk_int4` or `hybrid_triton_w4a16`). Kernel mismatch is a test failure. |
| `tflops` | baseline | Golden TFLOP/s value for this batch size. |
| `batch_size` | baseline | M dimension (number of tokens). |
| `expected_failure` | baseline | When present, out-of-band results are silently accepted. If the measurement lands *inside* the band, the test errors (unexpected pass). |
| `intermittent` | baseline | When `true`, this batch size is skipped unless `--intermittent` or `--write-golden` is passed. |

### Constraints

- Shapes in the JSON must be a subset of `SHAPES` in
  `test_hybrid_w4a16_perf.py`.  Extra shapes in the JSON cause a collection
  error.
- Batch sizes in the JSON must be a subset of `BATCH_SIZES` in the test file.
  Extra batch sizes cause a collection error.
- Shapes are sorted by `(in_features, out_features, group_size)` for clean
  diffs.

## Contributor Workflow (Local Testing)

### No performance impact expected

```bash
.venv/bin/python -m pytest tests/kernels/quantization/test_hybrid_w4a16_perf.py \
    -v -s
```

If it passes, done.

### Measuring a New Baseline

1. Verify that your system reproduces the current golden baselines
   *before* your change, using the command given above.  If it fails,
   your system does not match the baseline environment.  Investigate
   before proceeding: new baselines measured on a non-representative
   system will be invalid.
2. With your changes applied, measure new baselines:
   ```bash
   .venv/bin/python -m pytest tests/kernels/quantization/test_hybrid_w4a16_perf.py \
       --write-golden -s
   ```
3. Run the tests as many times as you feel necessary to be confident
   in the new baseline.
4. Review the output of `git diff golden/`.
5. `git add` and commit the updated golden files.

### Adding New Shapes

1. Add the shape to `SHAPES` in `test_hybrid_w4a16_perf.py`.
2. Follow the procedure above for measuring a new baseline.

> [!NOTE]
> It is recommended to retain the updated measurements of *all* cases
> in the golden file -- not just the new shapes.  Although the test
> harness includes strategic cool-down intervals, changing the set of
> tests can still affect long-term heat accumulation and subtly shift
> measurements.  Neglecting to update some values could lead to
> intermittent failures over time.

### Adding a New GPU Target

Follow the procedure above for measuring a new baseline.  No golden
file needs to exist first.  The test auto-discovers JSON files by GPU
match and bootstraps from `SHAPES × PROVIDERS × BATCH_SIZES`.
