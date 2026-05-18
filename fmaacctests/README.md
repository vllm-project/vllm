# FMA Accuracy and Benchmark Checks

These scripts compare the old fused indexer-Q RoPE arithmetic:

```python
r_even = x_even * cos - x_odd * sin
r_odd = x_odd * cos + x_even * sin
```

against the explicit `tl.fma` version used in the production kernel.

Run from the vLLM checkout root:

```bash
uv run --no-project python fmaacctests/accuracy.py
uv run --no-project python fmaacctests/benchmark.py
```

The accuracy script compares both variants against the unfused vLLM reference:
`ops.rotary_embedding` followed by `per_token_group_quant_fp8(...,
use_ue8m0=True)`. It returns non-zero only if the `fma` variant mismatches the
reference.

The benchmark script uses preallocated outputs and CUDA events to time only the
Triton kernel launch/execution path. Example with JSON output:

```bash
uv run --no-project python fmaacctests/accuracy.py --json fma_accuracy.json
uv run --no-project python fmaacctests/benchmark.py --json fma_benchmark.json
```

Observed on the local ROCm MI355X environment, the old `muladd` variant had two
FP8 Q mismatches against the vLLM reference for float32 RoPE caches at token
counts 257 and 1023. The `tl.fma` variant had zero mismatches for all tested
cases. The microbenchmark did not show a speed regression; the largest tested
cases were about 4% faster with `tl.fma`.
