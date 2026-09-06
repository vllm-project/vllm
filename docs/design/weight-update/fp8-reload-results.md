# FP8 per-tensor reload

This change covers serialized per-tensor FP8 weights using the CUTLASS FP8
linear kernel. Block-FP8, non-serialized FP8, and Marlin FP8 remain on the
existing fallback path.

The reload path stages checkpoint-layout `weight`, `weight_scale`, and optional
`input_scale` tensors. At FINISH it validates complete, non-overlapping shard
coverage, applies the existing per-tensor requantization/transpose/padding
logic, and copies the result into the existing runtime storage. It does not
call method or kernel `process_weights_after_loading()` during reload.

H200 validation (fixed environment
`/inspire/hdd/global_user/wangtongyu-25057/miniconda3/envs/vllm`):

- `tests/quantization/test_fp8.py -k test_per_tensor_refresh_without_pwal`:
  20 passed.
- The matrix covers dynamic/static activation scales, complete and missing
  shards, duplicate/overlapping shards, missing scales, repeated refreshes,
  and stable parameter addresses.
- Ruff 0.14.0 format/check passed for the modified Python files.

