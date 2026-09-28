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

## DSV4.1 H200 day0 reload validation

The DSV4.1 validation uses the day0 NCCL publisher and a reduced five-layer
checkpoint with two weight variants:

```text
checkpoint A -> cold load -> inference
                 |
                 +-> reset prefix cache
                 +-> NCCL trace reload with checkpoint B
                 +-> inference (warm-B)
checkpoint B -> fresh cold load -> inference (cold-B)
```

The prefix/KV cache must be reset before reload. Otherwise requests can reuse
hidden states computed with checkpoint A, and the warm-B output is not a valid
comparison with a fresh checkpoint-B load. The relevant API call is:

```text
POST /reset_prefix_cache
```

The validated H200 configuration was:

```text
model: DeepSeek V4.1 reduced five-layer checkpoint
transfer backend: NCCL
reload mode: trace
tensor parallel size: 1
data parallel size: 4
expert parallel: enabled
MoE backend: FLASHINFER_CUTLASS_MXFP4_BF16
```

The runner sends the same eight prompts in the same order for cold-A, warm-B,
and cold-B. For each four-request batch it enables forward capture through
`collective_rpc`, sends the request, then reads the capture. The capture
records tensor shape, dtype, SHA256, and invocation order for:

- language-model layers and attention;
- router inputs and outputs;
- routed experts;
- shared experts;
- intermediate FFN modules.

Object IDs and data pointers are ignored when comparing independent server
processes. Tensor hashes, shapes, dtypes, and the ordered call contents are
compared. The runner also compares model parameter hashes and generated text
with logprobs.

The successful evidence is stored at:

```text
/inspire/hdd/global_user/wangtongyu-25057/dsv41-route-audit-20260927-14/
```

The result was:

```json
{
  "status": "PASS",
  "backend": "nccl",
  "reload_mode": "trace",
  "tensor_parallel_size": 1,
  "data_parallel_size": 4,
  "expert_parallel": true,
  "runtime_changed": true,
  "warm_matches_cold": true
}
```

All 1,764 captured parameter records matched between warm-B and cold-B. For
the forward capture, batch 1 matched completely. Batch 0 had 1,820 paired
records with identical tensor hashes; cold-B contained 728 additional
records. Those records were repeated calls with `is_padding=True`, caused by
DP dummy/padding execution, and their tensor contents matched the
corresponding warm-B cycle. The eight generated outputs and logprobs also
matched exactly.

The earlier output mismatch was therefore caused by stale prefix/KV cache
state before reload, not by incorrect weight transfer. A separate Humming
MoE experiment showed nondeterminism in the Humming kernel on H200/CUDA 13;
that result must not be used as evidence of a reload data error.
