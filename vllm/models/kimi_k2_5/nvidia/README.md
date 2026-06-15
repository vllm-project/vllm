# Kimi-K2.5 NVFP4 specialized kernels

This package hosts the custom [CuTe DSL](https://docs.nvidia.com/cutlass/)
kernels used by the upcoming `nvidia/Kimi-K2.5-NVFP4` specialized model. The
kernels are checked in ahead of the model so they can be reviewed and tested in
isolation; the specialized model will import them from the [`ops/`](./ops)
package once it lands.

This PR scopes the kernels to the **decode** path, which is what we have
profiled and want to optimize. Two kernels are included; the prefill-path
kernels are intentionally omitted until they are profiled.

All kernels target **Blackwell (SM10x)** GPUs and require the optional `cutlass`
(`nvidia-cutlass-dsl`) dependency. They specialize the MLA attention path for
this checkpoint, which fixes:

- `kv_lora_rank == q_lora_dim == 512`
- `qk_rope_head_dim` (`pe_dim`) `== 64`
- `bfloat16` activations, `e4m3` FP8 paged KV cache.

Inputs are `bfloat16`; FP8 quantization converts to `e4m3` stored as `uint8`
(returned as the platform FP8 dtype). RoPE is the interleaved (non-NeoX) MLA
variant: the two halves of each rotary pair are adjacent in memory.

## Compilation & caching convention

Each kernel is a `@cute.kernel` device function plus a `@cute.jit` launcher,
one per module under [`ops/`](./ops). Compilation is cached following the
convention in
[`vllm/v1/attention/ops/deepseek_v4_ops`](../../../v1/attention/ops/deepseek_v4_ops):

- A `functools.cache`-decorated `_compile_*` helper builds **fake tensors**
  (`cute.runtime.make_fake_tensor`, via the shared
  [`ops/cutedsl_utils.py`](./ops/cutedsl_utils.py) helpers) with symbolic
  shapes/strides (`cute.sym_int` / `cute.sym_int64`) and a fake stream, then
  calls `cute.compile(..., options="--enable-tvm-ffi")`. The cache key is the
  set of compile-time (constexpr) parameters only, so a kernel compiles once
  per configuration and is reused across token counts.
- The compiled executor is invoked **directly with torch tensors** and sources
  its launch stream from the TVM-FFI environment, so the public `_run_*`
  wrappers do not build CuTe tensors or pass a stream at call time.

## Kernels

Public entry points are the `_run_*` helpers (torch-tensor in / out). They map
onto the two per-token steps of the Kimi-K2.5 MLA decode path.

### `_run_kimik25_qkv_rmsnorm_k_pe_fused`
[`ops/qkv_rmsnorm_k_pe_fused.py`](./ops/qkv_rmsnorm_k_pe_fused.py).
Runs once per layer on the full batch, right after the fused QKV-A projection.
In a single launch it fuses, over the fused Q/KV LoRA projection (`data`, width
`lora_dim_q + lora_dim_kv`, written in place) and the rotary key (`k_pe`, in
place):
1. Q-LoRA RMSNorm with `weights_q`/`eps_q` (the first three of four column
   groups, since `lora_dim_q == 3 * lora_dim_kv`),
2. KV-LoRA RMSNorm with `weights_kv`/`eps_kv` (the fourth group),
3. interleaved RoPE on `k_pe` from `positions` + `cos_sin_cache`.

### `_run_kimik25_decode_rope_concat_quant_fp8_and_cache_mla`
[`ops/decode_rope_concat_quant_fp8_and_cache_mla.py`](./ops/decode_rope_concat_quant_fp8_and_cache_mla.py).
The fused decode-query + KV-cache-write step. A single linearized grid covers
two halves:
- **Decode query:** RoPE on `q_pe`, concatenated after `ql_nope`, then
  FP8-quantized by `q_scale` into a fresh `uint8` buffer returned as FP8 (shape
  `(num_tokens, num_heads, q_lora_dim + pe_dim)`).
- **KV-cache write:** `kv_c`/`k_pe` quantized by `kv_scale` and written into the
  paged FP8 KV cache at the `slot_mapping` slots (tokens with `slot_idx < 0` are
  skipped). When the batch mixes decode and prefill tokens, the full
  `slot_mapping` is passed so this single launch writes the cache for the whole
  batch.

## Testing

The kernels are validated against PyTorch / vLLM-op reference implementations in
[`tests/`](./tests) (RMSNorm + RoPE references, and `scaled_fp8_quant` /
`concat_and_cache_mla` for the decode kernel). The tests skip automatically
unless they run on a Blackwell GPU with `cutlass` installed:

```bash
pytest vllm/models/kimi_k2_5/nvidia/tests
```
