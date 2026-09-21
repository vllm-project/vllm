# Meta CI Mode (`VLLM_META_MODE`)

Runs the v1 engine end-to-end on a CPU-only host with `meta`-device weights and
KV cache. Scope: functional/integration validation of construction and
execution paths (config, TP/EP, attention/MoE, quantization, sampling,
scheduling, offload). Numerics are out of scope — meta tensors carry no data.

Everything is gated by `if envs.VLLM_META_MODE:`; when unset,
`vllm/meta_mode.py` is not imported and there is zero impact. No monkeypatching
of vLLM classes — only third-party launch layers (Triton / CuteDSL /
`torch.cuda`) are intercepted.

---

## 1. Modifications

### 1.1 Weights and KV cache on meta

Weights load via `load_format="dummy"` onto `device="meta"`;
`_allocate_kv_cache_tensors` places the KV pool on meta. The worker keeps
`self.device = cpu` so IO buffers and slot mappings retain real values for
scheduling; attention is faked and never reads KV.

### 1.2 Kernels

Opaque custom ops (`unified_attention_with_output`, `mamba_mixer`,
`qwen_gdn_attention_core`, `moe_forward`, …) dispatch to their existing
`register_fake` kernels. The direct `torch.ops._C.*` ops that lacked one (fp8
quant ops, `_vllm_fa3_C::fwd`) got a `register_fake` at their definition site.

Native, non-op kernels launched as `kernel[grid](...)` (Kimi-K3 KDA/mamba,
Qwen3-Next `fused_qk_rmsnorm_rope_gate`, DeepSeek-V4/Kimi CuteDSL sparse
attention) cannot carry a `torch.library` meta kernel and are neutralised at the
launch layer instead:

- `disable_triton_launch()` — `JITFunction.run` → no-op
- `disable_cutedsl_launch()` — `cutlass.cute.compile` → no-op launcher
- `patch_cuda_primitives()` — `Event`/`Stream`/`get_device_properties`/
  `torch.accelerator.synchronize` → null stubs

Callers pre-allocate outputs (on meta) and write in place, so skipping the
launch preserves shapes.

> These launch-layer stubs are a harness-side fallback. Ops that read/write in
> attention should ideally gain a `register_fake` / meta branch at the op
> definition, owned by the kernel maintainer; the interception can then be
> retired per op.

### 1.3 Parallelism (TP / EP / DCP)

Emulated in-process. `ParallelConfig.__post_init__` records the requested
TP/DCP sizes into `meta_mode._logical_{tp,dcp}`, then collapses world size to 1
(uni executor). Before the runner/model are built, `enable_fake_tp` /
`enable_fake_dcp` set `world_size`/`rank_in_group` on the TP/DCP groups and
replace their collectives with shape-correct stubs. Result: layers perform a
**real** weight shard by `tp_size` (EP derived from TP), validating shard shapes
and index math — not cross-rank transfer.

### 1.4 CPU offload (L2)

`SimpleCPUOffloadConnector` is enabled throughout. The scheduler side (pure
Python) runs unchanged. On the worker side, `register_kv_caches` returns early:
`cpu_kv_caches = {}`, pinned allocation and `DmaCopyBackend` skipped, transfers
marked complete in `get_finished`. Hash bookkeeping is preserved, so
`num_cached_tokens` / `_load_event_counter` semantics hold with no real host
memory (`cpu_bytes_to_use_per_rank` only sizes the `num_cpu_blocks` ceiling).

### 1.5 Env vars and latency model

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_META_MODE` | `0` | Master gate |
| `VLLM_META_TOKEN_LATENCY_MS` | `0.05` | Simulated per-token latency (ms) |

With no real compute, `MetaAsyncOutput` sets a post-forward deadline of
`total_num_scheduled_tokens × VLLM_META_TOKEN_LATENCY_MS` and blocks in
`get_output()`, so pre-output scheduling overlaps the simulated latency (mirrors
the real async-output path). Timing only; no data effect.

---

## 2. Tests (`test_meta_models.py`)

`test_meta_model_e2e` builds each entry (`load_format="dummy"`,
`skip_tokenizer_init=True`, `enforce_eager=True`) and asserts:

1. Output is the meta sampler's constant token (`[100] × 8`).
2. `hf_overrides`-injected quantization stays live (not neutralised to bf16).
3. Two-tier offload round-trip (same prompt ×4): R1 cold (stores L2) → R2 L1
   hit (`_load_event_counter` unchanged) → `reset_prefix_cache()` → R3 L2 hit
   (counter++) → `reset_prefix_cache(reset_connector=True)` → R4 cold.

---

## 3. Support Matrix and Footprint

| Model | Notes |
|---|---|
| `moonshotai/Kimi-K3` | KDA/mamba "align" indices via pure-torch gather on meta |
| `Qwen/Qwen3.8-2.4T-A95B-FP8` | Hybrid GDN + full-attn MoE; block-wise `[128,128]` FP8 |
| `deepseek-ai/DeepSeek-V4-Pro` | `fp8_ds_mla` KV + `deepseek_v4_fp8`; CuteDSL/FA4 meta-gated |
| `zai-org/GLM-5` | `GlmMoeDsaForCausalLM` (DSA); FA3 `_vllm_fa3_C::fwd` fake |
| `Qwen/Qwen3-30B-A3B` | Quant carrier: FP8 per-tensor / NVFP4 (g16) via `hf_overrides` |

Parallelism: TP=8 + EP; DCP slot-mapping path. Quantization kept live on meta
(apply leaf short-circuited): FP8 per-tensor, FP8 block-wise, ModelOpt NVFP4,
`deepseek_v4_fp8`; other schemes neutralised to bf16.

### Measured footprint

CPU-only host, TP=8 + EP, `max_model_len=16384`, 8704-token prompt, offload on.
RSS via `/proc/self/status`, peak via `getrusage(VmHWM)`, in MB.

| Stage | Qwen3-30B-A3B | Qwen3.8-2.4T-A95B-FP8 |
|---|---:|---:|
| `import torch` | 668 | 668 |
| `import vllm` (baseline) | 1134 | 1134 |
| `LLM` build | 1460 (+325) | 1653 (+519) |
| `generate` (8704 tok) | 1473 (+14) | 1672 (+19) |
| **Peak RSS** | **1473** | **1672** |

- Footprint is decoupled from parameter count: a 2.4T model builds within
  ~200 MB of a 30B one (build delta is Python-side config/metadata, not weights).
- Peak is dominated by the torch+vllm import baseline (~1.1 GB).
- Inference adds ~14–19 MB; activations are meta and compute leaves are
  short-circuited.
- Weights, KV pool, and L2 buffer are 0 real bytes; the only genuine
  allocations are CPU scheduling tensors (order MB, scaling with
  `max_num_seqs × max_model_len`).
