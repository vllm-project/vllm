# Meta CI Mode (`VLLM_META_MODE`)

Runs the v1 engine end-to-end on a CPU-only host through Model Runner V2
(MRV2), with `meta`-device weights and KV cache. Scope: functional/integration
validation of construction and execution paths (config, TP/EP, attention/MoE,
quantization, sampling, scheduling, offload). Numerics are out of scope — meta
tensors carry no data.

Mode-wide setup is gated by `envs.VLLM_META_MODE`; tensor-local fallbacks use
`tensor.is_meta`. When unset, `vllm/meta_mode.py` is not imported and there is
zero impact. No monkeypatching of vLLM classes — only third-party launch layers
(Triton / CuteDSL / `torch.cuda`) are intercepted.

---

## 1. Modifications

### 1.1 Weights and KV cache on meta

Weights load via `load_format="meta"` (`MetaModelLoader`: model built on the
meta device, no download, no init; explicit accelerator placements made during
construction are redirected to meta by the loader); both runners' KV-cache
allocation paths place the KV pool on meta. The worker keeps `self.device = cpu`
so IO buffers retain real values for scheduling; attention is faked and never
reads KV.

### 1.2 Kernels

Opaque custom ops (`unified_attention_with_output`, `mamba_mixer`,
`qwen_gdn_attention_core`, `moe_forward`, …) use their owner-provided fake
kernels when one exists. `register_meta_impls()` fills other shape-safe gaps,
for compiled leaf ops and mutate-only `vllm::` ops alike, from operator
schemas (no return or aliased return); independent-output ops such as
`_vllm_fa2_C::varlen_fwd`, `_vllm_fa3_C::fwd`, and the FlashMLA sparse
prefill/decode ops use central shape overrides. Metadata builders project
compute-only tensors to the forward device after retaining their value-carrying
CPU metadata; this includes the MLA chunked-prefill workspace and KDA state
indices.

Native, non-op kernels launched as `kernel[grid](...)` (Kimi-K3 KDA/mamba,
Qwen3-Next `fused_qk_rmsnorm_rope_gate`, DeepSeek-V4/Kimi CuteDSL sparse
attention) cannot carry a `torch.library` meta kernel and are intercepted at the
launch layer instead by `install_launch_hooks()`:

- Triton — MRV2 host-only bookkeeping kernels run through the CPU interpreter;
  model/attention kernels and any launch touching meta tensors are skipped
- CuteDSL — `cutlass.cute.compile` → no-op launcher
- `torch.cuda` — `Event`/`Stream`/`current_device`/`get_device_properties`/
  `is_current_stream_capturing`, plus `torch.Event` and
  `torch.accelerator.synchronize`/`current_device_index` → null stubs

Callers pre-allocate model outputs (on meta) and write in place, so skipping
model launches preserves shapes. MRV2's request-state, input-preparation, and
block-table kernels need real host values and therefore execute in the Triton
interpreter. The DeepGEMM wrapper likewise keeps its kernels unbound in meta
mode even when the vendored package imports: in-place calls become no-ops and
value-returning calls use shape-only fallbacks.

> These launch-layer stubs are a harness-side fallback. Ops that read/write in
> attention should ideally gain a `register_fake` / meta branch at the op
> definition, owned by the kernel maintainer; the interception can then be
> retired per op.

### 1.3 Parallelism (TP / EP / PP / PCP / DCP / DP)

The requested topology is preserved. Model ranks run in the regular
`MultiprocExecutor`, form the standard vLLM groups over Gloo, and execute the
same rank-specific model shards and collective control flow as serving. Tensors
remain on meta, so collectives propagate shape without transferring payloads.
The portable all-gather/reduce-scatter EP manager is used when all-to-all is
required. PP uses synchronous scheduling because its async sampled-token
broadcast requires values that meta tensors do not carry. Offline DP follows
vLLM's normal external-rank launch model.

### 1.4 CPU offload (L2)

`SimpleCPUOffloadConnector` is enabled throughout. The scheduler side (pure
Python) runs unchanged. On the worker side, `register_kv_caches` returns early
(pinned allocation and `DmaCopyBackend` skipped) and every pending transfer
event polls as complete in `get_finished`. Hash bookkeeping is preserved, so
`num_cached_tokens` / `_load_event_counter` semantics hold with no real host
memory.

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

`test_meta_model_e2e` builds each entry (`VLLM_META_MODE` forces
`load_format="meta"` and eager mode; `skip_tokenizer_init=True`) and asserts:

1. Output is the meta sampler's constant token (`[META_TOKEN_ID] × 8`).
2. `hf_overrides`-injected quantization remains the requested scheme.
3. Two-tier offload round-trip (same prompt ×4): R1 cold (stores L2) → R2 L1
   hit (`_load_event_counter` unchanged) → `reset_prefix_cache()` → R3 L2 hit
   (counter++) → `reset_prefix_cache(reset_connector=True)` → R4 cold.

---

## 3. Support Matrix

| Model | Notes |
|---|---|
| `moonshotai/Kimi-K3` | KDA/mamba "align" indices from MRV2 host block tables |
| `Qwen/Qwen3.8-2.4T-A95B-FP8` | Hybrid GDN + full-attn MoE; block-wise `[128,128]` FP8 |
| `deepseek-ai/DeepSeek-V4-Pro` | `fp8_ds_mla` KV + `deepseek_v4_fp8`; CuteDSL/FA4 launch interception |
| `zai-org/GLM-5` | `GlmMoeDsaForCausalLM` (DSA); FA3 meta shape override |
| `Qwen/Qwen3-30B-A3B` | Quant carrier plus TP/EP, PP, and DCP coverage |
| `deepseek-ai/DeepSeek-V2-Lite` | MLA prefill-context-parallel coverage |

Parallelism: real TP=8 + EP in the main matrix, plus PP=2, PCP=2, and DCP=2
smoke cases; external DP=2 follows the standard offline rank launcher.
Quantization follows the requested production scheme. Covered paths include FP8
per-tensor, FP8 block-wise, ModelOpt NVFP4, and `deepseek_v4_fp8`; unsupported
schemes fail through their normal validation or meta-kernel path.
