# GLM-5.3 latency-specialized decode (NVIDIA)

GLM-5.3 (`GlmMoeDsaForCausalLM`) serves single-request decode at low latency on
8xB200 (SM100) and 8xB300 (SM103) by replacing vLLM's FULL uniform decode graph
of the 78 target layers with CuTe DSL kernels (`ops/cute_dsl`). The MTP drafter
(layer 78) is vLLM's DeepSeek-V3.2 MTP model on vLLM's speculator. Every other
forward (prefill, mixed steps, eager steps) stays on vLLM's stock layers.

## Layout

| Module | Role |
| --- | --- |
| `model.py` | The DSA decoder and `GlmMoeDsaForCausalLM` with the segment hooks, plus `Glm53ModelState` (Model Runner V2), which tracks requests and steps for the segment. |
| `switch.py` | Reads `decode_backend` once, in `GlmMoeDsaForCausalLM.__init__`, into a frozen `LowLatencyDecode` record; refuses unsupported deployments. |
| `decode.py` | `SpecializedDecode`, segment 1 (the target layers): build, launch, capture and the model hook. |
| `weights.py` | The segment's weight forms, built from the checkpoint during load. |
| `kv.py` | KV-cache hygiene between the segment and vLLM's stock forwards (index-key rank completion). |
| `arena.py` | The cross-rank peer arena and its placement (plain memory where the optional locality placement is unavailable). |
| `providers.py` | The segment's launch providers (prologue, selector, worker, FFN): each allocates its results and protocol state, names its peer-mailbox slices and fixes its launch's arguments. |
| `ops/cute_dsl/` | CuTe DSL kernel families (`registry.py`: served entry to kernel class, host sizes, compiled host wrapper and its object cache; `fused_moe.py` also owns the colocated-expert layout). |
| `ops/launch.py` | `Glm53Geometry`, the model's shapes and layer layout read from the config, and binding and running the compiled launches. |

The model class (`model.py`) owns the hooks: the target model's
forward offers each step to the installed segment, and `process_weights_after_loading`
builds the weight forms and installs the segment.
Runner-side hooks that are not model-specific stay in `vllm/v1/worker/gpu/`.

## Switch

`--additional-config '{"decode_backend": "latency_specialized"}'` turns on the
whole path, as Kimi K3's specialized decode does: the kernels, the weight forms,
and segment 1 (`auto`, the default, keeps vLLM's decode). With
the path on, startup refuses, on every TP rank (a MIN all-reduce over the TP group),
unless: the GPU is SM100 or SM103; tensor parallel size is 8 without pipeline
parallelism; the KV cache is FlashInfer MLA's 576-byte fp8 rows
(`--attention-backend FLASHINFER_MLA_SPARSE --kv-cache-dtype fp8`, vLLM's SM10x
default); the segment runs under Model Runner V2.

## Deployment

Set a persistent `VLLM_CACHE_ROOT` (node-local, reused across restarts). It holds
the compiled CuTe host wrappers (`glm53_kernels/<entry>.<arch>.<key>.o`; a cold build is about 40 s per rank) and
DeepGEMM's JIT cache (`deep_gemm`; a cold warm-up takes about 20 minutes). A cold
node also needs `VLLM_ENGINE_READY_TIMEOUT_S=3600`, past vLLM's default 600 s.

```bash
export VLLM_CACHE_ROOT=/local/ssd/vllm-cache VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve zai-org/GLM-5.3 --tensor-parallel-size 8 --block-size 64 \
  --attention-backend FLASHINFER_MLA_SPARSE --kv-cache-dtype fp8 \
  --max-num-seqs 1 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "use_local_argmax_reduction": true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"decode_backend": "latency_specialized"}'
```

- The drafter is vLLM's MTP with three draft tokens (`num_speculative_tokens` 3,
  the segment's T = 4 rows); `use_local_argmax_reduction` takes its greedy drafts
  from a per-rank argmax. Benchmarks at a fixed acceptance length add vLLM's
  synthetic acceptance:
  `"rejection_sample_method": "synthetic", "synthetic_acceptance_length": 2.99`.
- The path serves the prefill MoE from DeepGEMM's experts, whose packed layout the
  decode MoE kernel reads in place (`fused_moe.LAYOUT`); there is one copy of the
  weights. `--moe-backend` is no longer needed: the switch turns vLLM's `auto` into
  `deep_gemm` and refuses any other explicit backend.
- Without `--speculative-config` (MTP off) the segment claims vLLM's one-token
  decode graph and runs it as a T = 4 step: rows 1-3 repeat the token at p+1..p+3
  as padding, vLLM's head samples row 0, and padding rows past p's block write
  vLLM's null block.
- `--max-num-seqs 1`: the path serves one request; other batch shapes and
  every prefill or mixed step run vLLM's stock layers.
- The startup log shows `GLM-5.3 latency-specialized decode on`,
  `GLM-5.3 segment 1 installed (T = 4, claims 4-token forwards)` (1-token without MTP).
