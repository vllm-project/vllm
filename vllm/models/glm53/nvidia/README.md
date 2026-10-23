# GLM-5.3 latency-specialized decode (NVIDIA)

GLM-5.3 (`GlmMoeDsaForCausalLM`) serves single-request decode at low latency on
8xB200 (SM100) and 8xB300 (SM103) by replacing vLLM's FULL uniform decode graph
of the 78 target layers with CuTe DSL kernels (`ops/cute_dsl`). The MTP drafter
(layer 78) runs on vLLM's speculator, its decode steps as CuTe DSL kernels too
(Specialized drafter below). Every other forward (prefill, mixed steps, eager
steps) stays on vLLM's stock layers.

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
segment 1 and the drafter segment (`auto`, the default, keeps vLLM's decode). With
the path on, startup refuses, on every TP rank (a MIN all-reduce over the TP group),
unless: the GPU is SM100 or SM103; tensor parallel size is 8 without pipeline
parallelism; the KV cache is FlashInfer MLA's 576-byte fp8 rows
(`--attention-backend FLASHINFER_MLA_SPARSE --kv-cache-dtype fp8`, vLLM's SM10x
default); the segment runs under Model Runner V2; MTP, if configured, sets
`use_local_argmax_reduction`.

## Deployment

Set a persistent `VLLM_CACHE_ROOT` (node-local, reused across restarts). It holds
the compiled CuTe host wrappers (`glm53_kernels/<entry>.<arch>.<key>.o`; a cold build is about 40 s per rank) and
DeepGEMM's JIT cache (`deep_gemm`; a cold warm-up takes about 20 minutes). A cold
node also needs `VLLM_ENGINE_READY_TIMEOUT_S=3600`, past vLLM's default 600 s.
`VLLM_POST_FORWARD_CUDA_GRAPH=1` captures the sampling and state update after
each MTP decode step in a CUDA graph (Model Runner V2).

```bash
export VLLM_CACHE_ROOT=/local/ssd/vllm-cache VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_POST_FORWARD_CUDA_GRAPH=1
vllm serve zai-org/GLM-5.3 --tensor-parallel-size 8 --block-size 64 \
  --attention-backend FLASHINFER_MLA_SPARSE --kv-cache-dtype fp8 \
  --max-num-seqs 1 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "use_local_argmax_reduction": true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"decode_backend": "latency_specialized"}'
```

- The drafter is vLLM's MTP with three draft tokens (`num_speculative_tokens` 3,
  the segment's T = 4 rows); `use_local_argmax_reduction` takes its greedy drafts
  from a per-rank argmax (the switch requires it with MTP). Benchmarks at
  a fixed acceptance length add vLLM's synthetic acceptance:
  `"rejection_sample_method": "synthetic", "synthetic_acceptance_length": 2.99`.
- The path serves the prefill MoE from DeepGEMM's experts, whose packed layout the
  decode MoE kernel reads in place (`fused_moe.LAYOUT`); there is one copy of the
  weights. `--moe-backend` is no longer needed: the switch turns vLLM's `auto` into
  `deep_gemm` and refuses any other explicit backend.
- Without `--speculative-config` (MTP off) the segment claims vLLM's one-token
  decode graph and runs it as a T = 4 step: rows 1-3 repeat the token at p+1..p+3
  as padding, vLLM's head samples row 0, and padding rows past p's block write
  vLLM's null block.
- The KV pool needs no `--kv-cache-memory-bytes`: the segments built during
  vLLM's CUDA graph memory profiling are rebuilt over the pool vLLM binds after it.
- `--max-num-seqs 1`: the path serves one request; other batch shapes and
  every prefill or mixed step run vLLM's stock layers.
- The startup log shows `GLM-5.3 latency-specialized decode on`,
  `GLM-5.3 segment 1 installed (T = 4, claims 4-token forwards)` (1-token without MTP).

## Specialized drafter

With `latency_specialized` the MTP drafter (layer 78) also runs as CuTe DSL kernels.
The target model registers `GlmMoeDsaMTP` (`mtp.py`, vLLM's DeepSeek-V3.2 MTP model with
the drafter hooks) under the `DeepseekV32MTPModel` architecture before vLLM builds the
drafter; without the switch vLLM's stock drafter runs. Per decode step the drafter runs a
catch-up over the step's four rows (vLLM's draft prefill) and one proposal per later draft:

| Module | Role |
| --- | --- |
| `mtp.py` | `GlmMoeDsaMTP`, the drafter model with the hooks, and its registration. |
| `draft.py` | The drafter segment: hook, graphs, rebuild after a KV-cache rebind, warm-up, and weight forms (layer 78 bound in place to vLLM's tensors). |
| `draft_providers.py` | The drafter's launch providers (`providers.Provider`): results, protocol state and peer-arena slices, bound and run like segment 1's. |
| `ops/cute_dsl/mla_draft.py` | `DraftWorker`: the layer-78 worker over vLLM's eight heads of each rank (four rows for the catch-up, one for a proposal). |
| `ops/cute_dsl/mtp_preprocess.py` | The MTP preprocess (`enorm`, `hnorm`, `eh_proj`). |
| `ops/cute_dsl/lm_head.py` | The BF16 draft head `lm_head_top1_q1`: final RMSNorm, the rank's vocabulary-shard logits and their top-1. |

The catch-up's selection runs the only-row selector shards (`SelectorOnlyRow`, ranks 0-2)
before the worker; every rank runs the worker in both parts. The worker writes its attended
row four times, and the MoE is the target's `fused_moe_q4_ckpt_shared`; the BF16 draft head
(`lm_head_top1_q1`) reads row 0.

The drafter reads layer 78 as stock vLLM loads it for the shipped checkpoint: the worker's
`fused_qkv_a`, `q_b`, `kv_b` and `o_proj` and the selector's `indexer.wq_b` are E4M3 with
DeepGEMM UE8M0 scales, moved once into `VllmLinear` buffers like the target's; the routed
experts are vLLM's deep_gemm experts and the shared expert sits in slot 256, as in the target
layers; `eh_proj` (read through a TMA map), `wk_weights_proj`, the router, the norms, the
embedding and the head are BF16 as loaded; the KV cache is vLLM's FP8 cache. Activations are
BF16 with FP32 accumulation.
The startup log shows `GLM-5.3 drafter segment installed (T = 4, 2 proposals)`, and after
the first eager draft prefill `GLM-5.3 drafter segment built`.
