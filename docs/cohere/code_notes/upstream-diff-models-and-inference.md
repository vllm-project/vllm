# Upstream Diff Deep Dive: Models, Pooling, and Inference Extensions

Short tracker for fork-vs-upstream models/inference deltas. For full behavior
notes (online quant, LoRA, MXFP8 port, reload weights), see
[`models-and-inference.md`](models-and-inference.md).

## v0.21 path map (restructuring)

Upstream v0.21 refactored several areas the v0.19 doc referenced as single
files. Cohere customizations live in the **successor paths** below:

| Doc (v0.19) path | Current branch path | Notes |
|------------------|---------------------|-------|
| `layers/pooler.py` | `layers/pooler/` package | `DispatchPooler` in `pooler/special.py` |
| `quantization/mxfp8.py` | `quantization/online/mxfp8.py` | Online-quant frontend package |
| MoE routing in `commandr.py` only | `commandr.py` + `cohere2_moe.py` | c5/BLS MoE stack in `cohere2_moe.py` |
| `fused_moe/` flat layout | `fused_moe/experts/`, `fused_moe/oracle/`, `fused_moe/router/` | Kernel call sites split by backend |

## 1) Pooling as a First-Class Layer

Upstream v0.21 split pooling into `vllm/model_executor/layers/pooler/`:

- `abstract.py` — `Pooler` base
- `seqwise/` — sequence-level methods + heads (`embed`, `classify`, `score`)
- `tokwise/` — token-level methods + heads (`token_classify`, `token_embed`)
- `special.py` — **`DispatchPooler`** (task-aware dispatch-by-task routing)

Design consequence:

- pooling model support is no longer ad hoc per model class; models expose
  supported tasks and reuse shared pooler logic via `DispatchPooler`.

## 2) Reward Model Integration

Reward-capable models are wired in:

- `commandr.py` — `Cohere2ForRewardModel` (pooling-enabled, token-classify path)
- `cohere_reward.py` — `CohereForRewardModel` and vision reward variant
- `registry.py` — maps reward architectures into embedding/pooling model registry

Behavioral intent:

- reuse generation backbones for reward scoring while removing generation logits
  path (`del logits_processor`) and adding ranking head + pooler task routing.

## 3) Command-R / EAGLE Integration Details

`commandr_eagle.py` adds draft-model-specific loading behavior:

- draft model lives under `eagle_draft_model` prefix,
- quantization config can differ between target and draft,
- weight loading accommodates naming differences and embed token sharing rules.

Complementary runtime support:

- `vllm/v1/core/kv_cache_utils.py` groups `eagle_draft` layers separately to
  prevent draft layout from perturbing target-layer KV grouping.

Practical impact:

- speculative decoding can coexist with target model KV layout assumptions,
  reducing cache fragmentation side effects.

## 4) Guided Decoding: Structural Tags + Tool Grammar

`vllm/cohere/guided_decoding/*` utilities provide:

- model-specific tag registries (Command-A vs Command-R format differences),
- schema→structural_tag converter supporting mixed schema/tool pathways,
- EBNF tool grammar generation from tool JSON schema.

Notable contract:

- tool grammar and structural tags depend on text architecture resolution
  (`get_text_model_name`), especially for vision models with separate text config.

## 5) Spec Decode and Multimodal Benchmarking Utilities

`examples/features/speculative_decoding/spec_decode_offline.py` (Cohere CI
spec-decode entrypoint) supports:

- `ngram-eagle` method,
- custom multimodal prompts,
- local media path permissioning,
- acceptance-length and per-position acceptance metrics.

This aligns with benchmark dataset extensions where `custom_mm` data and
multimodal chat transforms are required for realistic Cohere model evaluation.

## 6) SigmoidRenorm Routing and `norm_topk_prob` for Cohere MoE

Cohere MoE models (c4/c5) use a sigmoid-based expert routing scheme that
differs from upstream's softmax-based paths. Changes span config enums, the
custom routing function, `FusedMoEConfig`, and FlashInfer/TRT-LLM kernel call
sites under `vllm/model_executor/layers/fused_moe/`.

### Routing method enum

`RoutingMethodType` in `fused_moe/config.py` gains `SigmoidRenorm = 6`.
`get_routing_method_type` returns it when `scoring_func == "sigmoid"` and
`top_k > 1`.

### Custom routing function

`token_choice_with_bias` (sigmoid → top-k → optional renormalize) is defined in
`commandr.py` and **`cohere2_moe.py`** (c5/BLS path). `CustomRoutingRouter` in
`fused_moe/router/custom_routing_router.py` maps this function to
`RoutingMethodType.SigmoidRenorm`.

### `norm_topk_prob` on `FusedMoEConfig`

`norm_topk_prob: bool = True` controls whether top-k weights are renormalized
after selection. Threaded into FlashInfer/TRT-LLM kernel call sites
(`experts/trtllm_{bf16,fp8,nvfp4}_moe.py`, `modular_kernel.py`). Set from HF
config in `commandr.py` / `cohere2_moe.py`.

### Kernel selection

`_supports_routing_method_bf16` includes `SigmoidRenorm` so fused kernels are
eligible for Cohere MoE models.

### Rebase risk

`# cohere` tagged changes are scattered across ~10 files in `fused_moe/`.
Upstream refactors to kernel call signatures or `RoutingMethodType` enum values
will silently conflict.

## 7) Online Quantization Frontend

Online quantization (quantize bf16/fp16 weights at load time) lives under
`vllm/model_executor/layers/quantization/online/`:

- `base.py` — `OnlineQuantizationConfig`; Cohere **`from_config`** parses
  checkpoint `quantization_config` blocks (`fp8_per_block`, `mxfp8`, etc.)
- `fp8.py` — per-tensor / per-block FP8 linear + MoE; includes Cohere PTPC path
- `mxfp8.py`, `int8.py`, `moe_base.py` — scheme-specific online methods

Two entry points: CLI `--quantization` (upstream) and Cohere `config.json`
`quantization_config` (see `tests/cohere/cpu/test_online_quant_from_config.py`).

Details: [`models-and-inference.md` §12](models-and-inference.md).

## 8) Secondary Performance-Oriented Deltas

Additional model/runtime deltas:

- expanded fused MoE configs for newer GPUs,
- compressed tensor / Marlin FP4 utility updates,
- benchmark support for Cohere2MoE in kernel benchmark script,
- pre-quantized MXFP8 via compressed-tensors (temporary `#38815` port — see
  [`models-and-inference.md` §11](models-and-inference.md)).

These are mostly enablement and tuning changes; easy to overlook during rebase
because they live outside "obvious model files".

## 9) Rebase Hotspots and Validation

High-conflict files:

- `vllm/model_executor/layers/pooler/` (especially `special.py`)
- `vllm/model_executor/models/commandr.py`
- `vllm/model_executor/models/cohere2_moe.py`
- `vllm/model_executor/models/commandr_eagle.py`
- `vllm/model_executor/models/cohere_reward.py`
- `vllm/model_executor/models/cohere2_vision.py` (`load_weights` must skip `lm_head`)
- `vllm/model_executor/layers/fused_moe/` (`config.py`, `experts/`, `router/`)
- `vllm/model_executor/layers/quantization/online/`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/cohere/guided_decoding/*.py`

Validation checklist:

1. Load reward models (text + vision) and run score endpoint.
2. Verify supported pooling tasks match model task requests (`DispatchPooler`).
3. Run EAGLE speculative path with Cohere models.
4. Run guided generation with both tool grammar and structural tag JSON schema.
5. Run multimodal spec decode script using local media path.
6. Load quantized vision checkpoint (e.g. `c4-25a218t_int4a16`) — must not fail on `lm_head` during `load_weights`.
7. Online quant from CLI: `pytest tests/quantization/test_online.py -v`
8. Online quant from `config.json`:
   `pytest tests/cohere/cpu/test_online_quant_from_config.py -v`
