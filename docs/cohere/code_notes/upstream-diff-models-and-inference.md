# Upstream Diff Deep Dive: Models, Pooling, and Inference Extensions

## 1) Pooling as a First-Class Layer

A major Cohere addition is `vllm/model_executor/layers/pooler.py`, which introduces:

- task-aware pooling interfaces (`embed`, `classify`, `score`, `token_classify`, `token_embed`),
- composable pooling method + head abstraction,
- dispatch-by-task routing (`DispatchPooler`).

Design consequence:

- "pooling model" support is no longer ad hoc per model class; models can expose supported tasks and reuse shared pooler logic.

## 2) Reward Model Integration

Reward-capable models are wired in two places:

- `commandr.py` adds `Cohere2ForRewardModel` (pooling-enabled, token-classify path).
- `cohere_reward.py` adds `CohereForRewardModel` and vision reward variant.
- `registry.py` maps reward architectures into embedding/pooling model registry.

Behavioral intent:

- reuse generation backbones for reward scoring while removing generation logits path (`del logits_processor`) and adding ranking head + pooler task routing.

## 3) Command-R / EAGLE Integration Details

`commandr_eagle.py` adds draft-model-specific loading behavior:

- draft model lives under `eagle_draft_model` prefix,
- quantization config can differ between target and draft,
- weight loading accommodates naming differences and embed token sharing rules.

Complementary runtime support:

- `kv_cache_utils.py` groups `eagle_draft` layers separately to prevent draft layout from perturbing target-layer KV grouping.

Practical impact:

- speculative decoding can coexist with target model KV layout assumptions, reducing cache fragmentation side effects.

## 4) Guided Decoding: Structural Tags + Tool Grammar

New `vllm/cohere/guided_decoding/*` utilities provide:

- model-specific tag registries (Command-A vs Command-R format differences),
- schema->structural_tag converter supporting mixed schema/tool pathways,
- EBNF tool grammar generation from tool JSON schema.

Notable contract:

- tool grammar and structural tags depend on text architecture resolution (`get_text_model_name`), especially for vision models with separate text config.

## 5) Spec Decode and Multimodal Benchmarking Utilities

`examples/offline_inference/spec_decode.py` now supports:

- `ngram-eagle` method,
- custom multimodal prompts,
- local media path permissioning,
- acceptance-length and per-position acceptance metrics.

This aligns with benchmark dataset extensions where `custom_mm` data and multimodal chat transforms are required for realistic Cohere model evaluation.

## 6) SigmoidRenorm Routing and `norm_topk_prob` for Cohere MoE

Cohere MoE models (c4/c5) use a sigmoid-based expert routing scheme that
differs from upstream's softmax-based paths. The changes span config
enums, the custom routing function, the `FusedMoEConfig` dataclass, and
every FlashInfer/TRT-LLM kernel call site.

### Routing method enum

`RoutingMethodType` gains `SigmoidRenorm = 6`. `get_routing_method_type`
returns it when `scoring_func == "sigmoid"` and `top_k > 1`.

### Custom routing function

`commandr.py` defines `token_choice_with_bias` (sigmoid -> top-k ->
optional renormalize). `CustomRoutingRouter.routing_method_type` maps this
function to `RoutingMethodType.SigmoidRenorm`.

### `norm_topk_prob` on `FusedMoEConfig`

New `norm_topk_prob: bool = True` field controls whether top-k weights are
renormalized after selection. Threaded into all FlashInfer/TRT-LLM kernel
call sites (bf16, fp8, nvfp4). Set from HF config in `commandr.py`.

### Kernel selection

`_supports_routing_method_bf16` includes `SigmoidRenorm` so fused kernels
are eligible for Cohere MoE models.

### Rebase risk

These are `# cohere` tagged changes scattered across ~8 files in
`fused_moe/`. Upstream refactors to kernel call signatures or
`RoutingMethodType` enum values will silently conflict.

## 7) Secondary Performance-Oriented Deltas

Additional but relevant model/runtime deltas:

- expanded fused MoE configs for newer GPUs,
- compressed tensor / Marlin FP4 utility updates,
- benchmark support for Cohere2MoE in kernel benchmark script.

These are mostly enablement and tuning changes; they are easy to overlook during rebase because they live outside "obvious model files".

## 8) Rebase Hotspots and Validation

High-conflict files:

- `vllm/model_executor/layers/pooler.py`
- `vllm/model_executor/models/commandr.py`
- `vllm/model_executor/models/commandr_eagle.py`
- `vllm/model_executor/models/cohere_reward.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/cohere/guided_decoding/*.py`

Validation checklist:

1. Load reward models (text + vision) and run score endpoint.
2. Verify supported pooling tasks match model task requests.
3. Run EAGLE speculative path with Cohere models.
4. Run guided generation with both tool grammar and structural tag JSON schema.
5. Run multimodal spec decode script using local media path.
