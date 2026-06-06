# Code Notes: Models, Pooling, and Inference Extensions

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

`examples/features/speculative_decoding/spec_decode_offline.py` now supports:

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

`RoutingMethodType` (in `fused_moe/config.py`) gains a `SigmoidRenorm = 6`
variant. The dispatch function `get_routing_method_type` returns it when
`scoring_func == "sigmoid"` and `top_k > 1` (i.e. anything that is not the
Llama4 single-expert sigmoid path).

### Custom routing function

`commandr.py` defines `token_choice_with_bias`:

- casts router logits to float32,
- applies `sigmoid` (not softmax),
- selects top-k experts,
- optionally renormalizes weights by dividing by their sum (controlled by
  `renormalize` / `norm_topk_prob`).

The function is registered via `custom_routing_function` on `FusedMoE` when
the HF config has `expert_selection_fn == "sigmoid"`. At runtime,
`CustomRoutingRouter.routing_method_type` detects this function and returns
`RoutingMethodType.SigmoidRenorm` so kernel selection can match.

### `norm_topk_prob` on `FusedMoEConfig`

`FusedMoEConfig` gains `norm_topk_prob: bool = True`. This flag is read from
the HF model config via `getattr(config, "norm_topk_prob", True)` in
`commandr.py` and propagated through `FusedMoE.moe_config`. It is threaded
into every FlashInfer / TRT-LLM kernel call site:

- `flashinfer_trtllm_moe.py` (`flashinfer_fused_moe_bf16`),
- `experts/trtllm_fp8_moe.py` (per-tensor, per-block, and c4/c5 paths),
- `experts/trtllm_nvfp4_moe.py`,
- `unquantized_fused_moe_method.py` (bf16 FlashInfer path).

When `norm_topk_prob` is `False`, the kernel skips the post-top-k weight
renormalization step. This matches model configs where sigmoid routing
weights are used as-is without rescaling.

### Kernel selection impact

`_supports_routing_method_bf16` in the FlashInfer/TRT-LLM oracle explicitly
includes `RoutingMethodType.SigmoidRenorm`, enabling the fused kernel path
for Cohere MoE models. Without this, sigmoid-routed models would fall back
to slower non-fused dispatch.

### Change hotspots

- `vllm/model_executor/layers/fused_moe/config.py` — enum, config field
- `vllm/model_executor/models/commandr.py` — `token_choice_with_bias`, FusedMoE wiring
- `vllm/model_executor/layers/fused_moe/router/custom_routing_router.py`
- `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`
- `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py`
- `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py`
- `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`
- `vllm/model_executor/layers/fused_moe/layer.py`

### Validation checklist

1. Load a Cohere MoE model with `expert_selection_fn = "sigmoid"` and verify
   routing selects the `SigmoidRenorm` path (check debug logs or breakpoint
   `get_routing_method_type`).
2. Toggle `norm_topk_prob` between `True` / `False` in HF config and verify
   output logits differ (renorm vs raw sigmoid weights).
3. Confirm FlashInfer bf16 and fp8 fused kernels are selected (not fallback)
   for sigmoid-routed models.

## 7) Secondary Performance-Oriented Deltas

Additional but relevant model/runtime deltas:

- expanded fused MoE configs for newer GPUs,
- compressed tensor / Marlin FP4 utility updates,
- benchmark support for Cohere2MoE in kernel benchmark script.

These are mostly enablement and tuning changes; they are easy to overlook during rebase because they live outside "obvious model files".

## 8) compressed_tensors_moe (MoE scheme matching + W4A8 load path)

Upstream v0.21 split MoE dispatch into
`compressed_tensors/compressed_tensors_moe/` (router in
`compressed_tensors_moe.py`, WNA16 Marlin in
`compressed_tensors_moe_wna16_marlin.py`). Cohere-specific logic from the
legacy monolithic `compressed_tensors_moe.py` must be ported into those
package files on every upstream sync — not only the top-level shim.

### MoE quantization scheme equality

`CompressedTensorsMoEMethod.get_moe_method` requires identical scheme dicts
across the unfused projection names (gate/up/down). Some checkpoints label
different projections with different `actorder` metadata even when the on-disk
tensor layout matches: for example `actorder="weight"` vs `"static"` after
llm-compressor inverse-permutes weights to natural order and omits
`weight_g_idx`. `_normalize_weight_actorder` maps those cases to
`actorder=None` on a **copied** `QuantizationArgs` (the original dict / args
may be shared or cached elsewhere). Only `ActivationOrdering.WEIGHT` is
normalized; other actorder values pass through unchanged. Without this step,
mixed-metadata MoE layers (e.g. up AWQ vs down GPTQ-static) can spuriously
fail the “All MoE projections need to have same quantization scheme” check.

### WNA16 Marlin `is_k_full` under TP

`CompressedTensorsWNA16MarlinMoEMethod.create_weights` must use Cohere's
`is_k_full` rule: only `actorder="group"` needs full-K w2 scales and
`is_k_full=False` under TP. Upstream's `(not self.actorder)` check treats
`actorder="weight"`/`"static"` as falsy-for-`is_k_full` and, with TP>1,
derives an effective Marlin `group_size` of 16 (`size_k / num_groups`) that
the kernel does not support — server startup fails with
`Invalid thread config ... group_size = 16, is_k_full = 0`.

### W4A8 weight post-processing

`CompressedTensorsW4A8Fp8MoEMethod.process_weights_after_loading` rewrites
packed-uint4 weights and bf16 groupwise scales into the layout expected by the
CUTLASS grouped GEMM. Two contracts in this path are easy to regress on rebase:

- **Sync between in-place int4 convert and reorder.** For both `w13` and
  `w2`, `convert_packed_uint4b8_to_signed_int4_inplace(...)` writes the
  weight buffer in place and `ops.cutlass_encode_and_reorder_int4b_grouped(...)`
  reads the same buffer. A `torch.cuda.synchronize()` between them is
  required — without it the reorder kernel can see pre-conversion data
  and produce silently wrong weights at load time. One-off load cost, no
  steady-state impact; do not remove.
- **`view(*orig_shape[:-1], -1)` in `convert_bf16_scales_to_fp8`**
  (`quant_utils.py`). The channel-scale reshape must unpack
  `orig_shape[:-1]`; passing the tuple directly
  (`view(orig_shape[:-1], -1)`) raises `TypeError`.

Validation: load a C4 W4A8 MoE checkpoint and run the
`C4-*-W4A8-*` lm-eval configs under `.buildkite/lm-eval-harness/configs/`
— a missing sync typically surfaces as degraded eval scores rather than
a crash.

## 9) Change Hotspots and Validation

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

## 9) LoRA Serving for `cohere2moe` (Cohere MoE Models)

LoRA adapter serving for `Cohere2MoE` is implemented in `cohere2_moe.py` (registry loads that module for `Cohere2MoeForCausalLM`). `commandr.py` still contains an unused duplicate `Cohere2Moe*` stack from pre-v0.21. Shared layer/config fixes:

- **`_CONFIG_REGISTRY` entry** (`vllm/transformers_utils/config.py`): `cohere2moe` maps to the string `"Cohere2Config"` for lazy resolution via `LazyConfigDict` (see `vllm/transformers_utils/configs/__init__.py` `_CLASS_TO_MODULE`), avoiding `NameError` when `VLLM_USE_MODELSCOPE` is enabled.

- **`hf_to_vllm_mapper`** (`cohere2_moe.Cohere2MoeForCausalLM`): strips the `backend.model.` prefix that the Faraday training framework adds when saving LoRA checkpoints, so vLLM can find weights at `layers.X...` paths.

- **`get_expert_mapping()`** (`cohere2_moe.Cohere2MoeModel` and `Cohere2MoeForCausalLM`): exposes `FusedMoE.make_expert_params_mapping(self, ...)` so vLLM can build the correct expert-weight → param-name mapping for MoE LoRA. The `self` positional argument is required by the v0.17.1 signature.

- **LoRA base-layer unwrap** (`commandr.Cohere2MoeDecoderLayer` only; upstream `cohere2_moe` decoder uses a simpler residual path): `getattr(self.mlp.experts, "base_layer", self.mlp.experts)` before `must_reduce_shared_expert_outputs()` when TP MoE all-reduce is enabled.

- **`lora_extra_vocab_size` on `LoRAConfig`** (`vllm/config/lora.py`): the fork adds this field (default `0`) so `Cohere2MoeModel` / `Cohere2MoeForCausalLM` can size embedding tables consistently with other LoRA-enabled models without silent `getattr` fallbacks.

- **Packed module validation** (`vllm/lora/worker_manager.py`): when a module appears in `packed_modules_mapping`, both the unpacked component names *and* the packed name itself are added to `expected_lora_lst`, so checkpoints saved with either naming convention (e.g. `qkv_proj` vs individual `q_proj`/`k_proj`/`v_proj`) are accepted.

- **Double-replacement guard** (`load_weights`): the `stacked_params_mapping` loop skips remapping when the target `param_name` is already present in `name` (e.g., `qkv_proj` contains `v_proj` as a substring), preventing incorrect double-replacement.

- **Dummy LoRA for CI** (`tests/cohere/scripts/create_dummy_lora.py`): when the c5 checkpoint is saved as a `cohere2_vision` top-level config, read attention dimensions from `text_config` (same keys as `cohere2moe`). vLLM maps `base_model.model.model.layers...` LoRA weights onto `language_model.model.layers...` via `parse_fine_tuned_lora_name` and `Cohere2VisionForConditionalGeneration.hf_to_vllm_mapper`.

- **`lm_head` skip on vision load** (`cohere2_vision.Cohere2VisionForConditionalGeneration.load_weights`): use `AutoWeightsLoader(self, skip_prefixes=["lm_head"])`. Quantized vision checkpoints (int4/fp4) can still ship `lm_head.*` tensors even though the wrapper has no top-level `lm_head` module (logits go through the nested `language_model` with tied embeddings). Upstream v0.21 uses plain `AutoWeightsLoader(self)`; the fork must keep this skip inside `# cohere start/end` or merge resolution will drop it.

## 10) Optional FP32 Final-Logits Projection

Cohere branch adds `VLLM_USE_LOGITS_FP32_COMPUTATION` (default **off**; set to
`1` or `true` to enable) to `vllm/envs.py` and uses it in
`vllm/model_executor/layers/logits_processor.py`.

Behavioral intent:

- keep full-model execution dtype unchanged,
- run final logits projection with the native operand dtype (for example bf16)
  while materializing logits outputs in fp32 when the LM head exposes a dense
  weight tensor.

Current constraint:

- quantized/custom lm-head implementations that do not expose a dense
  `weight` tensor continue to use their existing quantized projection path.
- multimodal wrappers such as
  `Cohere2VisionForConditionalGeneration` route final logits through a nested
  `language_model`, so runtime inspection/hooks that look for
  `logits_processor` or LM-head weights must walk through that wrapper instead
  of assuming those objects live on the top-level module.

## 11) Pre-quantized MXFP8 via compressed-tensors (temporary port)

Adds support for serving pre-quantized **MXFP8** (E4M3 weights + E8M0 `uint8`
per-group scales, `group_size=32`) checkpoints through the compressed-tensors
backend, for both dense linear layers and MoE layers. Activations are
dynamically quantized to MXFP8 at runtime via the existing `Mxfp8LinearOp`
(FlashInfer-CUTLASS or emulation backend). This complements the online-MXFP8
path added in `#665`, which quantizes bf16/fp16 weights at load time; the new
path loads already-quantized weights directly.

### Status: temporary port of upstream vllm-project/vllm#38815

This is a manual port, not a `git cherry-pick`, because two upstream refactors
that `df1e30e74` depends on are not yet in this fork:

- **vllm-project/vllm#39205** (`MxFp8LinearKernel` refactor). Upstream's scheme
  file imports `init_mxfp8_linear_kernel()`, which does not exist here. The
  cohere port wires directly to `Mxfp8LinearOp` instead, mirroring
  `Mxfp8OnlineLinearMethod` in
  `vllm/model_executor/layers/quantization/online/mxfp8.py` (moved from the
  former single-file path as part of the new online-quant frontend; see
  section 12).
- **vllm-project/vllm#39187** (CT MoE "Oracle Structure" package split).
  Upstream places the new MoE method at
  `compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_mxfp8.py`.
  This fork still has `compressed_tensors_moe.py` as a single file, so
  `CompressedTensorsW8A8Mxfp8MoEMethod` is appended in-place instead.

**Drop condition.** Once this fork merges upstream past `df1e30e74` (which
requires `#39187` and `#39205`), delete this port and take upstream's version.
At that point the directory layout and kernel-init API will match upstream,
and `df1e30e74` applies cleanly. The scheme file becomes redundant with
upstream's `schemes/compressed_tensors_w8a8_mxfp8.py`; the MoE class is replaced
by upstream's `compressed_tensors_moe/compressed_tensors_moe_w8a8_mxfp8.py`.

### Code layout

- `schemes/compressed_tensors_w8a8_mxfp8.py` *(new)* — `CompressedTensorsW8A8Mxfp8`
  linear scheme. Loads `float8_e4m3fn` weights + `uint8` scales, swizzles the
  scale for FlashInfer-CUTLASS in `process_weights_after_loading`, and dispatches
  to `Mxfp8LinearOp.apply`.
- `compressed_tensors_moe.py` — appends `CompressedTensorsW8A8Mxfp8MoEMethod`
  (structurally a twin of `CompressedTensorsW8A8Fp8MoEMethod`; reuses the
  already-present `select_mxfp8_moe_backend` / `make_fp8_moe_kernel` /
  `convert_to_fp8_moe_kernel_format` oracle helpers from `#665`).
- `compressed_tensors.py` — `_is_mxfp8` detection (group-quant, 8-bit float,
  `scale_dtype=uint8`, `group_size=32`, symmetric) + scheme dispatch.
- `schemes/__init__.py` — export.

### Rebase guidance

All port sites carry `# cohere: #38815` inline markers or `# cohere start` /
`# cohere end` block markers, plus `Upstream-Commit:` /
`Drop-After-Upstream-Merged:` trailers in the file / class docstrings. To find
everything at rebase time:

```bash
git grep -nE '# cohere.*#38815|Upstream-Commit:.*df1e30e74|Drop-After-Upstream-Merged:' \
  vllm/ tests/
```

### Validation checklist

1. Scheme + MoE method dispatch: `pytest tests/quantization/test_compressed_tensors.py::test_compressed_tensors_mxfp8_moe_setup -v` (Turing+ required; uses dummy weights so no real download).
2. Accuracy: `lm_eval --model vllm --trust_remote_code --model_args pretrained=AliEdalati97/Qwen3-30B-A3B-MXFP8 --tasks gsm8k,mmlu_pro --batch_size auto`. Upstream PR reports GSM8K strict-match 88.9, MMLU-Pro 68.8 vs bf16 89.4 / 69.3.
3. Throughput: confirm FlashInfer-CUTLASS backend is selected (check `logger.info_once("Using %s backend for MXFP8 GEMM", ...)` output) — emulation fallback is materially slower.
4. Coexistence with `#665` online path: a bf16 checkpoint served with `--quantization mxfp8` should still take the online path; only checkpoints whose `compressed_tensors` config has `type=float, num_bits=8, group_size=32, scale_dtype=uint8` should route to this new scheme.

## 12) Online Quantization Frontend + Cohere `config.json` Path

Online quantization (quantize bf16/fp16 weights at load time, no
pre-quantized checkpoint) now has a dedicated frontend package under
`vllm/model_executor/layers/quantization/online/`:

- `base.py` — `OnlineQuantizationConfig` (the `QuantizationConfig` subclass
  registered as `"online"`); dispatches per-layer to the per-scheme methods
  in `get_quant_method`.
- `fp8.py` — `Fp8PerTensorOnline{Linear,MoE}Method`,
  `Fp8PerBlockOnline{Linear,MoE}Method`.
- `mxfp8.py` — `Mxfp8Online{Linear,MoE}Method` (moved from the former
  single-file `vllm/model_executor/layers/quantization/mxfp8.py` in #40152).
- `int8.py` — `Int8OnlineMoEMethod` (consolidated from `experts_int8` in
  #38463; weight-only per-channel, MoE experts only — linear layers stay
  unquantized).
- `moe_base.py` — `OnlineMoEMethodBase`, the meta-device + QeRL
  `initialize_online_processing` shared base for all online MoE methods.

### Two entry points into the same config

`OnlineQuantizationConfig` accepts the same scheme set
(`OnlineQuantScheme` in `vllm/config/quantization.py`:
`fp8_per_tensor`, `fp8_per_block`, `mxfp8`, `int8_per_channel_weight_only`)
through two distinct call sites:

1. **CLI / `OnlineQuantizationConfigArgs`** (upstream path, exercised by
   [`tests/quantization/test_online.py`](../../../tests/quantization/test_online.py)):
   `--quantization {fp8_per_tensor|fp8_per_block|mxfp8|...}` flows through
   `resolve_online_quant_config` in `vllm/config/quantization.py`, which
   normalizes `--quantization` + `quantization_config` into a single
   `OnlineQuantizationConfigArgs` and rejects mismatched
   `quantization` / `global_scheme` pairs.
2. **Cohere `from_config`** (cohere-marked block in
   [`vllm/model_executor/layers/quantization/online/base.py`](../../../vllm/model_executor/layers/quantization/online/base.py)
   L91-149, exercised by
   [`tests/cohere/cpu/test_online_quant_from_config.py`](../../../tests/cohere/cpu/test_online_quant_from_config.py)):
   loads the config from a checkpoint's `config.json`
   `quantization_config` block. This path is what `model.config.json` driven
   workflows hit; the upstream CLI path bypasses it.

### `from_config` schema (cohere path)

```jsonc
"quantization_config": {
    // dispatch — either "online" (with explicit overrides below) or one of
    // the scheme shorthands. Shorthand auto-populates global_scheme.
    "quant_method": "fp8_per_block",

    // optional. Module names; "re:" prefix opts into regex matching
    // (semantics from compressed_tensors.should_ignore_layer).
    "ignore": ["re:.*self_attn\\..*", "model.layers.0.mlp.experts"],

    // optional aliases of "ignore" — merged in declared order
    // (back-compat with Mxfp8Config / HF / modelopt).
    "ignored_layers": [...],
    "modules_to_not_convert": [...],

    // optional per-layer-class overrides; either may be set without
    // global_scheme.
    "linear_scheme_override": "fp8_per_block",
    "moe_scheme_override":   "fp8_per_tensor",

    // optional. Only "dynamic" is accepted; anything else raises ValueError.
    "activation_scheme": "dynamic"
}
```

Validation invariants enforced by `from_config` + `__init__`:

- At least one of `global_scheme`, `linear_scheme_override`,
  `moe_scheme_override` must be set, else `ValueError("global_scheme...")`.
- `quant_method` shorthand never clobbers an explicit `global_scheme`.
- `activation_scheme != "dynamic"` raises `ValueError("activation_scheme...")`.

### Per-layer dispatch

`OnlineQuantizationConfig.get_quant_method` resolves
`linear_scheme_override or global_scheme` for `LinearBase` and
`moe_scheme_override or global_scheme` for `FusedMoE`, then routes to the
matching scheme class above. Layers matching `ignored_layers` (via
`compressed_tensors.utils.should_ignore_layer`, with
`packed_modules_mapping` honored) fall back to
`Unquantized{Linear,FusedMoE}Method`. `INT8_PER_CHANNEL_WEIGHT_ONLY` is a
MoE-only scheme — linear layers under that scheme are forced unquantized
with a `warning_once`.

### Change hotspots

- New entries are added by extending `OnlineQuantScheme` and registering a
  new `{Linear,MoE}Method` pair in `online/`; `from_config` automatically
  picks up the new shorthand because it derives `scheme_values` from the
  enum.
- The `skip_with_substr` knob was removed; ignore semantics now go through
  `compressed_tensors.utils.should_ignore_layer` exclusively.
- The cohere `from_config` path is wrapped in `# cohere start` /
  `# cohere end` markers; keep them in place when porting future
  upstream changes to this method.

### Validation

- Parsing path: `pytest -v tests/cohere/cpu/test_online_quant_from_config.py`
  (CPU-only; runs in the `cpu_check` group on every PR).
- Upstream CLI path:
  `pytest -v tests/quantization/test_online.py`.
