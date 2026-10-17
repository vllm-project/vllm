# Training-side model support

How to build a training implementation cheaply and prove it computes the same math as inference. Where training weights come from (master -> training forward weights) is in [Precision, quantization and QAT](precision-quant.md).

## Three routes

The routes differ in where parallelism and kernels come from. In all three we write the model.

| | MCore (Megatron-Bridge) | FSDP2 (VERL's HF interface path) | mLite |
| --- | --- | --- | --- |
| Positioning | Mature TP/PP/EP/CP, main production RL path | Cheapest: write only the model, no parallelism | Built for consistency: forward calls vLLM kernels, backward reuses the training implementation |
| Forward source | MCore kernels (TE, grouped GEMM) | Our PyTorch model to vLLM semantics, in the Transformers `PreTrainedModel` interface for the VERL loader and PEFT | vLLM kernels (`visible_forward(visible, native, ...)`) |
| Achievable consistency | BI=0, via R3 + importance sampling | BI=0 | BI=1, logprob raw bytes identical |
| What to write for a new model | Bridge weight mapping + modules MCore lacks (Mamba/SSM, special norms) | Whole model: config class, checkpoint loading (with dequantization), per-layer forward, per-operator precision contract; autograd backward, filled in where missing | One `model/<name>/`: config, checkpoint mapping, assembly, protocol; modules from `primitive/modules` |
| Typical pitfalls | Mapping checks values, not semantics; MCore and vLLM math differ (MoE activation clamp) | bf16 autocast around the VERL forward rewrites module precision; `--system-site-packages` ABI; PyTorch reference (Mamba scan) too slow; loading a third-party implementation from the image by mistake | Private modules inside the model directory, bypassing the shared skeleton |

Choose after the [module inventory](#module-inventory-and-placement-rules), by: is BI=1 needed, how many backwards are missing, what parallelism is needed. Nemotron looked cheap by name; FA4, Mamba scan and W4A16 MoE all lacked a backward.

## Find by symptom

| Symptom / question | Where to go |
| --- | --- |
| New model loads and config parses, but unclear whether it can train | [Module inventory](#module-inventory-and-placement-rules) |
| Want to reuse vLLM forward code, but tangled in paged KV / cudagraph / scheduling | [Call kernels only](#reproducing-the-forward-call-kernels-only-do-not-port-the-scheduler) |
| Weight mapping shapes are right, output silently degrades | [Mapping checks values and semantics](#43-mapping-checks-cover-values-and-semantics-not-only-shape) |
| Inference kernel has no backward; should we write one | [Three-case decision](backward-validation.md#three-cases) |
| Wrote a VJP; how to prove it is right | [Module-level comparison steps](backward-validation.md#comparison-steps) |
| Unit tests pass but real training fails / step 0 occasionally not bitwise | [Two test-infrastructure false negatives](backward-validation.md#two-test-infrastructure-false-negatives); [BI kernels](bi-kernels.md) |
| Changes after modifying packing or CP/TP/EP; wrong only on long sequences | [Layout and length](#distributed-layout-length-boundaries-and-training-side-acceptance) |
| `ppo_kl` non-zero after rollout switches parallel layout (e.g. DP+EP to TP1 per GPU) | [Bitwise identity after changing the parallel layout](#bitwise-identity-after-changing-the-parallel-layout) |
| `modified by an inplace operation` after no longer recomputing old-logprob | [Entropy when old-logprob is not recomputed](#entropy-when-old-logprob-is-not-recomputed-bypass) |
| How to hook into VERL / Megatron | [Framework integration](#verl--megatron-integration) |
| Which kernel an operator uses in BI mode and its cost | [BI kernels](bi-kernels.md) |

## Key rules

1. Semantics follow the vLLM implementation and the checkpoint.
2. Before starting, list for every module whether its inference kernel has a backward.
3. Call kernels only; do not port the scheduler. mLite (BI=1): FP32 logprob raw bytes vs vLLM-captured trajectories, target 0 diff. FSDP2 / MCore (BI=0): no 0-diff target; judge each side against an independent reference, then look at precision.
4. Mapping checks cover values and semantics, not only shape; fused parameters are overwritten whole.
5. The precision contract lives in the model. Where vLLM computes in FP32 (position-encoding angles, router logits and bias, norm statistics, etc.), the training module sets the dtype and disables the outer autocast. Under VERL's bf16 autocast, casting only the inputs of matmul-type operators (including `F.linear` and `einsum` decomposed into bmm) does not stop the downgrade. Measure output dtype and values under the real autocast.
6. Backward follows the formula, not the forward kernel. Only rounding differs -> reuse. Quantization rounding -> fix the STE contract, reuse BF16 GEMM. No whole-segment backward -> write glue, GEMM still TE. "No backward kernel at the same precision" is not a reason to rewrite.
7. Reuse model-agnostic operators: mLite `primitive/modules` and `ops` first, then MCore/TE/mamba_ssm; only then is it new code.
8. Backward need not be bitwise; report bitwise and tolerance separately; the reference builds its graph independently with ATen autograd; tolerances are fixed before seeing results; synthetic inputs are labeled as not real activations.
9. Do not recompute activations; use the forward's saved intermediates (DS4: gate_up / LSE).
10. The model directory holds only config / checkpoint / model / protocol; reusable parts go into `primitive/`; no same-named `primitive/` under the model directory; impl name `lite`.
11. Long-sequence boundary points (KV page, packing prepend, the 7000/9216 tiers) need dedicated tests.
12. MTP block, EPLB redundant experts and draft are explicitly skipped in mapping and sync.

## Module inventory and placement rules

List per module: inference kernel, same-math training implementation, quantization rounding, discrete selection, backward. Then pick the route and each module's place in the code tree. This catches missing backwards before they cost weeks.

Training support means an implementation we maintain that backpropagates correctly. Config registration, same-family fallback, the vLLM inference implementation and third-party implementations in the image (newer Transformers modeling, NeMo AutoModel) do not count. Write it per [Reproducing the forward](#reproducing-the-forward-call-kernels-only-do-not-port-the-scheduler) into real source files and verify by loading through the actual loader in a new process. A REPL-only implementation or runtime monkeypatch is not evidence.

### 1. How to build the inventory

One row per module, every column filled; write unknown, not "should exist".

| Column | Question to answer |
| --- | --- |
| Inference kernel | The function vLLM actually calls (not the model-file class name), with quantization variants, attention backend, cudagraph branch |
| Same-math implementation in the training framework | mLite `primitive/modules`, `ops`; MCore / TE; mamba_ssm, causal_conv1d. Rounding-only differences count as same math; a different activation clamp does not. Third-party same-model implementations are not entered |
| Quantization rounding | FP8/NVFP4 rounding in the forward; weights and activations separately |
| Discrete selection | top-k expert IDs, sparse indices; IDs fixed and not differentiated, the continuous score needs a derivative |
| backward | Does the library have one; do its intermediates (LSE, rstd, token permutation) match what our forward saves |
| Conclusion | Reuse / glue / new code ([decision rules](backward-validation.md#three-cases)) |

Nemotron-H:

| Module | Inference kernel | Same-math training implementation | Rounding | Discrete | backward | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| Mamba SSD scan | `mamba_chunk_scan_combined_varlen` | mamba_ssm | None | None | mamba_ssm `_mamba_chunk_scan_combined_bwd` | Reuse |
| conv1d | `causal_conv1d_fn` | causal_conv1d | None | None | Yes; dW needs fixed-order FP32 | Reuse + small glue |
| Attention (FP8 KV) | FA4 fixed split + FP8 Q/K/V | FA4 BF16 | Yes | None | FA4 BF16 backward only | Dequantize, recompute BF16 O/LSE, reuse |
| Mamba in/out_proj (FP8), shared expert (NVFP4) | FlashInfer / NVFP4 GEMM | TE `general_gemm` | Yes | None | TE | Fix STE, reuse |
| Routed experts (NVFP4 W4A16) | CuTe-DSL / Humming MoE | TE grouped GEMM | Yes | Yes | No whole-segment backward | Glue `routed_vjp` |
| RMSNorm / gated RMSNorm, routing score | BI kernel / grouped top-k | PyTorch formula | None | None for the score | autograd | Reuse |
| MTP block, draft | Not part of training | - | - | - | - | See §4 |

### 3. Placement rules

#### 3.1 mLite (Megatron `experimental/lite/megatron/lite`)

Read `experimental/lite/docs/porting.md` and `architecture.md` first. Three layers: `runtime` (lifecycle, training-step orchestration), `model` (registration + model-specific build/load/export protocol), `primitive` (reusable: parallel state, TP layers, checkpoint conversion, MoE utilities, optimizer wrappers). The runtime knows no model details: it gets the protocol from the registry, builds a typed config, and lets the protocol build the model.

| Rule | Content |
| --- | --- |
| The model directory holds only four things | `model/<name>/`: config, checkpoint mapping (WeightSpec), assembly, protocol. Impl name `lite` (`<name>/lite/`) |
| Modules go into primitive | norm / linear / router / experts / dispatcher / attention -> `primitive/modules`; generic VJP bridge -> `primitive/ops`; quantization encoding -> `primitive/quantization`; PP splitting, thd, CP -> `primitive/parallel`; HF read/write via `primitive/ckpt/hf_weights` (`SafeTensorReader`, `load_hf_weights`, `export_hf_weights`) |
| Model-specific math still goes into primitive and is registered | e.g. Mamba2 bridge, FP8-KV attention -> `primitive/modules/<x>.py`; no same-named `primitive/` under the model directory |
| No file for reused backward | Reusing TE / mamba_ssm backward means no `backward.py`; glue goes in the module file |
| Dependencies are one-directional | `model/` imports `primitive/`, never the reverse; top-level checkpoint/quantization does not import the impl subtree |
| Layering guards must be registered | Add the model to `tests/unit/runtime/test_layering_contracts.py` or the guard does not apply |
| No hybrid architecture packages | porting.md: "Keep these out of the first PR: Hybrid model implementation packages"; hybrid (Mamba + attention + MoE) modules are split into primitive |
| State third-party kernel dependencies | Document hard dependencies on vllm, flashinfer private APIs, mamba_ssm, causal_conv1d, and the `init_batch_invariance()` call in build_model |

DS4 is the positive example: `model.py` (650 lines) assembles te.RMSNorm, CSA, HCA/mHC, thd and the fp8 recipe with three model-specific deviations; `moe.py` (102 lines) = SigmoidTopKRouter + Experts + TokenDispatcher + SwiGLUMLP, hash-route the only model-specific code; `checkpoint.py` is one WeightSpec using shared load/export.

#### 3.2 MCore / Megatron-Bridge

- Bridge `models/<name>/`: provider + HF<->MCore mapping; modules MCore lacks (Mamba/SSM variants, special norms, HC) go into `megatron/core`, not Bridge.
- Mapping checks values, not only shape ([§4.3](#43-mapping-checks-cover-values-and-semantics-not-only-shape)).
- MCore and vLLM may use different formulas (MoE activation clamp); BI=0 cannot align those either, so list them first.
- The Megatron fork works only with Megatron-Bridge v0.6.2, pinned in the image.

#### 3.3 FSDP2 (VERL's HF interface path)

- Write a PyTorch model to vLLM semantics in the Transformers `PreTrainedModel` / config interface so the VERL loader, PEFT and HF save/load use it directly. "HF" is only the interface and weight format.
- Delivery: the model goes into the verl trunk, registered to the Auto classes (`AutoConfig` / `AutoModelForCausalLM`, etc.), and ships with the recipe as one verl PR ([Code and delivery §8](../practices/code-and-delivery.md#verl-a-thin-recipe-not-a-second-training-framework)). The verl FSDP extension guide's advice against copying modeling into `verl/models` targets models already in transformers. Register at the current checkout's FSDP load chain (`get_hf_auto_model_class` -> `from_pretrained`).
- Develop and deliver from the same verl source and registration entry. Do not modify the checkpoint's `config.json` / `auto_map` (the low-precision checkpoint is the single source of truth) or enable `trust_remote_code`. The load log must show our registered class was used.
- VERL's automodel engine (`automodel_actor`) uses NeMo AutoModel and is not this route; routes are distinguished by forward source, not by FSDP2 sharding.
- Under bf16 autocast, FP32 computations (position-encoding angles, router logits and bias, norm statistics, following vLLM) disable autocast inside the module and set the dtype.
- Backward prefers autograd; add operators next to the model file only when a backward is missing or too slow (the 160-line PyTorch Mamba scan reference).
- Common pitfalls: `--system-site-packages` ABI, the actual `use_remove_padding`, PEFT targets not going through nn.Linear.
- Under EP4 + FSDP2, all ranks' DCP expert shards share key and offset, so restore reads another rank's experts; the key must carry the EP rank ([checkpoint guide](checkpoint-resume.md)).

### 4. Parts that do not take part in training

Each gets its own inventory row:

| Part | Mapping | Facts |
| --- | --- | --- |
| MTP block | Not loaded for training; WeightSpec lists it as explicit skip, not missing-key tolerance; `_ignore_unexpected_suffixes` etc. only for listed suffixes | Nemotron `num_nextn_predict_layers: 1`, MTP unquantized bf16; usable with BI=0, then in both arms; BI=1 + MTP requires speculative-verification logprob bitwise equal to the full-segment forward |
| EPLB redundant experts | Training has logical experts only; physical-to-logical mapping is inference-side | Completeness counts logical experts, not physical tensors |
| draft / speculative model | Independent checkpoint and mapping | Shared weights (embedding, lm_head) state their owner |

Skipped parts appear in the load log and manifest as "explicitly skipped", not "no error reported".

### 5. Example: reuse audit of Nemotron-H against mLite

Nemotron-H is 3824 lines. From `primitive/` it reuses only protocol-layer orchestration utilities, `load_hf_config_dict` and `vocab_parallel_entropy`; every modeling module is rewritten under `model/nemotron_h/vllm/primitive/`.

| File (`N = model/nemotron_h`) | Lines | What it does | Verdict |
| --- | --- | --- | --- |
| `N/config.py` | 155 | HF config dataclass, validation | Keep (model dir) |
| `N/quantization.py` | 207 | NVFP4/FP8 requantize, E2M1 grid, FP8 encoding, hand-read index.json | Numerics -> `primitive/quantization/nvfp4.py` (parameterize E2M1 RNE vs mxfp4 ties-down); IO via `SafeTensorReader` |
| `N/checkpoint.py` | 486 | HF name mapping, Export spec, BF16 master invertibility check, θ0 threshold, own `load_hf_weights` | Ordinary tensors via `load_hf_weights` + WeightSpec; keep master check and θ0 threshold; importing the impl subtree is a layering inversion |
| `N/vllm/model.py` | 188 | Layer/Model assembly, PP stage | Keep; PP split via `build_pipeline_chunk_layout` |
| `N/vllm/protocol.py` | 409 | protocol, contract validation, BI check, FSDP2/dist_opt, post-step requant | Keep; hard-coded 52-layer pattern and (2688,128,1856,6) make it single-checkpoint |
| `N/vllm/primitive/dense.py` | 647 | visible forward + BF16-master VJP bridge; RMSNorm VJP; Humming/CuTe NVFP4 linear; projection factory; FP8/NVFP4 TrainingLinear | Bridge and RMSNorm VJP -> `primitive/ops/aligned_vjp.py`; `native_linear_vjp` duplicates `_te_general_gemm`; serving kernel adaptation stays |
| `N/vllm/primitive/logprob.py` | 36 | Chunked lm_head + vLLM `compute_token_logprobs` + FP32 log_softmax STE | Keep; the docstring's DS4 function does not exist |
| `N/vllm/primitive/attention/*` | 231 | `Fa4Fp8KVAttention` (FlashAttnFixedSplit 6768-token page, FP8 Q/KV), `_Fp8AttentionVJP` | Model-specific; reusable once GQAttention supports `rotary=None, qk_norm=False` |
| `N/vllm/primitive/mamba/*` | 316 | `_PackedConvVJP`, `_PackedScanVJP`, `SSMMeta`, `MambaMixer` | Keep both VJPs; generic Mamba2 -> `primitive/modules/mamba2.py`; derive `SSMMeta` from `PackedSeqParams`/`ThdPackMeta` |
| `N/vllm/primitive/moe/module.py` | 185 | `_FixedRouteVJP` + Router, `_CombineVJP`, ReLU² SharedExperts | Router VJP equals `SigmoidTopKRouter`, reusable, not hooked to router_replay; add an activation argument to `mlp.py` for SharedExperts |
| `N/vllm/primitive/moe/grouped.py` | 705 | FlashInfer W4A16 signature/sha pinning, `CuteDslRoutedExperts`, `Nvfp4ExpertWeights`, `_te_grouped_gemm` rewrite, ReLU² `routed_vjp` | Keep W4A16 forward + grouped VJP; TE grouped GEMM wrapper duplicated; `ep4_routed_experts` hard-codes 4 ranks |
| `N/vllm/primitive/moe/communication.py` | 251 | DeepEP normal-mode EP, `reduce_ep4_parts` (FlashInfer one-sided combine order), `EPRoutedExpertsVJP` | Use `_build_deepep_buffer`; keep serving-order reduction, parameterize EP size/topk |

Six placement violations: impl name `vllm` not `lite`; hybrid architecture package; same-named `primitive/` in the model layer; circular imports between `checkpoint.py`/`quantization.py` and the impl subtree; `nemotron_h` missing from the layering guard; third-party kernel dependencies and `init_batch_invariance()` undocumented.

Should have been reused: safetensors index reading (5 copies); HF load of ordinary tensors; DeepEP buffer; TE GEMM wrapper (2 copies); visible forward + reference VJP bridge; Router (`SigmoidTopKRouter` + `group_limited_topk`, forward switched to vLLM grouped_topk, hooked to router_replay); shared expert; E2M1 grid and FP8 encoding; PP splitting; `SSMMeta`; two docstrings citing nonexistent DS4 functions.

Model-specific, keep: the two Mamba2 VJPs + packed conv/scan; `Fa4Fp8KVAttention` + FP8 attention VJP; W4A16 MoE forward + ReLU² grouped VJP; serving-order reduction (parameterized); Humming/CuTe NVFP4 and FP8 serving linear; vLLM forward-value logprob; BF16-master invertibility check and θ0 threshold; ModelOpt 4over6 requantize rule.

## Reproducing the forward: call kernels only, do not port the scheduler

The first deliverable after the inventory is a training forward with vLLM's math. Logprob measurement basis and R3 route replay: [workflow](../workflow/README.md), [rollout engine](rollout-engine.md).

### 1. First separate the responsibilities of the model implementation and the training framework

A training framework (FSDP2, MCore, mLite runtime) shards a trainable model; it does not make an inference-only kernel differentiable or write the model. Write a minimal PyTorch model for the current loader interface covering the config, checkpoint loading, forward, gradients and save interface actually used; full Transformers integration, generation and all parallelism are not needed first. The semantic basis is vLLM and the checkpoint, not a third-party implementation in the image.

| Capability | What the training route still has to do | What in inference / Megatron is not copied |
| --- | --- | --- |
| Model math | Real semantics of attention/SSM, norm, position encoding, routing, residuals, special embeddings | vLLM paged KV, decode scheduling, cudagraph management; MCore provider/ModuleSpec wiring |
| Backpropagation | Input gradients of every training-path operator; trained-parameter gradients; recompute correctness ([Backward validation](backward-validation.md)) | TP/SP/CP-specific collectives and gradient reduction |
| Weights | Names, axes, fused order, expert IDs, sharing, derived state | HF<->Megatron shard layout, PP export protocol |
| LoRA | Target coverage, A/B direction and scaling, non-zero adapter taking effect | Parallel linear wrapper, TP adapter aggregation |
| Memory | Initialization peak, large frozen table residency, save/export peak | Megatron memory saver, PP scheduling |

dtype, dequantization, QAT and update format: [precision guide](precision-quant.md). Here each module states its input/output precision and differentiable boundary.

### 2. How to wire kernel-only calls

Each layer's forward runs once: call the vLLM kernel for the visible output, bitwise identical to inference, and save what the backward needs; see [backward wiring](backward-validation.md#how-to-wire-it-forward-once-backward-uses-what-the-forward-saved). Borrow only the kernel, no vLLM runtime object:

- No KV paging, decode scheduling or CUDA Graph management. Training runs a full-sequence teacher-forced forward; packing passes boundaries via thd/cu_seqlens.
- Reuse order: model-agnostic differentiable operators -> concise PyTorch reference to vLLM semantics -> ready-made operators with backward -> custom autograd/kernel only if necessary. This order only says where to look first; the training-step cost of each choice is checked at C6 against the C1 budget, not deferred to C10.
- Handle discrete top-k / index selection apart from the selected differentiable tensors. What is trained, frozen or given an auxiliary loss follows the model definition; do not detach to get it running or add router/indexer objectives.
- vLLM layer modules used directly (FusedMoE, ModelOpt linear) need `get_tp/ep_group`, so the training process creates vLLM process groups (Nemotron does; DS4's own modules do not).
- Take the same branch as inference: reproduce the quantization config, attention backend and BI-switch kernel selection, or the comparison is against a different path. Tests use the production `CompilationConfig(custom_ops=...)` ([the two false negatives](backward-validation.md#two-test-infrastructure-false-negatives)).

"The same kernel" still has pitfalls (Nemotron 7-layer localization):

1. Direct and compiled paths of the same gated RMS kernel differ; calling it through an opaque custom-op boundary removed the difference.
2. A differing reduction block was wrongly blamed alone; the same captured kernel showed both blocks match.
3. The last 4.768e-7 logprob difference came from the HF forward calling `.float()` after lm_head; keeping the real BF16 head output and sharing the scoring fixed it. This is not truncating an FP32 projection to BF16 to manufacture equality.

### 3. Read the inference implementation along the full support chain

Order: model file (layer types and order, residual and norm positions, position encoding, attention visibility, router/indexer, special embeddings) -> each layer's actual kernel and quantization variant -> attention backend and KV dtype -> weight loader (name mapping, fused order, derived scale) -> BI-switch branches. Confirm the parameter/buffer list from the same checkpoint/config.

Sizes sharing one implementation need no separate port. For ordinary dense models, look for a reusable generic skeleton first; we still write the model. Multimodal is validated per task; diffusion is out of scope.

### 4. The minimal implementation that really has to be filled in

#### 4.1 Reproduce the math

Confirm from the checkpoint/config and inference code: layer types and order, parameters/buffers, residual and norm positions, position encoding, attention visibility, router/indexer, special embeddings. Model-specific math (`sqrt(softplus)` scoring, SwiGLU with clamp, mHC Sinkhorn, ReLU² experts) is in no library; write it from the inference source, not the paper.

#### 4.2 A frozen base still has to propagate input gradients

```python
W.requires_grad_(False)
y = linear(x, W) + scale * B(A(x))  # no gradient for W, but still propagate to x
loss(y).backward()
```

Wrapping a frozen block in `no_grad()` cuts LoRA gradients of earlier layers; only a frozen table lookup needing no input/parameter derivative is exempt. With initial B=0 a zero gradient for A can be normal; test with a non-zero adapter or a real step. A finite loss does not prove backward.

<a id="33-shape"></a>

#### 4.3 Mapping checks cover values and semantics, not only shape

Per weight class: source name -> training name -> serving name, axis transforms, sharing. Focus on fused QKV, gate/up order, expert numbering, extra norm/bias, tie weights, non-persistent buffers, derived state (FP8 KV scale, NVFP4 alpha). Every missing/extra item needs a reason; quantization-derived items follow the precision guide and need not be byte-identical to the low-precision source. FSDP needs no HF<->Megatron bridge but may need checkpoint->training and export mappings.

#### 4.4 Packing boundaries for Mamba / stateful modules

Every stateful operator (conv1d, SSM scan, recurrence, PLE shift) must consume sequence boundaries; a correct attention mask proves nothing about them. Incidents:

- verl via Megatron-Bridge did not pass boundaries to Mamba; sequence states mixed and logprob was silently wrong, visible only as a larger mismatch.
- Qwen3.8 Flash-Next's PLE shift made the second sample reference the first sample's last token.

Check: unequal-length A/B run separately and merged; sentinel-initialized outputs to catch missed writes. In mLite, derive boundary metadata from `PackedSeqParams`/`ThdPackMeta`, not a separate `SSMMeta`. See also [Long-sequence boundary points](#long-sequence-boundary-points).

#### 4.5 Recompute and small-model trimming

With activation checkpointing on, verify recompute reads the same state and gives reasonable gradients. Trim to the smallest layer set covering every non-equivalent path and its dependencies, usually 4-6 layers; keep mixed attention, cross-layer sharing and layer-specific table lookups. Record kept layer indices and the config transform; a slice proves only covered paths, not full-model mismatch or score gain.

### 5. How to prove it computes the same math

Feed the training forward vLLM-captured trajectories (fixed tokens, weight version, position, mask, routing); never compare probability summaries or argmax. mLite (BI=1): FP32 logprob raw bytes, target 0 diff. FSDP2 / MCore (BI=0): judge each side against an independent reference, then check the remaining difference comes only from precision rules.

- Same policy weights. Differing dtypes are a contract mismatch; do not convert to fake bitwise.
- Reference artifact (Nemotron): frozen trainer image, full actor, 2048 tokens, FP32 0 diff; five real updates, 883,998 tokens of identical FP32 logprob.
- Old-logprob recomputation and the grad-enabled forward are two paths; evidence for each ([dual-path contract](../workflow/validation.md#old-logprob-and-training-forward-dual-path-contract)).
- Find the first divergence per [debugging](debugging-mismatch.md): per-module dump, single-request prefill, then the operator.

### 6. Small tasks handed to agents and stop conditions

Each outputs a short conclusion, source version, reproducible command and artifact.

| Task | Minimal acceptance | If it fails |
| --- | --- | --- |
| Implementation inventory | Training model class, operator backward, checkpoint loader; gaps listed | Write the minimal model; stop cold-starting the large model |
| Module comparison | Same weights/inputs; gradients vs a differentiable reference | Locate the earliest difference |
| Few-layer training | Non-zero loss, finite gradients, intended adapter changes | Separate broken path, operator bug, gradient-less target |
| Save and restore | Fresh-process restore: consistent forward and next step | Check state, adapter, optimizer, residency |

Distribute only after the first two pass; scale up after the few-layer two-step loop passes.

### 7. How to judge development cost

Training implementation exists: wire loading, LoRA, sharding. Inference only with differentiable references: fill in assembly, mapping, interfaces. Key operator lacks backward or clear semantics: separate blocker; reference and gradient validation first; no uniform person-day figure.

## Distributed layout, length boundaries and training-side acceptance

Record trainer and rollout topologies separately. TP/CP/EP, padding, packing and collectives change data mapping and reduction paths; long sequences hit untested boundaries; forward 0 diff does not vouch for gradients or the optimizer.

### Compute resources first, then request them

Write trainer and rollout process groups separately. At four GPUs per node, an inference DP8/EP8 group spans two nodes; two DP4/EP4 replicas are not one EP8 group and skip cross-node EP communication. The runtime decides grouping.

If dense DP excludes CP: `W = TP × PP × CP × DP`. Expert side: verify `W = ETP × EP × PP × EDP` and the CP mapping separately. **Do not multiply EP into the dense world again.** W=8, TP1/PP2/CP2 gives DP=2; a log calling DP×CP "DP" shows 4. Clarify naming and members; do not change GPU count to match a config string.

GPU memory follows the layout. With PP1 each rank holds all layers' activations, linear in sequence length. Per-route EP backward state stored for other ranks' slots grows by the average ranks per token; store only own routes. Rollout `gpu_memory_utilization` is set from what training still holds during rollout, per training mode, in the recipe ([Memory budget 02 §5](checkpoint-resume.md#5-gpu-memory-set-gpu_memory_utilization-per-mode-when-training-and-rollout-are-colocated)).

### Packing and CP

For the target UID reconstruct global request `[seq_start, seq_end)`, rank range `[rank_start, rank_end)`, local rows, cu_seqlens, positions, all compression/reordering mappings.

The intersection `[max(seq_start, rank_start), min(seq_end, rank_end))` contributes rows only when start < end. The overlap window must not give a disjoint request this rank's output. Host cumulative boundaries equal the rows the device compactor produces.

Cover: before/after/at the boundary, empty intersection, cross-rank, compression-ratio and tile remainders. Window constant 8: check 1…7, not only multiples. Start from a hand-computable counterexample, then enumerate nearby legal configs.

Passing `cu_seqlens` does not mean a fallback consumes it ([§4.4](#44-packing-boundaries-for-mamba--stateful-modules)).

### EP and collectives

Record expert mapping, tokens per rank, padding capacity, route order, shared/routed combine, quantization scales. Empty ranks may still join collectives; partially written reused buffers leave stale data. Use different sentinels per side; zero-filling both gives a false green.

Pinning NCCL algorithm/channels may remove reduction differences on one stack and cut throughput significantly; verify it per topology, not as a universal env list, before relaxing config or building a deterministic collective.

MCore dev's grad-ready assertion takes first-batch hook counts as golden and errors on a different count. Short-sequence micro-batches can make counts data-dependent (Nemotron r10 at step 2; r12 did not reproduce). With dp×cp=1 and expert_dp=1, `overlap_grad_reduce` off avoids it without changing values; root cause unknown. With EP4 + dist_opt and dp>1, multi-micro-batch reduction fires early (reported).

### Bitwise identity after changing the parallel layout

**Principle.** Floating-point addition is not associative. Layout sets reduction grouping, order and rounding: TP splits a matmul into per-rank partial sums; EP computes a token's top-k experts on different ranks and merges. Last-bit differences become different tokens after dozens of layers and sampling, so bitwise identity holds only for the verified layout; any TP/EP/DP change must be proven again.

**Rules.**

1. The reduction tree is part of the numerical contract: per level, accumulation dtype, order and rounding points. All legal rollout layouts and training walk the same tree.
2. A degenerate layout (TP=1, EP=1) must simulate the tree: per-rank partial sums, same rounding, same merge order.
3. After a layout change, compare per-token logprob raw bytes on fixed inputs before training. Non-zero `ppo_kl`/probs_diff is fixed, not avoided by reverting the layout.
4. Locate: dump per layer, find the first differing layer, confirm its inputs (MoE: router logits too) are bitwise equal; the bug is that layer's reduction ([Fault localization](debugging-mismatch.md#bisect-and-first-divergence)).
5. Measure the cost of simulating the order and report it apart from the layout gain.

**Examples.**

| Model | Layout change | Where the difference is | Fix and result |
| --- | --- | --- | --- |
| Nemotron-H (W4A16 MoE, BI=1) | rollout from DP4EP4 per node to TP1 per GPU | EP4: per-rank FP32 sums rounded to BF16, merged in FP32 as `((s0+s1)+(s2+s3))+(s4+s5)`. EP1: six outputs summed sequentially in FP32, no intermediate rounding. Max logprob diff 0.4; greedy tokens diverge at positions 10–129 | Without EP, BI mode computes all experts in one launch and reduces in EP4 order (vLLM `661cde6833`, not in the image). Both layouts match training bitwise; `ppo_kl` 0 over 58 steps; multi-turn replay 404 s vs 485 s, reduction cost about 2.5% |
| DeepSeek V4.1 | training TP=1, inference TP>1 | order of per-TP-rank partial sums | both sides use one fixed reduction tree, TP=1 included (`VLLM_TP_INVARIANT_LEAVES`) |

### Asynchronous communication and lifetime

Check streams, events, buffer owners and reuse timing. Passing with a global synchronize only proves a timing issue and is not the fix; find the missing dependency and accept on the same path.

### Long-sequence boundary points

Test at real lengths:

| Boundary | Fact | How to test |
| --- | --- | --- |
| KV page | training attention replicates vLLM's KV page size; page boundaries are numerical boundaries | sparse dY across the page boundary |
| Fixed split attention | fixed split count and keys per split; the resulting `MAX_SEQ_LEN` is an invariance bound, not a performance knob | above: error, not silent truncation; below: two real lengths × random/zero/sparse dY |
| Packing prepend | one prepended short sequence shifts every target row offset | two prepend controls vs none, bitwise |
| Micro-batch fits the longest sequence | verl needs each micro-batch to fit the longest sequence (`ppo_max_token_len_per_gpu`, Nemotron 16384 vs DS4 4096; Nemotron training has no CP) | OOM probe at formal length; do not extrapolate from 8K |
| Embedding backward | CUDA `embedding_dense_backward` accumulates at reduced precision with ≤3072 indices, e.g. a short sequence alone in a micro-batch | length-tiered tolerance |
| CP compressed boundary | see [Packing and CP](#packing-and-cp) | before/after/at the boundary |

<a id="attention-d"></a>

### The D term in attention backward

With P rounded to FP8, visible O = fp8(P)·V ≠ softmax(QKᵀ)·V. If `D = rowsum(dO⊙O)` uses visible O while dS uses recomputed P, `rowsum(dS_i) ≠ 0` and dQ gains a same-sign per-row bias (Nemotron: about 3% of dQ/dK, v2 FAIL at steps 150/250; fixed 0.006–0.009, PASS). Mainstream FP8 attention (TE fp8_dpa, SageBwd) uses visible O, so a human decides after being informed.

Four checks:

1. **FA4 interface signature**: `_flash_attn_bwd` derives D in preprocess from the `out` passed; D cannot be passed. The public interface errors under `requires_grad`; the actor calls `_flash_attn_fwd/_bwd`. Fix: recompute BF16 O and LSE from dequantized Q/K/V, pass to bwd (layer backward +4–15%, end to end <0.05%).
2. **varlen LSE layout**: ln vs log2, per head vs per token, softmax scale folded or not. On mismatch, recompute with the training kernel.
3. **GQA 32/2**: dK/dV summed within each Q head group; A/A needs fixed summation order.
4. **When the saved Q/K/V are released under recompute=full**: if only dequantized Q/K/V are saved, recompute must read the same state; early release changes the values.

### Microbatch split and gradient difference

Different microbatch splits give about 2e-3 relative gradient difference (loss normalization, accumulation order, reduced-precision embedding backward at ≤3072 indices). Not a bug, but:

- Criterion denominator = `max(B16 reference floor, noise floor of the implementation under test)`, never the FP32 ideal.
- Both arms use the same split; if it changes, rerun A/A.
- grad_norm is not comparable across different microbatch counts.

### Control matrix

With policy/tokens fixed, test original layout, reversed order, target first/last, changed micro-batch, CP-boundary sample, each reconstructed by UID/global token coordinates. Real TP/EP changes may legally change the reduction; the experiment contract decides whether cross-topology bitwise equality is required.

Safe packing is not a fix for a metadata bug; it may only lower trigger probability. After the fix, the original layout and nearby boundaries must pass.

<a id="v2-optauth"></a>

### Training-side correctness acceptance: gradient comparison criterion v2 and optauth

Two independent checks beyond forward 0 diff.

#### Gradient comparison criterion v2

- Freeze the criterion (tolerance, denominator, thresholds) before the run, record its sha, never change it after.
- Planted-error check first: gradients ×1.25 and ×1.05 must both FAIL.
- Accept on several snapshots (steps 0/50/150/250); the D-term problem appeared only at step 150.
- On failure, inject reference gradients block by block to find the first module over the limit.
- Extended to FSDP2: EP4 + FSDP2 all four snapshots PASS, worst-class median 1.08–1.39, all floors from B16.
- Tool bug: with `overlap_grad_reduce=False` gradients were reduced again before the EP4 dump (fixed). The export path needs A/A too: ‖g‖ bitwise equal, local shards bitwise equal to the dump, replica diff 0/0.
- Cross-process bitwise backward is not required. An occasional non-bitwise first step comes from the thread-local Inductor deterministic config (the autograd thread's first compile picks 512/1024 blocking by timing); fix in [BI kernels: determinism](bi-kernels.md#determinism-takes-effect-only-on-the-main-thread).

#### Optimizer acceptance (optauth)

Verify the optimizer consumes clipped `main_grad`, the update lands on master, and deployed bytes = requantize(master). Freeze the criterion first. Split failures into criterion modeling gaps (e.g. a check inapplicable without replicated parameters; the equal-magnitude first Adam step invalidating γ-type criteria while the per-coordinate bound holds), documented and accepted, and implementation errors, fixed. Cover all parameter group types: a first-step probe without an FP32 wd_mult=0 parameter misses weight decay applied to that group.

#### Facts outside the two criteria

- Frozen base with LoRA: base modules still propagate input gradients ([§4.2](#42-a-frozen-base-still-has-to-propagate-input-gradients)).
- The two-step RL loop (θ0→θ1→θ2) accepts the forward; three-arm SFT accepts training effect; report all three separately ([validation](../workflow/validation.md)).

## VERL / Megatron integration

Upstream config keys may differ from what was used here. Find the actual call paths of rollout dump, old-logprob and skip/repeat in the current checkout and record the commit; local-patch options are not upstream API. Process content (recipe and PR contract, review material, two-step acceptance, E2E targets) is in [validation.md](../workflow/validation.md#recipe-review-materials-and-two-step-acceptance-agreement); PR ownership in [Code and delivery §8](../practices/code-and-delivery.md#8-three-repo-delivery-from-experimental-implementation-to-three-prs-and-one-image).

### Where the integration layer is

Megatron's `experimental/lite/examples/verl/verl_mlite/` provides the verl engine adapter (`model_engine=mlite`, `pkg://verl_mlite.config`), installable into the unified image without reimplementing the engine in verl. Check engine registration, model factory, batch conversion, loss normalization and optimizer path; list reusable interfaces and gaps. Consistency uses mLite; the training baseline uses the existing recipe's default MCore path; do not mix them.

### Capture location

Save the DataProto (or equivalent) after rollout returns and before the first old-logprob; append trainer scores after. Record policy version, UID, input/response IDs, mask, position IDs, reward, route replay data if needed, and the UID -> packed-row mapping if reordered. Keep a pre-update checkpoint bound to the batch by content fingerprint; after strict load compare structure and keys and reject missing/unexpected keys.

### Historical DS4 entry point

DS4 used patched VERL options (`skip.rollout.*`, `DS4_SKIP_UPDATE=1`); confirm in code and logs that any such switch takes effect.

### Minimal adapter contract

The synchronous entry point takes batch, checkpoint, variant and output directory, then:

1. Records actual policy, effective config, libraries and parallel layout.
2. Reads the same rollout, repacks per variant, keeps logical UIDs.
3. Runs the real old-logprob forward and exports per-token logprobs in UID order ([Saving and comparing](debugging-mismatch.md#compare-per-token-logprobs)).
4. Records no generation and no update, and mutable-buffer initialization.
5. Waits for workers/collectives, closes its resources, returns an exit code.

It keeps native `no_grad`/`forward_only` semantics and replaces neither the [grad-enabled forward comparison](../workflow/validation.md#old-logprob-and-training-forward-dual-path-contract) nor update acceptance. A/B/C may share one long-lived worker once weight and state isolation is verified. The runner does not take over Ray/Slurm; the launcher blocks on remote jobs and reclaims on timeout. For injected or overlaid adapters, record the actual class and source file and assert the callsite was hit; `PYTHONPATH` can shadow the overlay.

### Entropy when old-logprob is not recomputed (bypass)

Bypass uses rollout logprob as old-logprob: `actor/ppo_kl` then measures rollout vs training forward exactly (0 under BI=1), and entropy comes from the grad-enabled update forward.

- **Put metric-only quantities under `no_grad`.** With `entropy_coeff=0` the loss still holds `0 * entropy_loss`, so entropy's backward runs and saves a logits copy.
- **Chunked entropy in the verl Megatron engine cannot backpropagate.** With `entropy_from_logits_with_chunking=True`, each chunk view's backward overwrites the shared saved logits in place; the shared version counter then raises `one of the variables needed for gradient computation has been modified by an inplace operation`. Hit whenever chunking meets entropy. Fix: [verl-project/verl#8195](https://github.com/verl-project/verl/pull/8195); until merged, with `entropy_coeff=0` compute entropy under `no_grad` with the same chunking.
- **Re-verify gradients after changing forward computation.** Nemotron chunked logprob: BF16 bitwise equal to unchunked; FP32 max diff about 1e-8 at max gradient about 4.

### Training extension

Pairing a shared inference forward with a native VJP: check recompute, CP metadata lifetime, PP payload, parameter registration, through the standard training entry point. After a real update, export a new rollout and trainer scores; old bundles replay only with the old policy, and new-vs-old output differences are expected, not a consistency failure. Modules using vLLM layers need vLLM process groups. `param_offload=True` changes the quantized master's `data_ptr` and fails the pre-export check; use optimizer offload only.
