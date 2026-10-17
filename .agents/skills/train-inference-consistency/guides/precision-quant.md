# Precision, quantization and QAT

The weight conversion chain from optimizer master to vLLM bytes, QAT and low-precision training, and known pitfalls. Numerical contracts without quantization: [numerical-contracts](../workflow/README.md).

## Find by symptom

| Symptom / question | Where to go |
| --- | --- |
| Weights unchanged, probs_diff grows after export | [P1](#p1-scale-rule-differs-from-the-checkpoint-requantization-flips-codes), [P10](#p10-both-sender-and-receiver-quantize), [P11](#p11-receiver-picks-quantization-targets-by-name-suffix-and-misses-fused-moe) |
| NVFP4/FP8 scale or packed bytes differ from the checkpoint | [Conversion chain §2](#2-inventory-the-checkpoints-full-quantization-rules-first), [P1-P5](#quantization-rules-and-scale) |
| NVFP4 rollout: QAT or BF16 actor? | [QAT §1](#1-choose-the-precision-route-first), [§4](#4-how-much-mismatch-qat-can-remove) |
| QAT is on but the mismatch did not change | [P7](#p7-qat-switch-is-on-target-modules-were-not-replaced), [P8](#p8-training-grid-differs-from-the-deployment-grid) |
| QAT loss does not drop, deployed weights byte-for-byte unchanged | [QAT §6](#6-why-qat-does-not-train), [P24](#p24-master-starts-on-the-quantization-grid) |
| Whether to add activation fake quantization | [QAT §4.1](#41-activation-fake-quant-aligned-per-layer-worse-end-to-end) |
| Same operator input, inference module output differs | [P13-P15](#training-side-numerics) |
| Fixed logprob offset under FSDP | [P16](#p16-fsdp-bf16-policy-overrides-the-fp32-parameter-contract), [P20](#p20-dividing-by-temperature-in-bf16) |
| Some LoRA/adapter gradients are always zero | [P17](#p17-fake-quant-cuts-off-the-gradient) |
| Adding a new quantization format | [Conversion chain §3](#3-what-each-format-outputs), [§5](#5-minimal-adaptation-for-a-new-quantization-format) |
| Implementing or changing backward/STE | [Backward validation](backward-validation.md) |

## Key rules

1. "NVFP4/FP8" is only a format name; a different block scale selection, global scale convention, rounding or static/dynamic activation gives different bytes. Inventory the checkpoint's full rules first.
2. Use the rules of a mainstream implementation (TE, ModelOpt, NeMo-RL) that matches the checkpoint; do not invent your own. Verify from the real BF16 release, not from dequantized weights.
3. Training QAT and the exporter use the same quantization function on the same grid (range, axis, scale, rounding).
4. Compute scale on the full logical tensor. A fusion group shares scale (VERL NVFP4: QKV, gate/up fused global scale). Quantizing each shard separately and concatenating is forbidden.
5. Each tensor family has exactly one conversion owner and is quantized exactly once; post-load reordering does not count as quantization.
6. Activation scales (`input_scale`, `input_amax`, `k_scale`, `v_scale`) are calibration values. Pass them through unchanged; they cannot be derived from weights. Read runtime scales from the actual runner.
7. High-precision training + low-precision rollout is not QAT. Neither fake-QAT nor real low-precision GEMM implies a low-precision optimizer or low-precision gradients.
8. Weight QAT lowers Nemotron-H probs_diff mean from about 0.025 to about 0.010; it reaches 0 only when the training forward uses the same kernels as inference. Whether activation fake-quant helps depends on upstream mismatch; measure first.
9. STE backward has its own acceptance: an independent reference, the same dY, a tolerance agreed in advance. A forward 0 diff does not prove the gradient is right.
10. Check that deployed bytes = `requantize(master)`.
11. Start the master from the real BF16 release, not a dequantized checkpoint, whose values sit on grid points so NVFP4 deployed values only change after thousands of steps at small lr. θ0 bytes then differ from the official checkpoint (Nemotron-H: NVFP4 99.776% identical, FP8 24/46 tensors differ); quantify the cost with proxy SFT and have a human confirm it ([QAT §6.2](#62-fix)).

## Terms

- **NVFP4**: FP4 E2M1 values, one FP8 E4M3 block scale per 16 elements, plus one FP32 global scale.
- **block scale / global scale**: per-block scale (`weight_scale`, `scale`) / per-tensor or per-fusion-group scale (`weight_scale_2`, `weight_global_scale`, `scale2`).
- **amax**: maximum absolute value of a tensor (or block), used to compute scale.
- **4over6 / MSE**: TE NVFP4 block scale selection settings; the Nemotron-H checkpoint uses them.
- **E4M3 / E2M1 / UE8M0**: FP8 (4 exponent, 3 mantissa bits) / FP4 (2 exponent, 1 mantissa bit) / unsigned 8-bit exponent (power-of-2 scale).
- **W4A16 / W4A4 / W8A8**: weight bit width / activation bit width.
- **QDQ**: quantize then immediately dequantize, giving a high-precision value on the grid.
- **fake-quant / fake-QAT**: the training forward uses the QDQ value while GEMM stays high precision; not low-precision training.
- **STE**: straight-through estimator; treats the derivative of rounding as 1 (or with a clipping mask).
- **master**: high-precision parameters managed by the optimizer.
- **conversion owner**: the single party that converts a tensor family to the inference format.
- **fusion group**: tensors fused in serving (QKV, gate/up) that must share scale.
- **R3**: MoE route replay; training uses the expert selection recorded by rollout.
- **BI=1 / BI=0**: the `VLLM_BATCH_INVARIANT` switch; 1 is the true on-policy arm where training calls the inference kernels, 0 is the regular baseline.
- **probs_diff**: verl's `training/rollout_probs_diff_*`, the token probability difference between rollout and the training forward.
- **TIS / `rollout_is`**: rollout importance sampling correction.
- **θ0**: policy weights before the first update.
- **oracle**: the reference implementation or value that decides pass or fail.
- **VJP**: the backward function that computes input and parameter gradients from the upstream gradient.

## Conversion chain: from master weights to the bytes the vLLM kernel uses

### 1. Four stages

```text
A. master weights     FP32/BF16 optimizer master (plus the FP32 exceptions the model declares)
   ↓ forward materialization
B. weights the training forward actually sees
                      BF16; under QAT it is DQ(Q(W)), the dequantized value on the deployment grid
   ↓ export (gather → HF mapping → quantize/pack)
C. inference-format bytes
                      packed NVFP4 + block scale + global scale / FP8 + scale /
                      INT4 packed + scale / layers still BF16 in mixed precision
   ↓ loader → post-load
D. tensors loaded by vLLM and used by the kernel (including post-load reordering and derived scales)
```

| Stage | Invariant | How to verify |
| --- | --- | --- |
| A | Record dtype separately for parameters and buffers; an FP32 master does not mean forward uses FP32 | Read optimizer parameter identity and dtype; see [§6](#6-dtype-contract-six-boundaries) |
| B | Under QAT, `Q` is the same function on the same grid as the export in C | Unit test: weights of B = dequantize(bytes of C), bitwise |
| C | Scale computed on the full logical tensor; fusion group shares the global scale; excluded layers keep their dtype; recomputing on the same master is idempotent | Compare packed bytes, block scale, global scale per tensor; see [§2](#2-inventory-the-checkpoints-full-quantization-rules-first) |
| D | Each family quantized once; post-load only reorders, does not recompute scale; calibration values arrive unchanged | Per tensor, compare exported bytes with the bytes vLLM uses after loading |

Record the real format and scale of weights, activations and KV (packed UE8M0, FP32 scale, MXFP4/FP8 scope); activation and packing rounding may differ between SM90 and SM100. Before measuring quantization error, confirm the scale comes from the current weights, not an old tensor.

### 2. Inventory the checkpoint's full quantization rules first

1. From `hf_quant_config.json` and the `quantization_config` in `config.json`, list each layer's format, which layers carry `input_scale`, and whether the KV cache carries k/v scale. Read layer by layer; do not infer from the model name ([P12](#p12-inferring-a-layers-precision-from-the-models-overall-name)). Nemotron-H Lightning 30B-A3B: `MIXED_PRECISION`; 46 FP8 layers (`mixer.in_proj`/`out_proj` of 23 Mamba layers); 5935 `W4A16_NVFP4` layers (routed/shared experts, lm_head); FP8 KV cache; 46 fp32 scalar `input_scale`, FP8 layers only.
2. Encode the BF16 release with the checkpoint's own scales; 100% identical packed bytes confirm the release matches the checkpoint.
3. From the **real BF16 release**, requantize with the candidate rule and report per tensor the match ratio of packed bytes, block scale and global scale. Dequantized weights already sit on the grid and pass under any rule ([P2](#p2-verifying-a-rule-with-dequantized-weights-gives-a-false-positive)). Measured match rates: [P1](#p1-scale-rule-differs-from-the-checkpoint-requantization-flips-codes), [P3](#p3-fp8-per-tensor-scale-convention-is-not-amax448).

Decisions:

- Call the rules of a mainstream implementation (TE, ModelOpt, NeMo-RL) directly; do not invent a scale policy. Deviate, and report it, only when the mainstream implementation cannot do it. Nemotron's self-invented "enlarge scale only on overflow" was replaced by an existing TE rule.
- θ0 deployed bytes = the real BF16 release requantized by the rule, which differs from the official checkpoint: on Nemotron-H, NVFP4 bytes are 99.776% identical and FP8 24/46 tensors differ; the proxy SFT cost is about 0.03 nat ([QAT §6.2](#62-fix)). This policy needs human confirmation.
- The BI=1 training forward, BI=0 QAT and the exporter call the same `requantize`; θ0 bytes of the two arms are identical. `input_scale`, `k_scale`, `v_scale` are calibration values passed through from the checkpoint.
- When a single FP8 checkpoint is the source of truth, the training base is a deterministic dequantized derivative, and requantizing need not restore the checkpoint byte for byte. Measure separately: (1) with zero updates, re-exported bytes equal θ0; (2) the precision difference from the original checkpoint.

### 3. What each format outputs

| Format | Artifacts | Must check |
| --- | --- | --- |
| block FP8 | FP8 weight + `weight_scale_inv`, scale per block | Commonly 128×128; FP32/UE8M0 branch, backend, exclusion scope |
| per-tensor FP8 | FP8 weight + `weight_scale` | Scale from the amax of the full weight, not of a shard; convention may not be amax/448 |
| MXFP8 | E4M3 weight + UE8M0 `weight_scale_inv` | 1×32 block direction, scale encoding, pad and high-precision exceptions |
| NVFP4 W4A16 (compressed-tensors, existing `fp4` sender) | `.weight_packed` + `.weight_scale` + `.weight_global_scale` | 1×16, two-level scale, shared outer scale for gate/up, activation contract |
| NVFP4 W4A4 (ModelOpt, Inkling) | packed `uint8` under the original parameter name + FP8 `scale` + FP32 `scale2` + BF16 `input_amax` + `original_shape` | Naming not interchangeable with compressed-tensors; needs a separate adapter |
| INT4 compressed-tensors | `weight_packed` + `weight_scale` + `weight_shape` | group_size, symmetric quantization, pack layout, list of tensors actually quantized |
| DeepSeek V4.1 attention FP8 | F8_E4M3 weight + F8_E8M0 `.scale`, 32×32 block | Not rowwise 32-element MXFP8 |

`weight_scale_inv` is only a loader protocol name; interpret the value per implementation.

### 4. Who converts: conversion owner

A tensor family has one conversion owner. Once the sender has packed, the receiver must not quantize again. dtype reinterpret, layout staging and post-load repack are not quantization.

| Path | Training | Export | Rollout load | owner |
| --- | --- | --- | --- | --- |
| FP8 rollout-only | BF16 | Full BF16 | `quant_weights()` converts to block-FP8 | rollout receiver |
| VERL FSDP QAT NVFP4 | BF16 + fake quant | After gathering the full tensor, `QATQuantizer` outputs packed NVFP4/scale | compressed-tensors | FSDP sender |
| VERL Megatron QAT NVFP4 | BF16 + ModelOpt fake quant | `QATWeightExporter` | ModelOpt | Megatron sender |
| Megatron FP8 E2E | TE FP8 | Bridge-specific FP8 export planning | vLLM FP8 | Bridge / rollout, per the pinned recipe |
| Nemotron-H BI=0 (verl Megatron) | BF16 + ModelOpt weight STE | Requantized by the checkpoint's rules into ModelOpt `MIXED_PRECISION` | ModelOpt | sender (`mixed_precision_export=True`) |
| Pre-quantized checkpoint passthrough | No requantization | Original tensor/scale | Per the checkpoint layout | checkpoint producer |

- Full merge: convert logically complete HF tensors one at a time by their owner; no whole-model all-gather or per-step disk write.
- QAT: the training forward simulates quantization; export generates the actual packed weights and scales.

#### 4.1 FSDP path (VERL)

- With `actor_rollout_ref.actor.fsdp_config.qat.enable=true`, quantizable `nn.Linear` modules become `QATLinear` after the PEFT wrapper and before the FSDP wrap. Forward does NVFP4 W4A16/W4A4 fake quant and ends in `F.linear`; after the optimizer step, `invalidate_all_scales` clears the weight scale cache. [Implementation][qat-linear]
- Export: `FSDPEngine.get_per_tensor_param()` calls `full_tensor()` on each `DTensor`, then `QATQuantizer.quantize_with_fusion()`:

```text
FSDP local shards → full_tensor() all-gather → full HF weight
→ full-weight amax / global_scale → QKV or gate/up fused_global_scale
→ per-row, per-group block scale → packed NVFP4 weight + scales
```

  The cost is the all-gather and GPU staging. The quantizer caches to CPU per decoder layer and moves back to GPU, so peak GPU memory is not necessarily one tensor. [Export][qat-export]

- `get_per_tensor_param_shard()` outputs only the local BF16 shard, with no quantizer and no scale. Shard-local block quantization is an unsupported new protocol and must at least solve:
  1. aligning shard and block boundaries, or handling cross-shard blocks;
  2. a collective for the full-weight amax/global scale;
  3. a consistent fused global scale for QKV and gate/up;
  4. W4A4 explicitly carrying the activation-derived `input_global_scale`, which cannot be rebuilt from weight shards.

  Until these contracts and the oracle exist, quantizing each shard and concatenating is forbidden.
- Non-QAT: export BF16 in HF coordinates; BF16 rollout does not quantize, and `rollout.quantization=fp8` is quantized by the receiver. LoRA merge exception: [P6](#p6-lora-mergetrue-bypasses-the-fsdp-qat-export).

#### 4.2 Megatron path

- VERL: ModelOpt installs fake-quant modules; `MegatronEngine.get_per_tensor_param()` exports HF weights through Megatron-Bridge, and `QATWeightExporter` reads the module quantizer's amax/block metadata to produce NVFP4 and scales. [ModelOpt exporter][modelopt-source]
- VERL FP8 E2E: TE handles FP8 forward/backward in training; when the Bridge declares `export_weight_dtype=fp8`, VERL does not cache generic conversion tasks. The rollout format is decided by `rollout.quantization` and the serving checkpoint contract; FP8 training does not mean exported tensors are in vLLM layout.
- Megatron capability cannot be inferred as FSDP capability.
- Distinct artifacts: the offline converter prepares the initial checkpoint (may write to disk); the online exporter streams in memory; post-load reorders without recomputing scale; a training checkpoint is not a servable quantized HF checkpoint.

#### 4.3 Receiver quantization (rollout-only)

`rollout.quantization=fp8`: the trainer exports BF16; `load_quanted_weights()` in the vLLM worker calls `quant_weights()`, which uses `scaled_fp8_blockwise()` to produce the FP8 weight and `weight_scale_inv`, then replays the quant method's post-load. There are also a DeepSeek V4-specific entry and an Ascend MXFP8 branch with different scale naming. [receiver][receiver-source] When the checkpoint is already FP8/MXFP4, confirm whether it is requantization or passthrough. vLLM cannot quantize BF16 online into ModelOpt `MIXED_PRECISION`.

### 5. Minimal adaptation for a new quantization format

1. From the checkpoint config and the actual loader, list key, shape, dtype, block size, packing and fusion group of weight/scale, and the unquantized tensor families.
2. Trace export -> loader -> post-load and mark the single owner of each family.
3. If a converter is missing, add the minimal conversion at the existing owner's entry; verify on one tensor / fusion group, then cold-load a small model. Changing the dtype is not a conversion.
4. Test zero merge and a non-zero update; get quantize-after-full-gather right before shard optimizations.

One evidence table is enough: tensor family -> source -> export format -> serving format -> owner -> unit/small-model verification.

#### Minimal acceptance

- [ ] Record training dtype, export dtype/layout, rollout dtype/layout and owner.
- [ ] Fixed-token logprob of the zero-adapter path and the base are within the noise baseline of the same backend.
- [ ] After a non-zero update, tokenwise logprob/KL between trainer and rollout stays within bounds.
- [ ] name/shape/dtype/count/bytes of packed weight, scale and ignored families all close.
- [ ] Deployed bytes = `requantize(master)`.
- [ ] FSDP QAT proves scale is computed after `full_tensor()`; with shard export, prove it is not quantized, or provide the four shard-local contracts of §4.1 and a cross-rank oracle.
- [ ] Each family is quantized exactly once.

### 6. dtype contract: six boundaries

For each model, record `checkpoint → load/bridge → wrapper/sharding → forward → backward/optimizer → export/serving` separately.

| Boundary | Megatron | VERL/FSDP | Must check |
| --- | --- | --- | --- |
| Source weights and conversion | Bridge may reorder, split, cast; model may override default conversion | HF loader, dequantized derived checkpoint, init dtype | Same revision; logical value/scale/layout match, not only target dtype |
| Parameter storage | `params_dtype` plus model exceptions; FP32 master/optimizer | HF keep-precision list, PEFT master, FSDP shard | An FP32 master does not mean forward uses FP32 |
| forward parameters | Float16Module converts selectively | FSDP2 `MixedPrecisionPolicy.param_dtype` | Measure inside forward, not only before the wrap |
| Operator computation and output | TE/FLA/TileLang/fused each define product, accumulation and output precision | HF/PEFT kernel and autocast | FP32 output does not prove FP32 product; BF16 output does not prove BF16 accumulation |
| Gradient and optimizer | TP/EP/SP/CP collective, DDP | FSDP reduce dtype, gradient accumulation | FP32 reduction does not restore values rounded in forward |
| Export and serving | Undo the parallel layout, convert adapter/quantization format | gather/export HF adapter | Compare values on the same logical layout |

FP32 protection takes several forms: mark-to-skip before conversion, reset after declaration (which does not restore values already lost), bypassing the default BF16 downcast, and operator-level selection. It is not "all special parameters are raised in precision".

FSDP: HF `_keep_in_fp32_modules_strict` is a load contract and does not become a per-parameter cast exception in FSDP ([P16](#p16-fsdp-bf16-policy-overrides-the-fp32-parameter-contract)). Respect the model's existing contract first, and verify any change of computation precision against a same-model reference and actual serving. `param_dtype=None`, `param_dtype=fp32` and FP32 autocast are different things.

### 7. Code entry points

FSDP [engine][fsdp-source], [module selection][qat-core], [fake quant][qat-linear], [NVFP4 export][qat-export]; Megatron [ModelOpt exporter][modelopt-source]; rollout [receiver][receiver-source]. Historical PR pointers (not the current state): early return on LoRA merge [#5115](https://github.com/verl-project/verl/pull/5115)/[#5411](https://github.com/verl-project/verl/pull/5411)/[#7014](https://github.com/verl-project/verl/pull/7014); fake quant and export quantization [#7149](https://github.com/verl-project/verl/pull/7149)/[#7224](https://github.com/verl-project/verl/pull/7224)/[#7577](https://github.com/verl-project/verl/pull/7577).

[fsdp-source]: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/workers/engine/fsdp/transformer_impl.py
[qat-core]: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/utils/qat/core.py
[qat-linear]: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/utils/qat/linear.py
[qat-export]: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/utils/qat/quantizer.py
[modelopt-source]: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/utils/modelopt/qat_weight_exporter.py
[receiver-source]: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/utils/vllm/vllm_quant_utils.py

## QAT, low-precision training and STE

### 1. Choose the precision route first

Rule out input, weight, adapter, layout and operator correctness errors first, then choose precision:

1. When GPU memory allows and rollout can use BF16, use BF16 for both training and inference; check FP32 exception parameters one by one.
2. When rollout must be low precision (such as NVFP4), use weight QAT or a BF16 actor for training. On Nemotron-H the weight QAT arm is stable. Whether `rollout_is=token` stabilizes the BF16 actor is **not tested**.
3. Consider real low-precision training last: backward/kernel support and maintenance cost more.

Same dtype is not same computation, and QAT is not bit-exact. QAT and low-precision training shrink the **weight value** gap; the kernel and reduction-order gap goes only when the training forward calls the inference kernels (the BI=1 arm of the [BI switch recipe](recipe-and-acceptance.md)). A natively low-precision model does not necessarily run with `dtype=bf16`; do a capacity pre-check first.

### 2. What the four kinds of path are

| Path | Training forward | backward / optimizer | Is it QAT |
| --- | --- | --- | --- |
| High-precision training + low-precision rollout | BF16 weights | High precision | No; quantized only at export |
| fake-QAT | `DQ(Q(W))`, GEMM is still BF16 | STE sends the gradient back to the high-precision master | Yes |
| Real low-precision GEMM (FP8/MXFP8/NVFP4) | TE changes the compute kernel | High-precision master/exception tensors remain | Different from fake-QAT |
| Frozen quantized base + high-precision LoRA | Dequantized base + adapter | Only the adapter is trained | No |

- Neither fake-QAT nor real low-precision training makes the optimizer/gradients low precision. W4A16 quantizes only weights; W4A4 also quantizes inputs.
- Real low precision: NVFP4 forward is not FP4 backward; configurations include high-precision backward and backward on the dequantized operand.
- Only weight-only formats can compute scale from the full weight; formats with activation quantization also need activation scale/calibration state, which weights cannot provide. QAT simulation and export follow the same format rules, but scales are not necessarily constant during training.
- When LoRA is applied separately, do not fake-quant the merged base+adapter yourself.
- Open-source frameworks (verl [#6992](https://github.com/verl-project/verl/pull/6992), [#7971](https://github.com/verl-project/verl/pull/7971), NeMo-RL) essentially only do W4A16; activation inconsistency only appears at W4A8.

### 3. Implementing fake-QAT

Forward form (Nemotron-H):

```text
w_visible = w + (dequant(requantize(w)) - w).detach()
```

- `requantize` is the same function as the exporter; replace the library's built-in quantizer when its rules differ.
- Requantization runs on every forward, and again under full recompute; Nemotron-H step time is about +60%.
- A temporary `QuantModuleRegistry` replacement and the old class-attribute patch give the same probs_diff mean on the same quick workload (0.0095/0.0091/0.0100 vs 0.0096/0.0092/0.0101); module hit check: [P7](#p7-qat-switch-is-on-target-modules-were-not-replaced).

### 4. How much mismatch QAT can remove

Nemotron-H (`rollout_probs_diff` mean, NVFP4 mixed-precision deployment, single node 4×GB200):

| Approach | Training forward | probs_diff mean |
| --- | --- | --- |
| BF16 training + quantized rollout | BF16 weights | About 0.025; no rollout correction |
| Weight QAT (STE) | Dequantized deployed weights, BF16 GEMM | About 0.010 (pearson 0.986→0.997); three 3-step smoke runs of one recipe, 0.0091-0.0102 |
| Weight QAT + activation fake-quant | Same, plus static FP8 activations | About 0.0107, about 10% worse than weights only |
| true on-policy (BI=1) | Deployed bytes + the same kernels as vLLM | max = mean = 0 at every step for 100 steps |

The QAT arm's probs_diff max is still 0.73-1.0; a small mean does not rule out large per-token deviations. Other models (different metrics, not thresholds): DeepSeek-V4 BF16 train + FP8 rollout has a first-step logprob drift of about 0.023 and still reports rising reward/eval ([DeepSeek][dsblog]); Kimi K3 frozen quantized base + LoRA has a train/rollout KL noise floor of about 0.002, a Schulman KL estimate, not a mean absolute logprob diff ([K3][k3blog]).

#### 4.1 Activation fake-quant: aligned per layer, worse end to end

Adding vLLM's static FP8 activation fake-quant to training aligns single-layer output to the BF16 noise floor, but end-to-end probs_diff gets about 10% worse: FP8 rounding is discontinuous, so any element-wise upstream mismatch flips codes and amplifies the difference. To judge, add relative noise δ only to the training-side input and sweep the output error; the crossover was about δ≈1%, and activation quantization hurts when upstream mismatch exceeds it. This is one measurement on one model, not a threshold; for a new model, measure upstream mismatch and repeat the sweep. Nemotron-H did not merge it.

Rule: the weight-side range/axis/scale/rounding must match serving ([rule 3](#key-rules)); matching the activation side depends on upstream mismatch and is not unconditional. Mainstream frameworks (verl, NeMo-RL, slime) use the same **dynamic** activation rule on both sides and do not reuse an external checkpoint's static activation scales; the NeMo-RL Nemotron recipe quantizes only routed experts and leaves Mamba projections, attention and lm_head in BF16. On DeepSeek V4.1, KV quantization simulation adds only about 1.93% improvement on top of R3. Training-side simulation of the FP8 KV cache is not covered.

#### 4.3 When QAT is worth it

Evaluate unified FP8/QAT only when evidence shows the train-inference quantization difference is the main limit; "no score gain" is not a reason to switch precision first. For DSv4.1 rank2 LoRA, the post-merge quantized delta is only 5.1-7.7% of the pre-quantization delta (a norm ratio, not information retention), which only supports treating step size/quantization as controlled variables. The long-running Qwen Flash-Next experiment is same-source BF16/BF16 and does not validate the FP8 rollout route.

### 5. STE contract

The STE surrogate (identity or clipped), whether dX uses the master or dequantized value, scale handling and verification are in [Backward validation](backward-validation.md#write-down-the-gradient-contract-first). By default dX uses the dequantized value; when using the master value, quantify the bias.

### 6. Why QAT does not train

Train-inference consistency is the strictest QAT: forward uses quantized weights, the high-precision master accumulates updates, STE passes the gradient back. Whether it trains depends only on the master's distance to the nearest decision boundary. In W4A16 (the FP4 code is decoded to BF16 in a register and goes through the BF16 tensor core; the gain is 1/4 the weight bandwidth), W4A4 and W8A8 alike, the computed values are "code × scale".

#### 6.1 Dead zone

- Each NVFP4 element has only 8 values, 0/0.5/1/1.5/2/3/4/6 × scale, with RNE rounding. From a dequantized checkpoint the master sits on grid points, half a spacing from every decision boundary, and must travel that half spacing before the code changes.
- At lr 1e-6 one step moves the master about 0.01%; a few hundred steps move it far less than half a spacing, so deployed values stay byte-for-byte unchanged. RL gradients are more random than SFT, closer to a random walk, and need more steps.
- Result: NVFP4-covered parameters (on Nemotron-H, routed/shared experts and lm_head, the bulk) have zero deployed change, and validation NLL is clearly worse than the native BF16 baseline; freezing these weights explains about half the gap.
- Recomputing block scale does not flip codes earlier: from grid points requant is idempotent, the E4M3 scale only changes code after block amax moves 3-6%, and a tiny global scale change only rescales the whole tensor.
- Mainstream QAT (NeMo-RL QARL, slime/Kimi INT4, Jet-RL) starts from the original BF16 weights, which lie between grid points, so any small drift flips a proportional share of codes and there is no dead zone. INT4 with 1×32 groups from a dequantized checkpoint stalls the same way.
- FP8 is barely affected: floating-point relative spacing is 6-12%, small elements have small absolute spacing, and some elements are always near a boundary. DS4 also starts from dequantized FP8 and is affected only mildly.

#### 6.2 Fix

Preferred: use the original BF16 weights (the real BF16 release) as the master and requantize every step by the same rule (Nemotron-H: TE 4over6, E4M3 upper bound 256). State two points:

- **θ0 is not equal to the official checkpoint bytes.** Requantizing from the real BF16 release, 99.776% of Nemotron-H's NVFP4 bytes are identical and 24 of 46 FP8 tensors (ModelOpt calibrated scale) differ. "The official checkpoint was generated by this rule" holds only on dequantized weights, where MSE and MAE cannot tell rules apart on grid points ([P2](#p2-verifying-a-rule-with-dequantized-weights-gives-a-false-positive)). Proxy SFT gives the cost: 300-step val NLL A-bf16 3.4685 vs C-dequant 3.4641 / C-bf16 3.4373, about 0.03 nat, present at θ0 and not growing during training. This policy needs human confirmation.
- **Three acceptance items** (all pass on Nemotron-H): (1) with zero updates, re-exported bytes equal θ0 (not the official checkpoint); (2) fast-tier RL θ0/θ1/θ2 is 0 diff and deployed bytes start changing; (3) proxy SFT gap to the native BF16 baseline is within budget.

Alternative, keeping θ0 bytes: add a fixed small jitter to move the master off grid points (toward the neighboring cell, at most half a step; after requant check per block that bytes are unchanged, and roll back changed blocks). Not recommended: stochastic-rounding requantization (about 1.4× deployment error, needs a self-written quantizer, no mainstream precedent), raising lr (under Adam equivalent to QuRL UAQ, only shortens the dead zone, 10× lr is outside the usual RL range), and frozen base + LoRA (QeRL, gives up full-parameter updates). Whichever you choose, state that from a dequantized checkpoint at lr 1e-6, NVFP4 deployed values do not change for roughly the first 10³-10⁴ steps. See [P24](#p24-master-starts-on-the-quantization-grid).

[dsblog]: https://www.lmsys.org/blog/2026-04-25-deepseek-v4/
[k3blog]: https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support/

## Known pitfalls

Ordered along the conversion chain: quantization rules -> export and receive -> training-side numerics -> experiment measurement basis. P24 sits under training-side numerics by content.

### Quantization rules and scale

#### P1 scale rule differs from the checkpoint, requantization flips codes

- Symptom: weights unchanged, rollout shifts after a requantized export.
- Cause: recomputing scale on the dequantized master with the ModelOpt default global = amax/(6·448) differs from the checkpoint's TE 4over6 rule and flips about 40% of Nemotron-H's FP4 codes; recomputing with TE 4over6 is idempotent and byte-for-byte unchanged.
- Fix: use a mainstream rule that matches the checkpoint; training QAT and the exporter call the same `requantize`.
- How to verify: requantizing the same master twice is byte-for-byte identical; requantize from the real BF16 release and compare per tensor with the checkpoint. Nemotron-H:

  | Rule | Match with the checkpoint |
  | --- | --- |
  | TE NVFP4 4over6 + MSE, E4M3 upper bound 256, global = amax/1536 | packed 99.776%, global 100%; no setting closes the remaining gap |
  | ModelOpt default global = amax/(6·448) | packed about 50% |
  | Standard FP8 per-tensor amax/448 | See P3 |

- Source: procedure in [Conversion chain §2](#2-inventory-the-checkpoints-full-quantization-rules-first).

#### P2 Verifying a rule with dequantized weights gives a false positive

- Symptom: several candidate rules all report a 100% match.
- Cause: dequantized weights already sit on the grid and do not change under any rule.
- Fix: verify from the real BF16 release.
- How to verify: at least one known-wrong rule fails the test.

#### P3 FP8 per-tensor scale convention is not amax/448

- Symptom: FP8 layers differ from the checkpoint after requantization.
- Cause: Nemotron-H's official Mamba projection scale is about 1.5×amax/448; 24 of 46 FP8 tensors differ from amax/448.
- Fix: derive the convention from the checkpoint, or pass the checkpoint's scale through.
- How to verify: report the scale match rate per tensor.

#### P4 Computing scale separately on each shard

- Symptom: packed weights concatenated after an FSDP/TP sharded export differ from quantizing the full tensor.
- Cause: blocks cross shards; global scale and fused global scale need the full tensor.
- Fix: `full_tensor()`/gather, then quantize; shard-local quantization must first meet the four contracts of [Conversion chain §4.1](#41-fsdp-path-verl).
- How to verify: evidence shows scale is computed after the gather.

#### P5 gate/up not quantized as a pair

- Symptom: wrong values after serving fuses gate/up, or the exporter reports a missing pair.
- Cause: the two HF tensors are fused in serving and need a shared outer scale.
- Fix: quantize as a pair; raise an error when a complete pair is missing, and test pairing across safetensors shards.
- How to verify: test pair completeness, cross-shard gate/up and a TE bitwise reference on matching hardware; a test existing does not mean it ran and passed.

#### P6 LoRA merge=true bypasses the FSDP QAT export

- Symptom: QAT and LoRA merge are on, and the export is not the QAT quantizer's packed weights.
- Cause: VERL FSDP returns `_merged_lora_per_tensor_param()` directly for `merge=true`, skipping `QATQuantizer`; with `rollout.quantization=fp8` the receiver may still requantize. The `_qat_enabled` check still follows the `merge=false` branch.
- Fix: the first merge fallback does gather -> merge -> quantize -> yield -> release one full tensor at a time.
- How to verify: check actual module replacement, naming and export scope; both switches being on does not make the combination usable.
- Source: [VERL engine](https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/workers/engine/fsdp/transformer_impl.py).

#### P7 QAT switch is on, target modules were not replaced

- Symptom: the QAT arm's probs_diff equals the BF16 arm's.
- Cause: VERL `QATLinear` enters through `nn.Linear` and does not cover three-dimensional expert Parameters or arbitrary PEFT wrappers; the framework's built-in QAT changes only the GPT layer spec, while Mamba hybrid models read the hybrid stack spec.
- Fix: attach the quantizer after building the model by the actual spec; use the exporter's `requantize` as the quantizer.
- How to verify: every rank prints the count and list of replaced modules and fails closed (139 for Nemotron-H, the same on all four ranks).
- Source: [Conversion chain §4.1](#41-fsdp-path-verl), [BI=0 integration pitfalls](recipe-and-acceptance.md#bi0-arm-integration-pitfalls).

#### P8 Training grid differs from the deployment grid

- Symptom: QAT is declared, the mismatch remains.
- Cause: the converter and the trainer use different group sizes (for example 32 vs 128). An external TPU case is bitwise 0 on the FP8 per-channel grid, but the native weights are 128×128 block, and the author notes the result may not hold on a different grid.
- Fix: align quantized module scope, block/group size, axis, scale dtype/rounding, global scale scope, gate/up fusion, activation scale, TP/ETP partition and high-precision exceptions. FP4 has 1 mantissa bit, so error comes mainly from block partitioning; block size and direction must match the native weights. MiMo-V2.6 does QDQ on the experts after every update under the rollout MXFP4 GEMM's constraints.
- How to verify: training-side QDQ output = dequantize(deployed bytes), bitwise.
- Source: unified NVFP4 constraints in the [technical blog](https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl/) (its bit-exact quantizer test verifies the format, not whole-model bit-exactness).

#### P9 Activation scale derived from weights

- Symptom: W4A4 export raises, or the exported activation scale differs from serving.
- Cause: activation scale depends on actual input. VERL W4A4 export raises when an initialized `input_global_scale` is missing; Inkling's `input_amax` cannot be derived from BF16 weights.
- Fix: reuse unchanged from the QAT calibration state or rollout checkpoint; pass through `input_scale`, `k_scale`, `v_scale`.
- How to verify: manifest activation scales are byte-for-byte identical to the checkpoint.
- Source: [VERL quantizer](https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/utils/qat/quantizer.py).

### Export and receive

#### P10 Both sender and receiver quantize

- Symptom: deviation larger than one quantization; or a packed tensor treated as BF16 and quantized again.
- Cause: a tensor family has two conversion owners; the Bridge already outputs packed tensors but the generic packer does not recognize them.
- Fix: one owner per family; pass packed tensors through.
- How to verify: the evidence table lists the owner per family; count/bytes close.
- Source: [Conversion chain §4](#4-who-converts-conversion-owner).

#### P11 Receiver picks quantization targets by name suffix and misses fused MoE

- Symptom: under `rollout.quantization=fp8` some experts are still BF16.
- Cause: `gate_up_proj`/`down_proj` do not end in `.weight` and are skipped; fixed by VERL [#7443](https://github.com/verl-project/verl/pull/7443).
- Fix: check coverage using checkpoint names together with runtime modules/dtype; review the list for each new model.
- How to verify: list the serving dtype of every family.
- Source: [receiver](https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/utils/vllm/vllm_quant_utils.py).

#### P12 Inferring a layer's precision from the model's overall name

- Symptom: diagnosis goes in the wrong direction.
- Cause: in the Nemotron-H "NVFP4" checkpoint, `backbone.layers.16.mixer.out_proj` has `quant_algo` FP8.
- Fix: read `hf_quant_config.json` layer by layer.
- How to verify: the diagnosis states the actual `quant_algo` of the first differing layer.

### Training-side numerics

#### P13 scale taken from the wrong object

- Symptom: the unit-scale unit test passes; real attention output and old-logprob fail.
- Cause: the training adapter quantizes Q with `k_scale`. On the real runner `q_scale=1.0`, `k_scale=0.03173828125`, `v_scale=0.003326416015625`.
- Fix: read scale from the runner/module actually executed; do not infer a default from a missing checkpoint field or another tensor.
- How to verify: replay with real non-unit scales and frozen Q/K/V; after the Nemotron-H fix, 5,832,704/5,832,704 BF16 outputs are bitwise identical.

#### P14 Quantization modules with the same name take different dispatch

- Symptom: calling `ops.scaled_fp8_quant` + `flashinfer_scaled_fp8_mm` directly matches training bitwise, yet the inference linear module differs in 1,361 BF16 elements with the same scale.
- Cause: the module selects the **compiled** `forward_native`, which has 44 more differing FP8 elements than uncompiled native and the CUDA custom op; those two agree. Which compile transformation is responsible is not confirmed.
- Fix: under Nemotron-H BI=1, static per-tensor `QuantFP8` switches to the device custom op (GEMM and format unchanged).
- How to verify: with the same operator input, weights and scale, compare FP8 bytes, GEMM output and module output in turn, calling the native/CUDA/actual module paths separately. When the direct GEMM matches training and the module does not, check quantization dispatch inside the module first; do not change GEMM.

#### P15 Quantizer is not in the `nn.Module` tree, the patch misses layers

- Symptom: the patch "ran", the target layer still has 1,361 differing elements.
- Cause: traversing `model.modules()` finds only 6 registered modules and misses quantizers on plain objects such as `linear.quant_method.kernel.quant_fp8`.
- Fix: traverse the registered tree, the quant-method kernel objects and that version's legacy fields together.
- How to verify: record per worker the hit count (52 after the fix), target layer names, actual `_forward_method` and quantized output bytes; fail if expected targets are zero or a key layer is missing. Private field paths are only a versioned case.

#### P16 FSDP BF16 policy overrides the FP32 parameter contract

- Symptom: Hy4 four-layer fixed-token logprob MAE 0.022982/0.023052.
- Cause: HC and attention sink, FP32 after HF load, are downcast by the uniform BF16 FSDP policy; a later `.float()` in the operator cannot restore them. The router's FP32 conversion is also overridden by the outer BF16 autocast, which changes the expert set.
- Fix: opt in with `wrap_policy.preserve_original_param_dtype=True` (`param_dtype=None`, BF16 autocast kept). This is a whole-model policy, not a per-parameter whitelist; measure communication/memory impact; it does not raise a BF16 base to FP32. Disable autocast inside the router forward.
- How to verify: with original dtype kept, singleton, two ranks and unwrapped are exactly identical; the router has a regression test.

#### P17 fake-quant cuts off the gradient

- Symptom: in DSv4.1 LoRA training, 31 of 160 B matrices are zero, all in `self_attn.kv_proj`.
- Cause: `_fake_quant_fp8_block` goes through an FP8 cast on the autograd path, so small gradients underflow to 0; `_fake_quant_fp4_block` has no STE (integer comparison and LUT gather, so gradient flows only through scale, and the ue8m0 branch cuts even that).
- Fix: keep the quantized forward and use identity for the input gradient: `output.detach() + (x - x.detach())`.
- How to verify: a 1e-6 upstream gradient is preserved exactly; after the fix 160/160 B matrices are non-zero. The loss still rises, so this does not prove training improved.

#### P18 Training forward and export do not quantize at the same place

- Symptom: the model already has FP4/FP8 STE, the mismatch remains.
- Cause: DSv4.1's existing STE acts on indexer/compressed/window KV activations, not as weight fake-quant on `W+ΔW`; the training forward is unmerged base + LoRA, while export is an FP32 merge followed by block32 FP8/E8M0 conversion.
- Fix: confirm whether training sees the post-export rounding before deciding on weight QAT.
- How to verify: sample the actual dtype inside forward, not only the code path.

#### P19 KV cache aligned only on the dtype label

- Symptom: attention results differ with the same weights.
- Cause: in the old vLLM image, KV is FP8/block64 for the non-RoPE part and BF16 for the RoPE part, and the training side handles values differently.
- Fix: set `kv_cache_quantization="fp8_ds_mla"` explicitly in training, use the same rounding for forward with and without gradients, and keep the gradient with STE. Re-verify after the upstream KV layout changes.
- How to verify: compare attention output with the same weights.

#### P20 Dividing by temperature in BF16

- Symptom: logprob has a fixed offset.
- Cause: the old VERL FSDP path divides by temperature in BF16/FP16, then raises precision.
- Fix: `logits.float()` first, then divide by temperature, even when temperature is 1.
- How to verify: both scorers match on the same logits.

#### P21 Derived scale after resume still belongs to the old weights

- Symptom: the first rollout after resume is wrong.
- Cause: BF16 parameters were updated, but the FP8 source-scale registry still trusts the initial weights.
- Fix: invalidate derived caches in the weight restore lifecycle.
- How to verify: first-step probs_diff after resume is on the same measurement basis as before (Nemotron-H kill->resume acceptance: still 0 after resume).
- Source: [checkpoint guide §13](checkpoint-resume.md#13-stale-quantization-scale-on-the-first-step-after-restore).

#### P24 Master starts on the quantization grid

- Symptom: QAT loss barely drops and NVFP4 deployed weights are byte-for-byte unchanged while the master moves.
- Cause: a master initialized from a dequantized checkpoint sits on grid points ([QAT §6.1](#61-dead-zone)).
- Fix: master from the real BF16 release, requantized every step by the same rule, cost quantified with proxy SFT; without the original weights, a fixed within-cell jitter. Not stochastic rounding, larger lr, or LoRA.
- How to verify: the three acceptance items of [QAT §6.2](#62-fix).

### Experiment measurement basis

#### P22 The quantization oracle itself is wrong

- Symptom: the GPU block32 quantization gate fails (zero block expected 2.27373675e-13, actual 1.16415322e-10).
- Cause: the oracle assumes Triton's epsilon position; the CUDA UE8M0 path also clamps scale to 1e-10 before taking the power of 2.
- Fix: fix the oracle, not the kernel. CUDA and Triton are not thereby bitwise identical on tiny/zero blocks.
- How to verify: check the oracle's epsilon/floor position against the kernel source before judging a failure.

#### P23 Attributing every logprob difference to quantization

- Symptom: the difference persists after changing precision.
- Cause: routing selection, attention/kernel and tokenization/mask all produce differences. R3 constrains routing and does not make FP8 GEMM equal BF16; TIS is a statistical correction and does not fix wrong weights.
- Fix: rule out protocol errors first, then record the quantization/numerical noise floor.
- How to verify: after ruling out protocol items (routing, mask) one by one, measure the quantization noise floor on the same input.
