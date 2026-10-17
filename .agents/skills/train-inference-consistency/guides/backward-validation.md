# Backward: reuse and validation

## Reusing backward: when you actually need to write your own

When to reuse backward and when to write glue.

### Conclusion

Backward is determined by the **math formula**, not by the forward kernel. The backward of `y = f(x, W)` needs only the formula and the saved inputs: `dX = dY·W`, `dW = dYᵀ·x`. A vLLM forward kernel does not by itself require a new backward. "No backward operator at the same precision" is not a reason to rewrite: under the high_precision contract the backward runs in BF16/FP32 anyway.

### How to wire it: forward once, backward uses what the forward saved

The training forward is the inference forward. Each module is an autograd Function: its forward calls the inference kernel to produce the visible output and saves what its own backward needs (inputs, visible output, LSE, fixed routing IDs, quantized values and scales); its backward uses only those, through a reused library backward or a closed-form VJP. If the inference kernel does not output something the backward needs, change the kernel to output it.

Delivery requires 0 diff and a training step within budget on the main topologies fixed at C1; C6 checks both together. Nemotron audited this against DS4: attention saves q/k/v, out and LSE; routed MoE saves x, the visible FC1 output and the visible output; linear and norms save their inputs; the router saves logits and fixed IDs. Its open gap is Mamba SSD, because vLLM's visible SSD does not output the chunk state.

### Three cases

| Case | Characteristics | Examples | What to do |
| --- | --- | --- | --- |
| A only rounding differs | Same math; summation order/chunking differs; difference in the last BF16 bit | RMSNorm, Mamba SSD scan, attention, conv1d, unquantized linear | Reuse the training framework's backward (MCore/TE/mamba_ssm); write no code |
| B forward has rounding | FP8/NVFP4 quantization rounds; the true derivative is 0 almost everywhere | Quantized linear, FP8 Q/K/V | First fix the STE proxy (rounding passes through as 1; dX uses dequantized values by default; DS4 and Nemotron currently approximate with TE high_precision master values, and the deviation must be quantified per [layer 2](#layer-2-gradient-contract-comparison)), then reuse the ordinary BF16 GEMM backward |
| C no ready-made backward for the whole segment | Composite of several operators with no library backward for the whole segment | Routed experts: sort -> grouped GEMM -> relu² -> restore -> top-k weighted sum | Write glue that chains ready-made operators; GEMM still uses TE grouped GEMM; compare gradients against an independent reference |

Decision flow: (1) does the forward have rounding/quantization (top-k ID selection does not count; IDs are fixed and not differentiated)? No -> A. Yes -> (2) fix the STE contract -> (3) is there a packaged training backward for the whole segment (TE Linear/GroupedLinear, MCore MoE)? Yes -> B, reuse it and feed master values; no -> C, glue. If A has no same-math training implementation, also write only glue. Look first in mLite `primitive/modules` and `ops`, then in MCore/TE/mamba_ssm.

### How DS4 did it

DS4's only hand-written kernel is the small FP8 quantization Triton kernel in the forward. The backward has no hand-written kernel, only PyTorch closed-form formulas and glue, all in FP32.

| Operator | Case | Backward source | What we wrote |
| --- | --- | --- | --- |
| FP8 dense linear (qkv, shared expert, o-proj) | B | TE `general_gemm`, BF16 master | Nothing |
| Routed experts (grouped) | B+C | TE `general_grouped_gemm`; reuses gate_up, the output of the forward vLLM FC1, without recomputing | Glue: sort, activation derivative, summation |
| Sparse attention | A | cuDNN sparse attention backward, consuming the forward's saved out and LSE | Nothing |
| SwiGLU (with clamp), RMSNorm, q/kv norm, mHC | A | Hand-written closed-form derivative, FP32, `torch.compile` | PyTorch formula |
| Routing score | A (IDs fixed) | FP32 replay of `sqrt(softplus)` + normalization, handed to autograd | PyTorch formula |
| Others | A | Generic bridge: forward via vLLM, backward recomputes the PyTorch graph then `autograd.grad` | One utility function |

DS4 wrote its own for four reasons, none of them a missing same-precision backward kernel: (1) library backwards need intermediates from the library's own forward (TE RMSNorm needs rstd, MCore MoE the token permutation; the vLLM forward has neither); (2) model-specific math is not in the library (`sqrt(softplus)` scoring, SwiGLU with clamp, mHC Sinkhorn); (3) discrete selection must copy the forward (recomputed top-k may pick a different expert on ties, so store the IDs and differentiate only the continuous score); (4) the generic bridge costs too much memory and time (autograd stores all 20 mHC iterations; use a closed-form derivative fused with compile). A custom backward function has no autograd graph inside, so its activation derivative must be closed-form (`dynamic_clamped_swiglu_vjp`; likewise relu² in Nemotron `routed_vjp`).

### Three rules when reusing

1. Intermediate definitions must match: attention backward uses the forward's out and LSE, so LSE base (ln or log2), softmax scale and memory layout must line up. If they do not, recompute the forward with the training kernel inside the backward.
2. Backward need not be bitwise identical: recomputed intermediates differ from the vLLM forward in the last BF16 bit and gradients differ by the same amount, which is allowed. The requirement is that the gradient contract holds and the error is near the noise floor.
3. Reused code must be verified too, per [layer 1](#layer-1-module-level-vjp-comparison): compare dX and dW against an FP32 reference that does not call the backward under test, with the same dY, in multiples of the noise floor.

### Two test-infrastructure false negatives

- A unit test that instantiated QuantFP8 directly lacked production training's vLLM config context (`CompilationConfig(custom_ops=['none','+quant_fp8'])`, Humming, PP4), and the first long-sequence test failed before computing anything. Fix: the test wrapper uses the production config; production operators and tolerances are unchanged.
- An import check without GPU visibility makes vLLM select a non-CUDA platform, which cannot import `reshape_and_cache_flash` (exported only on CUDA). Dependency checks need real device visibility.
- Both are test errors, not train-inference differences.

## Backward validation: is the implementation right, is the approximation reasonable, is training biased

Forward 0 diff shows the logprobs match, not that gradients are right. Validate the backward in three layers:

| Layer | Question answered | Method | Cannot replace |
| --- | --- | --- | --- |
| 1. Module-level VJP comparison | Does the implementation match the agreed gradient contract (missing scale, wrong transpose, broken gradient chain) | compare dX, dW and a one-step update of every module against an independent reference that does not call the backward under test | layers 2 and 3 |
| 2. Gradient contract comparison | Is the contract reasonable: does an approximation (e.g. dX from the master value) bias training | same batch, compare gradients under the two conventions per class and per layer | layer 1 |
| 3. Proxy SFT three-arm comparison | Does the whole training chain have a systematic bias | three SFT paths (under test, gradient reference, native baseline) with the same data and hyperparameters | layers 1 and 2 |

Run 1 → 2 → 3. If layer 1 fails, fix the implementation before any SFT. If layer 3 is abnormal, go back to layers 1 and 2. Passing all three clears the backward; none proves long-term RL stability.

### Write down the gradient contract first

#### The three tiers of a weight

In quantized training one parameter has three values, and the gradient must end up at the one the optimizer manages:

| Tier | Precision | What it is |
| --- | --- | --- |
| main | FP32 | optimizer state; updates accumulate here |
| master | BF16 | the model parameter, converted from main every step; non-quantized layers compute with it directly |
| deployed value | NVFP4 / FP8 etc. | master quantized per the deployment rule; both the training forward and inference use these bytes |

Keep the optimizer's parameter identity: when replacing the forward, do not create parameters the optimizer does not know about. The deployed value is not a trainable parameter.

#### Contract table

A kernel without autograd does not need a surrogate gradient: non-quantized GEMM, norm and attention reuse the existing backward when the math matches; STE is needed only through discrete operations such as quantization rounding. Before implementing, decide per module class:

| Item | What must be decided |
| --- | --- |
| Visible forward | the inference implementation; input/output dtype, layout; quantize/dequantize positions, scale granularity, rounding, clamp, accumulation precision |
| Trainable variables | input, master weight, bias, norm, router; whether scale is fixed, stop-gradient or trainable |
| Surrogate function | whether dX uses the master or dequantized value (see below); identity or clipped STE, and the value at the threshold |
| Discrete state | fixed expert IDs, sparse indices, ordering; continuous score/combine weights that still need gradients |
| backward | reused implementation or custom VJP; cast, accumulation/reduction precision, communication and normalization |
| Save/recompute | quantized values, scale, norm/attention intermediates, routing, RNG; memory cost and recompute rules |
| Reference and acceptance | independent reference implementation, target shape/execution path, metrics and tolerances agreed in advance |

Identity STE sets the surrogate derivative of Q(x) to 1; clipped STE sets 1 inside the range and 0 outside, with the threshold value stated and never switched silently. Declare weight and activation quantization separately; frozen weights do not mean nothing upstream of the activations needs gradients.

#### Linear: which weight dX uses

For `Y = X · Wᵀ + b` with upstream gradient `G`:

```text
dW = Gᵀ · X_ref        # returns to master via STE, then the optimizer updates main
db = sum_over_tokens(G)
dX = G · W_ref
```

Two choices for `W_ref`:

- **Dequantized value**: `W_ref = dequant(quant(W_master))`, the value the forward actually used. This is the true derivative of `Y = X · Wqᵀ` with respect to X, and what fake-quant QAT gets from autograd. **Use this by default.**
- **Master value**: `W_ref = W_master` (BF16). Reuses the plain BF16 GEMM backward without dequantizing; the cheapest hookup when the inference kernel takes only packed bytes and has no backward. Cost: dX is no longer the exact forward derivative and ignores the quantization error's effect on the input gradient.

The two give different gradients and must not be swapped silently. With the master value, the layer-2 comparison is mandatory: quantify and record the bias. Likewise for activation quantization, state whether `X_ref` is the quantized or original activation.

Before reusing a linear or grouped GEMM backward, check flatten/transpose, padding and valid tokens, bias reduction, saved inputs, scaling and the accumulation/backprop dtype. Matching output shape does not prove reusability.

### Layer 1: module-level VJP comparison

#### How to build the reference implementation

The reference may use PyTorch ops and autograd but **must not call the backward under test**, or both paths share the error. For discrete operations implement the agreed surrogate backward explicitly; the zero gradient of bare rounding is not a reference.

This bridge takes the output value from the inference implementation and the gradient from an independent surrogate. Use it to build the reference path:

```python
import torch


class VisibleWithSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible, surrogate):
        assert visible.shape == surrogate.shape
        assert visible.device == surrogate.device
        assert visible.dtype == surrogate.dtype
        return visible

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output


with torch.no_grad():
    visible = inference_linear(x, deployment_weight, scales)
w_ref = dequantize(deployment_weight, scales)  # default contract; use master_weight for the master-value contract
surrogate = torch.nn.functional.linear(x, w_ref, bias)
y = VisibleWithSurrogate.apply(visible.detach(), surrogate)
```

- Build the surrogate graph in normal grad mode, outside the custom Function's forward.
- With the dequantized surrogate, `w_ref`'s gradient must reach master via STE: bridge a tensor valued as the dequantized weight whose gradient passes straight to master, then feed it to a plain linear.
- Do not use `visible + surrogate - surrogate.detach()`: floating-point add/subtract and compiler transforms change the result. Do not modify the bridge output in place downstream.
- This bridge is the independent validation reference, not the production wiring; production wiring is in [How to wire it](#how-to-wire-it-forward-once-backward-uses-what-the-forward-saved).
- `visible_forward(visible, native, *inputs)` is another way to write the same bridge: `visible` saves inputs without building a graph, `native` is the training framework's implementation of the same math and reruns the forward on those inputs in backward; precision and speed depend on which `native` is chosen.

#### Comparison steps

1. **Fix the state.** Fix inputs, weights, quantization config, mask/routing. Reference and candidate get independent copies; clear accumulated gradients. Check that the visible forward matches inference, then the saved state the backward needs; at integration, confirm the real forward produces that state.
2. **Same upstream gradient.** Both sides use the same external dY, dtype and values. Cover fixed random, zero, and sparse (non-zero at selected positions) gradients, then samples from real training. `y.sum().backward()` alone misses transpose, permutation and broadcast errors.
3. **Compare item by item.** Compare dX, dW and every bias/norm/router/scale gradient separately: missing, shape, finiteness, non-zero where expected. Do not ignore unused differentiable inputs by default; list parameters the branch genuinely does not use. Report max absolute error, relative L2 `‖g − g_ref‖₂ / max(‖g_ref‖₂, eps)`, norm ratio and cosine; direction alone is not enough. With a zero norm the cosine is undefined; use absolute error.
4. **Set tolerances in advance.** Derive atol/rtol and the relative L2 threshold from dtype, reduction length and reference precision; record the reasoning. No single threshold for all modules, no relaxing after a failure, and no relaxing a same-semantics kernel optimization because "STE is an approximation anyway". Cross-framework bitwise identity is not required.
5. **Coverage.** Small M (decode), large M (training/prefill), different context lengths, unaligned tails, padding/packing/mask, zeros and extreme values; both sides of the quantization threshold, the saturation region, different scales. For MoE cover empty experts, repeated routing, shared experts, top-k ties; for shared parameters check gradient accumulation across use sites. Test under the target dtype/layout, recompute, autocast, CUDA Graph, gradient accumulation and the actual parallel reduction; local first, then integrated.
6. **One-step update.** From the same parameters and optimizer state, run one update each and compare the change in master. If gradients are close but updates differ, check scaling, clipping, optimizer state and parameter binding.

Cross-check smooth non-quantized modules with small, high-precision [gradcheck](https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html) or finite differences: fix the discrete branch, avoid non-smooth boundaries, and perturb the weights the forward actually sees. **Do not validate STE with finite differences of the real quantized forward**: that measures the discrete forward, which the STE surrogate derivative deliberately differs from.

#### What to check per module

| Module | Reuse first | Semantics to check |
| --- | --- | --- |
| Dense/QKV/LM head | training linear/grouped VJP, gradient returns to master | dX value convention, cast, bias, gradient accumulation of shared parameters |
| Norm/RoPE | existing training VJP, FP32 statistics if needed | epsilon, RMS or LayerNorm, position offset, rotation layout |
| Activation + quantization | the actual composition of activation derivative and STE | saturation region, whether scale is detached, fusion order |
| Attention/sparse attention | training backward with the same mask/index semantics | causal boundary, softmax scale, LSE definition, GQA, order of selected KV; do not mix auxiliary quantities from different forwards |
| MoE | expert linear/grouped VJP and a differentiable combine | inverse dispatch permutation, gradient accumulation of repeated tokens, empty experts, shared experts, top-k weight normalization |
| Router/sparse selection | fix the discrete selection, keep the continuous score's derivative | no continuous gradient for IDs/indices; do not detach the whole router output |
| Linear attention/SSM | training backward with the same chunked arithmetic | chunk boundary, initial state, state precision, conv state |
| Selected logprob | log-softmax/gather VJP matching the definition | vocab-shard reduction, temperature, token mask, loss normalization |

Recompute must save or deterministically rebuild the scale, routing/indices, RNG and necessary intermediates, and must not switch the surrogate function.

### Layer 2: gradient contract comparison

Layer 1 answers "does the implementation equal the contract", not "is the contract reasonable". When the contract contains an approximation (most often dX from the master instead of the dequantized value), quantify its distance from the exact derivative:

1. On a fixed checkpoint and batch, compute dX (and gradients related to activation quantization) for every quantized layer under both conventions.
2. Report cosine and norm ratio per module class and per layer; inspect outlier layers.
3. Measure each target precision combination separately: changing weight or activation precision changes the bias.

On Nemotron-H (NVFP4 routed experts, FP8 Mamba projections), master-value and dequantized-value dX have cosine ≈0.99 and norm ratio ≈1.00 per class and layer, with both BF16 and FP8 activations, so this approximation does not cause the training score gap. Remeasure for other precisions, a larger learning rate or longer training.

### Layer 3: Proxy SFT three-arm comparison

After the first two layers pass, run a controlled SFT over the whole training chain:

| Path | Visible forward | backward | Purpose |
| --- | --- | --- | --- |
| Under test | aligned inference numerics | production backward | the scheme to deliver |
| Gradient reference | same as under test at the same parameter snapshot | independent surrogate reference (the bridge above works) | isolate backward implementation errors |
| Native baseline | original training numerical path | verified original training backward | measure the recipe's overall quality change |

- First compare gradients and one-step update of under-test vs gradient-reference on a fixed checkpoint and batch, then run curves; once trajectories diverge, visible values differ too. The under-test vs native-baseline gap includes quantization and forward-path differences, not only backward. If the gradient reference's full curve is unaffordable, compare on representative batches and state the missing range.
- Controlled variables: initial weights and optimizer state, data and order, labels/mask, packing, loss normalization, global batch and gradient accumulation, learning rate and schedule, clipping, precision and loss scaling, seed, recompute policy.
- Record train loss, fixed-validation-set NLL, gradient norm, update RMS, clipping ratio, non-finite values, throughput, against optimizer step or cumulative effective tokens. Fix the comparison window and acceptable deviation before the run; a short run is only a smoke test.
- Localization: loss differs at the initial weights -> check forward, data, mask; initial loss matches but one-step update differs -> backward, reduction, scale, clipping, optimizer; diverges only after several steps -> save the first abnormal checkpoint, check recompute, quantization refresh, learning rate, RNG, then replay at the same weights.
- Mark unexplained or unbudgeted differences inconclusive; do not cover them with forward 0 diff.

On Nemotron-H the three paths (aligned trainer with BF16 activations, Megatron QAT, Megatron BF16) ran 40 steps on gsm8k with the same data in the same order: loss 0.3388 / 0.3383 / 0.3374, gradient norm 0.685 / 0.674 / 0.728, curves overlapping. The training loop itself is fine; the RL score gap comes from RL-specific factors.

### Common false differences

Each of these made gradients look mismatched when the comparison method was the problem:

- **Upstream gradient precision differs.** FP32 dY on one side and BF16-rounded dY on the other creates a difference unrelated to the backward. Use the same external dY.
- **Dump timing.** Expert gradients are multiplied by the expert-DP to DP scaling ratio only inside the optimizer step; dumping earlier shows a false multiplicative difference.
- **Unsharded copy under FSDP2.** After forward-only the root module is not resharded, so the parameter fetched by name is the unsharded compute copy, not the real master. Perturbing it or reading its gradient both mislead.
- **"Frozen" is not frozen.** Zeroing the gradient still lets AdamW weight decay change the parameter; a real freeze needs grad None or removal from the optimizer.
- **Test environment differs from production.** A unit test missing production config context, or an import without GPU visibility, selects the wrong implementation or fails to import (see [Two test-infrastructure false negatives](#two-test-infrastructure-false-negatives)).

### What to write in the report

Per module class: gradient contract (dX value choice, STE type, scale handling), reference implementation, upstream gradient construction, shapes/paths covered, metrics and tolerances with reasoning. Layer 2: per-class per-layer cosine/norm ratio and outlier layers. Layer 3: config diff table of the three paths, curves, conclusion within the window. List missing coverage explicitly, not as "the rest is analogous".

For the basic definition of STE see [Bengio et al.](https://arxiv.org/abs/1308.3432); it does not decide the surrogate gradient rule for a specific model.
