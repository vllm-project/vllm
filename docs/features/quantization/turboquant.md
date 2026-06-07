# TurboQuant (Online 2-4 bit Weight Compression)

`--quantization turboquant` is an online weight-only quantization scheme that compresses BF16 weights to 2, 3, or 4 bits at model load time and requires **no calibration data**.

It combines three techniques:

1. **Walsh–Hadamard randomized rotation** of weight groups. After rotation each coordinate is approximately `N(0, 1/d)`, which lets a small shared codebook work well across all weights.
2. **Lloyd–Max optimal scalar quantization** at 2/3/4 bits into a 4/8/16-entry codebook.
3. **Per-group shape-gain decomposition** — we store `original_norm / reconstruction_norm` (the classical shape-gain factor, Gray 1984) rather than raw L2 norms. This halves the 3-bit reconstruction error in practice by accounting for the norm shrinkage introduced by the quantization step itself.

## Background and naming

The algorithm implemented here is the **scalar case of HIGGS** (Malinovskii, Panferov, Ilin, Guo, Richtárik, Alistarh — *Pushing the Limits of Large Language Model Quantization via the Linearity Theorem*, [NAACL 2025](https://aclanthology.org/2025.naacl-long.543/); preprint [arXiv:2411.17525](https://arxiv.org/abs/2411.17525)). HIGGS describes exactly this combination of Random Hadamard Transform pre-processing, MSE-optimal Lloyd–Max grid, and per-group L2 normalization. A reference implementation also exists in [HuggingFace transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization/higgs.md).

The implementation was originally based on **TurboQuant** ([Zandieh et al., ICLR 2026, arXiv:2504.19874](https://arxiv.org/abs/2504.19874)), which is actually an **online vector quantizer for KV-cache and approximate nearest-neighbour search**, not for static weight compression. Engineering simplifications during development — choosing scalar over vector quantization, WHT over general random rotations, Lloyd–Max over learned grids — converged the weight path onto the HIGGS scalar algorithm. The KV-cache application of TurboQuant is implemented separately in [#38479](https://github.com/vllm-project/vllm/pull/38479) by @vibhavagarwal5.

The `turboquant` name is kept for API and plugin-package compatibility, but **HIGGS is the correct primary citation** for the algorithm in this weight-compression path.

## When to use TurboQuant

- You have a standard BF16 checkpoint and want to serve it at ~4× smaller GPU memory with **zero calibration work**.
- You want a serving-time drop-in: no offline pre-processing of the checkpoint, no additional tools.
- You accept a small quality cost in exchange for fitting larger models on smaller GPUs.

## Quick start

```bash
vllm serve <model> --quantization turboquant            # TQ3, group 128 (default)
vllm serve <model> --quantization turboquant_4bit       # TQ4 (conservative)
vllm serve <model> --quantization turboquant_2bit       # TQ2 (aggressive)
```

Or via the Python API:

```python
from vllm import LLM

llm = LLM(model="...", quantization="turboquant")        # TQ3
llm = LLM(model="...", quantization="turboquant_4bit")   # TQ4
```

Each shorthand maps to a distinct `QuantKey` (`kTurboquantW2` / `kTurboquantW3` / `kTurboquantW4`) and a matching method class. Group size 128 is the only validated codebook today.

## What gets quantized

- **Linear layers** — compressed to the selected bit width via `TurboQuantW2/W3/W4OnlineLinearMethod`.
- **MoE experts** — kept at BF16 in this release. The dispatch table has no MoE entry for the TurboQuant `QuantKey`s, so `OnlineQuantizationConfig` falls through to the default unquantized MoE method. MoE support is planned in a follow-up.
- **Hardware-native FP4 (MXFP4/NVFP4) is out of scope.** The Walsh–Hadamard rotation is applied globally across each weight group, which is fine for per-row scaling but conflicts with per-block scaled formats — a global rotation spreads outlier mass across block boundaries and pollutes the per-block scales. Block-aligned rotation for those formats is a separate PR.

Layers can be excluded via the `ignore` list on `quantization_config`:

```python
llm = LLM(
    model="...",
    quantization="turboquant",
    quantization_config={
        "ignore": ["re:.*lm_head.*", "model.layers.0.self_attn.o_proj"],
    },
)
```

## Implementation notes

- **Zero GPU memory at init**. Weights are allocated on the meta device; only the compressed form materializes on GPU, per layer, during `process_weights_after_loading`.
- **Two Triton kernels** are registered as `torch.library.custom_op` with `register_fake`, so they're fullgraph- and `torch.compile`-safe. The runtime picks between them based on output dimension — a fused dequant-GEMM for smaller outputs, and an FWHT-on-input GEMM for larger outputs (single rotation of the activation instead of N inverse rotations of rows).
- **BF16 tensor cores** are used for the main GEMM. The accumulator is kept in FP32.
- A PyTorch fallback path runs when Triton is unavailable (useful for CPU debugging).

## Minimum hardware

| Implementation | Volta | Turing | Ampere | Ada | Hopper |
| -------------- | ----- | ------ | ------ | --- | ------ |
| TurboQuant     | ❌    | ❌     | ✅︎     | ✅︎  | ✅︎     |

Minimum compute capability is 8.0 (Ampere). The GEMM kernels cast to the model's activation dtype (BF16 or FP16) before `tl.dot` to use Tensor Cores. BF16 Tensor Cores were introduced in Ampere; Turing's 2nd-gen Tensor Cores only support FP16 and would fail on BF16 weights.
