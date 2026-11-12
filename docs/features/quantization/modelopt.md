# NVIDIA Model Optimizer

The [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) is a library designed to optimize models for inference with NVIDIA GPUs. It includes tools for Post-Training Quantization (PTQ) and Quantization Aware Training (QAT) of Large Language Models (LLMs), Vision Language Models (VLMs), and diffusion models.

We recommend installing the library with:

```bash
pip install nvidia-modelopt
```

## Supported ModelOpt checkpoint formats

vLLM detects ModelOpt checkpoints via `hf_quant_config.json` and supports the
following `quantization.quant_algo` values:

- `FP8`: per-tensor weight scale (+ optional static activation scale).
- `FP8_PER_CHANNEL_PER_TOKEN`: per-channel weight scale and dynamic per-token activation quantization.
- `FP8_PB_WO` (ModelOpt may emit `fp8_pb_wo`): block-scaled FP8 weight-only (typically 128×128 blocks).
- `NVFP4`: ModelOpt NVFP4 checkpoints (use `quantization="modelopt_fp4"`).
- `W4A16_NVFP4`: weight-only ModelOpt NVFP4 checkpoints with 16-bit
  activations (use `quantization="modelopt_fp4"`).
- `MXFP8`: ModelOpt MXFP8 checkpoints (use `quantization="modelopt_mxfp8"`).
- `MIXED_PRECISION`: per-layer formats specified by `quantized_layers`
  (use `quantization="modelopt_mixed"`).

For per-tensor `FP8`, vLLM converts E4M3FN weights and scales to E4M3FNUZ
when required by the GPU, for example on AMD MI300.

For Qwen3-based DFlash and DSpark drafts, `quantized_layers` and exclusion lists
use the exported checkpoint's module names. vLLM maps these names to the draft's
module prefix and decoder layer indices before constructing quantized layers.

!!! note
    For NVFP4 checkpoints, vLLM selects a GEMM kernel automatically at load
    time from the backends available on the current platform (CUTLASS,
    FlashInfer, Marlin, and others). On GPUs without a supported native FP4
    GEMM kernel, vLLM falls back to weight-only (W4A16) execution via Marlin
    and logs a warning; this may reduce throughput for compute-heavy
    workloads. Use `--linear-backend` to override the automatic selection
    (this replaces the deprecated `VLLM_NVFP4_GEMM_BACKEND` environment
    variable). Values relevant to NVFP4 include `cutlass`,
    `flashinfer_cutlass`, `flashinfer_cutedsl`, `flashinfer_trtllm`,
    `flashinfer_cudnn`, and `marlin`; see `KernelConfig` on the
    [Engine Arguments](../../configuration/engine_args.md) page and shown by
    `vllm serve --help=KernelConfig`.

!!! note
    For models quantized to MXFP8 with BF16 activations on SM100-family GPUs,
    use `--linear-backend flashinfer_trtllm` to select FlashInfer's TensorRT-LLM
    GEMM backend.

For Muse Spark NVFP4 on NVIDIA, the prepared-input FlashInfer TRT-LLM path
skips CUDA-graph padding rows when routing runs outside the expert kernel.
The router writes expert IDs of `-1` and zero weights in its existing fused
kernel. This requires a padding mask matching the routed token layout;
other dispatch paths retain their backend-specific behavior.
Set `VLLM_MOE_SKIP_PADDING=0` before starting the engine to disable this
optimization for comparisons.

## Benchmarking with dummy weights

For performance experiments with random weights, use `--load-format dummy`.
ModelOpt NVFP4 MoE layers initialize positive quantization scales with matching
gate/up global scales, including for FlashInfer MegaMoE
(`--moe-backend flashinfer_moe_ep_cutedsl`).
Muse Spark's NVIDIA `deep_gemm_mega_moe` backend also supports dummy loading:
it initializes valid NVFP4 scales and finalizes its expert weights and residual
constants before the first forward. Enable expert parallelism for this backend.
Dummy weights are for benchmarking, not model quality evaluation. Normal
checkpoint loading uses the checkpoint's scales.

## Quantizing HuggingFace Models with PTQ

You can quantize HuggingFace models using the example scripts provided in the Model Optimizer repository. The primary script for LLM PTQ is typically found within the `examples/llm_ptq` directory.

Below is an example showing how to quantize a model using modelopt's PTQ API:

??? code

    ```python
    import modelopt.torch.quantization as mtq
    from transformers import AutoModelForCausalLM

    # Load the model from HuggingFace
    model = AutoModelForCausalLM.from_pretrained("<path_or_model_id>")

    # Select the quantization config, for example, FP8
    config = mtq.FP8_DEFAULT_CFG

    # Define a forward loop function for calibration
    def forward_loop(model):
        for data in calib_set:
            model(data)

    # PTQ with in-place replacement of quantized modules
    model = mtq.quantize(model, config, forward_loop)
    ```

After the model is quantized, you can export it to a quantized checkpoint using the export API:

```python
import torch
from modelopt.torch.export import export_hf_checkpoint

with torch.inference_mode():
    export_hf_checkpoint(
        model,  # The quantized model.
        export_dir,  # The directory where the exported files will be stored.
    )
```

The quantized checkpoint can then be deployed with vLLM. As an example, the following code shows how to deploy `nvidia/Llama-3.1-8B-Instruct-FP8`, which is the FP8 quantized checkpoint derived from `meta-llama/Llama-3.1-8B-Instruct`, using vLLM:

??? code

    ```python
    from vllm import LLM, SamplingParams

    def main():
        model_id = "nvidia/Llama-3.1-8B-Instruct-FP8"

        # Ensure you specify quantization="modelopt" when loading the modelopt checkpoint
        llm = LLM(model=model_id, quantization="modelopt", trust_remote_code=True)

        sampling_params = SamplingParams(temperature=0.8, top_p=0.9)

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    if __name__ == "__main__":
        main()
    ```

## Running the OpenAI-compatible server

To serve a local ModelOpt checkpoint via the OpenAI-compatible API:

```bash
vllm serve <path_to_exported_checkpoint> \
  --quantization modelopt \
  --host 0.0.0.0 --port 8000
```

## Testing (local checkpoints)

vLLM's ModelOpt unit tests are gated by local checkpoint paths and are skipped
by default in CI. To run the tests locally:

```bash
export VLLM_TEST_MODELOPT_FP8_PC_PT_MODEL_PATH=<path_to_fp8_pc_pt_checkpoint>
export VLLM_TEST_MODELOPT_FP8_PB_WO_MODEL_PATH=<path_to_fp8_pb_wo_checkpoint>
pytest -q tests/quantization/test_modelopt.py
```

## GB10 Mamba2 input projection and convolution

A CUDA13 build can fuse input projection and convolution for an already
selected Marlin NVFP4 W4A16 Mamba2 layer on GB10 SM121 hardware with 48 SMs.
The specialization requires BF16, tensor parallel size 1, and the supported
4096-hidden/128-head geometry using FlashInfer ReplaySSM. It supports batch
sizes `B=1–4` and configured token lengths `T=1–8` when `B*T <= 16`.

Batch size is a runtime value; the configured token length `T` is compiled
for each graph. With `B=1`, runtime metadata may select fewer valid tokens
than the configured length. With `B=2–4`, every request must have the same
length, equal to the configured `T`. The fast path requires structural
metadata and padded-layout checks to pass. Inactive cache slots remain
protected. Unsupported or unproven geometry, metadata, wrappers and backends
retain the original projection and convolution path.

On SM121, supported ModelOpt NVFP4 weight-only Mamba2 input/output projections
select Marlin locally before normal post-load weight packing, including shapes
that have not been tuned and may be slower. Attention,
other linears, MoE and the shared backend configuration are unchanged; only one
weight pack is created. Unsupported codecs/dtypes, LoRA, batch-invariant mode
and explicitly disabled Marlin kernels retain their configured owners.
The fused operators require those existing Marlin packs and never convert
another backend's packed weights at runtime. Unmatched token/batch shapes use
ordinary Marlin for the same Mamba-owned weights. The fusion preserves ReplaySSM
state and stochastic-rounding seed handling.

Only the matching Mamba output callsite selects the tuned existing Marlin
kernel for BF16 `M=8, K=8192, N=4096` with NVFP4 group16 weights. The generic
Marlin operator and its dispatcher keep their stock behavior.

For compatible FP16 ReplaySSM calls, this GB10 path applies measured launch
profiles to 29 combinations of uniform batch/token shape and caller-selected
rounding policy among `B=1–4`, `T=1–8`, `B*T <= 16`. The policy must be Philox
with five configured rounds or shared seed, independent of its unused round count.
The selector never changes the caller's rounding mode or round count. It
requires the actual fused Marlin-convolution producer, equal configured/runtime
verification lengths, BF16 inputs, a 16-token checkpoint window
(`16+T` physical ring entries; 24 at `T=8`), 128 value heads of width 64,
128 state channels and eight shared Q/K groups, with PDL disabled and the
supported state/ring layouts and scratch buffers. FP8, INT8, round-to-nearest
and unmatched shapes retain normal FlashInfer selection/autotuning.

For `B=1`, both policies use the two-kernel profile through `T=6`, then
monolith with `d_split=1` at `T=7–8`. `B=2` uses the same rule for Philox5;
shared seed retains normal selection at `T=1–4`, uses the two-kernel profile
at `T=5–6`, and monolith with `d_split=1` at `T=7–8`. `B=3, T=5` with Philox5
uses monolith with `d_split=2`; all other `B=3–4` points retain normal selection.

Profiles use existing monolithic or two-kernel algorithms and launch controls.
The installed FlashInfer API is inspected once at backend initialization;
profiles requiring new MAIN controls retain normal selection when those
controls are unavailable. Shared seed requires a compatible FlashInfer API.
An explicit SSU algorithm setting retains precedence, including `auto`.
Setting any of `FLASHINFER_SSU_HEADS_PER_CTA`,
`FLASHINFER_SSU_PRECOMPUTE_NUM_WARPS`, `FLASHINFER_SSU_MAIN_CTA_PER_SM` or
`FLASHINFER_SSU_MAIN_PIPELINE_STAGES` also preserves the configured FlashInfer
path.
