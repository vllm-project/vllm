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
- `MXFP8`: ModelOpt NVFP4 checkpoints (use `quantization="modelopt_mxfp8"`).

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
