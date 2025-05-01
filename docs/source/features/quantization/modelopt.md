# NVIDIA TensorRT Model Optimizer

The [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) is a library designed to optimize models for inference with NVIDIA GPUs. It includes tools for Post-Training Quantization (PTQ) and Quantization Aware Training (QAT) of Large Language Models (LLMs), Vision Language Models (VLMs), and diffusion models.

We recommend installing the library with:

```console
pip install nvidia-modelopt
```

## Quantizing HuggingFace Models with PTQ

You can quantize HuggingFace models using the example scripts provided in the TensorRT Model Optimizer repository. The primary script for LLM PTQ is typically found within the `examples/llm_ptq` directory.

Here's an example of how you might run the quantization script (refer to the [specific examples](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/llm_ptq) for exact arguments and usage):

```console
# Quantize and export
python hf_ptq.py --pyt_ckpt_path meta-llama/Llama-3.1-8B-Instruct --qformat fp8 --export_fmt hf --export_path <quantized_ckpt_path> --trust_remote_code

# After quantization, the exported model can be potentially deployed with vLLM.
```

This process generates a quantized model checkpoint. As an example, the following codes show how to deploy `nvidia/Llama-3.1-8B-Instruct-FP8` which is the fp8 quantized checkpoint from `meta-llama/Llama-3.1-8B-Instruct` with vllm.

```console
from vllm import LLM, SamplingParams

def main():

    model_id = "nvidia/Llama-3.1-8B-Instruct-FP8"
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    llm = LLM(model=model_id, quantization="modelopt")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()
```
