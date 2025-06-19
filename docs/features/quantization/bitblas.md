---
title: BitBLAS
---
[](){ #bitblas }

vLLM now supports [BitBLAS](https://github.com/microsoft/BitBLAS) for more efficient and flexible model inference. Compared to other quantization frameworks, BitBLAS provides more precision combinations.

!!! note
    Ensure your hardware supports the selected `dtype` (`torch.bfloat16` or `torch.float16`).
    Most recent NVIDIA GPUs support `float16`, while `bfloat16` is more common on newer architectures like Ampere or Hopper.
    For details see [supported hardware](https://docs.vllm.ai/en/latest/features/quantization/supported_hardware.html).

Below are the steps to utilize BitBLAS with vLLM.

```bash
pip install bitblas>=0.1.0
```

vLLM reads the model's config file and supports pre-quantized checkpoints.

You can find pre-quantized models on:

- [Hugging Face (BitBLAS)](https://huggingface.co/models?search=bitblas)
- [Hugging Face (GPTQ)](https://huggingface.co/models?search=gptq)

Usually, these repositories have a `quantize_config.json` file that includes a `quantization_config` section.

## Read bitblas format checkpoint

```python
from vllm import LLM
import torch

# "hxbgsyxh/llama-13b-4bit-g-1-bitblas" is a pre-quantized checkpoint.
model_id = "hxbgsyxh/llama-13b-4bit-g-1-bitblas"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization="bitblas"
)
```

## Read gptq format checkpoint

??? Code

    ```python
    from vllm import LLM
    import torch

    # "hxbgsyxh/llama-13b-4bit-g-1" is a pre-quantized checkpoint.
    model_id = "hxbgsyxh/llama-13b-4bit-g-1"
    llm = LLM(
        model=model_id,
        dtype=torch.float16,
        trust_remote_code=True,
        quantization="bitblas",
        max_model_len=1024
    )
    ```
