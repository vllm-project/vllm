# BitsAndBytes

vLLM now supports [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) for more efficient model inference.
BitsAndBytes quantizes models to reduce memory usage and enhance performance without significantly sacrificing accuracy.
Compared to other quantization methods, BitsAndBytes eliminates the need for calibrating the quantized model with input data.

Below are the steps to utilize BitsAndBytes with vLLM.

```bash
pip install bitsandbytes>=0.44.1
```

vLLM reads the model's config file and supports both in-flight quantization and pre-quantized checkpoint.

You can find bitsandbytes quantized models on [Hugging Face](https://huggingface.co/models?search=bitsandbytes).
And usually, these repositories have a config.json file that includes a quantization_config section.

## Read quantized checkpoint

For pre-quantized checkpoints, vLLM will try to infer the quantization method from the config file, so you don't need to explicitly specify the quantization argument.

```python
from vllm import LLM
import torch
# unsloth/tinyllama-bnb-4bit is a pre-quantized checkpoint.
model_id = "unsloth/tinyllama-bnb-4bit"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
```

## Inflight quantization: load as 4bit quantization

For inflight 4bit quantization with BitsAndBytes, you need to explicitly specify the quantization argument.

```python
from vllm import LLM
import torch
model_id = "huggyllama/llama-7b"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization="bitsandbytes",
)
```

## OpenAI Compatible Server

Append the following to your model arguments for 4bit inflight quantization:

```bash
--quantization bitsandbytes
```

## Note on Batch Size Dependency

Weight-only quantization via bitsandbytes may reduce throughput at small batch sizes (e.g., batch=1) due to dequantization overhead. Each linear layer requires casting weights from INT8/INT4 back to float16 before matrix multiplication, adding a memory movement step that can dominate memory-bandwidth-bound workloads. This overhead is typically amortized at larger batch sizes (>= 8). For latency-sensitive single-request serving, it is recommended to benchmark bitsandbytes against FP16 on your target hardware before deployment.
