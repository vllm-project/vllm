# BitsAndBytes

vLLM now supports [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) for more efficient model inference.
BitsAndBytes quantizes models to reduce memory usage and enhance performance without significantly sacrificing accuracy.
Compared to other quantization methods, BitsAndBytes eliminates the need for calibrating the quantized model with input data.

Below are the steps to utilize BitsAndBytes with vLLM.

```bash
pip install bitsandbytes>=0.49.2
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

## INT8 throughput caveat at small batch sizes

:::{warning}
BitsAndBytes INT8 (`load_in_8bit=True`) can cause significant throughput regression at
small batch sizes on memory-bandwidth-bound GPUs. In benchmarks on an NVIDIA L4 GPU
across 317,486 real prompts, INT8 at `batch_size=1` was **4× slower than FP16**
(see [#43700](https://github.com/vllm-project/vllm/issues/43700)).

**Root cause:** BitsAndBytes dequantizes weights from INT8 to FP16 before each matrix
multiplication. At small batch sizes, this dequantization overhead dominates inference
time — CUDA profiling confirms it consumes ~34% of total CUDA time, while attention
kernels consume less than 1%.

The regression disappears at larger batch sizes (≥8) where compute becomes the
bottleneck rather than memory bandwidth.

**Recommendation:** For latency-sensitive or low-concurrency serving scenarios, prefer
GPTQ, AWQ, or FP8 quantization instead. Use INT8 only when batch sizes are consistently
large enough to amortize the dequantization cost.
:::

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
