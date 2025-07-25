# AutoRound

AutoRound is Intel’s advanced quantization algorithm designed to produce highly efficient **INT2, INT3, INT4, and INT8** quantized large language models—striking an optimal balance between accuracy and deployment performance.

AutoRound applies weight-only quantization to transformer-based models, enabling significant memory savings and faster inference while maintaining near-original accuracy. It supports a wide range of hardware platforms, including **CPUs, Intel GPUs, HPUs, and CUDA-enabled devices**.

Key Features:

✅ **GGUF, AutoGPTQ, AutoAWQ, and AutoRound** are supported:

✅ **10+ vision-language models (VLMs)** are supported

✅ **Per-layer mixed-bit quantization** for fine-grained control

✅ **RTN (Round-To-Nearest) mode** for standard quantization

✅ **Multiple quantization recipes**: best, base, and light

✅ Advanced utilities such as immediate packing and support for 10+ backends

## Installation

```bash
pip install auto-round
```

## Quantizing a model

for vlms, please change to `auto-round-mllm` in CLI usage and `AutoRoundMLLM` in API usage.

```bash
auto-round \
    --model Qwen/Qwen3-0.6B \
    --bits 4 \
    --group_size 128 \
    --format "auto_round" \
    --output_dir ./tmp_autoround
```

### API usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

bits, group_size, sym = 4, 128, True
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym)

## the best accuracy, 4-5X slower, low_gpu_mem_usage could save ~20G but ~30% slower
# autoround = AutoRound(model, tokenizer, nsamples=512, iters=1000, low_gpu_mem_usage=True, bits=bits, group_size=group_size, sym=sym)

## 2-3X speedup, slight accuracy drop at W4G128
# autoround = AutoRound(model, tokenizer, nsamples=128, iters=50, lr=5e-3, bits=bits, group_size=group_size, sym=sym )

output_dir = "./tmp_autoround"
## format= 'auto_round'(default), 'auto_gptq', 'auto_awq'
autoround.quantize_and_save(output_dir, format="auto_round")
```

## Running a quantized model with vLLM

To run an AutoRound quantized model with vLLM, you can
use [DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2](https://huggingface.co/ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2)
with the following command:

