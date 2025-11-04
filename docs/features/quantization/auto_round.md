# AutoRound

[AutoRound](https://github.com/intel/auto-round) is Intel’s advanced quantization algorithm designed to produce highly efficient **INT2, INT3, INT4, and INT8**
quantized large language models—striking an optimal balance between accuracy and deployment performance.

AutoRound applies weight-only quantization to transformer-based models, enabling significant memory savings and faster
inference while maintaining near-original accuracy. It supports a wide range of hardware platforms, including **CPUs,
Intel GPUs, HPUs, and CUDA-enabled devices**.

Please refer to the [AutoRound guide](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md) for more details.

Key Features:

✅ **AutoRound, AutoAWQ, AutoGPTQ, and GGUF** are supported

✅ **10+ vision-language models (VLMs)** are supported

✅ **Per-layer mixed-bit quantization** for fine-grained control

✅ **RTN (Round-To-Nearest) mode** for quick quantization with slight accuracy loss

✅ **Multiple quantization recipes**: best, base, and light

✅ Advanced utilities such as immediate packing and support for **10+ backends**

## Installation

```bash
uv pip install auto-round
```

## Quantizing a model

For VLMs, please change to `auto-round-mllm` in CLI usage and `AutoRoundMLLM` in API usage.

### CLI usage

```bash
auto-round \
    --model Qwen/Qwen3-0.6B \
    --bits 4 \
    --group_size 128 \
    --format "auto_round" \
    --output_dir ./tmp_autoround
```

```bash
auto-round \
    --model Qwen/Qwen3-0.6B \
    --format "gguf:q4_k_m" \
    --output_dir ./tmp_autoround
```

### API usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

bits, group_size, sym = 4, 128, True
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym)

# the best accuracy, 4-5X slower, low_gpu_mem_usage could save ~20G but ~30% slower
# autoround = AutoRound(model, tokenizer, nsamples=512, iters=1000, low_gpu_mem_usage=True, bits=bits, group_size=group_size, sym=sym)

# 2-3X speedup, slight accuracy drop at W4G128
# autoround = AutoRound(model, tokenizer, nsamples=128, iters=50, lr=5e-3, bits=bits, group_size=group_size, sym=sym )

output_dir = "./tmp_autoround"
# format= 'auto_round'(default), 'auto_gptq', 'auto_awq'
autoround.quantize_and_save(output_dir, format="auto_round")
```

## Running a quantized model with vLLM

Here is some example code to run auto-round format in vLLM:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95)
model_name = "Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound"
llm = LLM(model=model_name)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Acknowledgement

Special thanks to open-source low precision libraries such as AutoGPTQ, AutoAWQ, GPTQModel, Triton, Marlin, and
ExLLaMAV2 for providing low-precision CUDA kernels, which are leveraged in AutoRound.
