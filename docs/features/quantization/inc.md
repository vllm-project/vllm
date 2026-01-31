# Intel Quantization Support

[AutoRound](https://github.com/intel/auto-round) is Intel’s advanced quantization algorithm designed for large language models(LLMs). It produces highly efficient **INT2, INT3, INT4, INT8, MXFP8, MXFP4, NVFP4**, and **GGUF** quantized models, balancing accuracy and inference performance. AutoRound is also part of the [Intel® Neural Compressor](https://github.com/intel/neural-compressor). For a deeper introduction, see the [AutoRound step-by-step guide](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md).

## Key Features

✅ Superior Accuracy Delivers strong performance even at 2–3 bits [example models](https://huggingface.co/collections/OPEA/2-3-bits)

✅ Fast Mixed `Bits`/`Dtypes` Scheme Generation Automatically configure in minutes

✅ Support for exporting **AutoRound, AutoAWQ, AutoGPTQ, and GGUF** formats

✅ **10+ vision-language models (VLMs)** are supported

✅ **Per-layer mixed-bit quantization** for fine-grained control

✅ **RTN (Round-To-Nearest) mode** for quick quantization with slight accuracy loss

✅ **Multiple quantization recipes**: best, base, and light

✅ Advanced utilities such as immediate packing and support for **10+ backends**

## Supported Recipes on Intel Platforms

On Intel platforms, AutoRound recipes are being enabled progressively by format and hardware. Currently, vLLM supports:

- **`W4A16`**: weight-only, 4-bit weights with 16-bit activations
- **`W8A16`**: weight-only, 8-bit weights with 16-bit activations

Additional recipes and formats will be supported in future releases.

## Quantizing a Model

### Installation

```bash
uv pip install auto-round
```

### Quantize with CLI

```bash
auto-round \
    --model Qwen/Qwen3-0.6B \
    --scheme W4A16 \
    --format auto_round \
    --output_dir ./tmp_autoround
```

### Quantize with Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "Qwen/Qwen3-0.6B"
autoround = AutoRound(model_name, scheme="W4A16")

# the best accuracy, 4-5X slower, low_gpu_mem_usage could save ~20G but ~30% slower
# autoround = AutoRound(model, tokenizer, nsamples=512, iters=1000, low_gpu_mem_usage=True, bits=bits, group_size=group_size, sym=sym)

# 2-3X speedup, slight accuracy drop at W4G128
# autoround = AutoRound(model, tokenizer, nsamples=128, iters=50, lr=5e-3, bits=bits, group_size=group_size, sym=sym )

output_dir = "./tmp_autoround"
# format= 'auto_round'(default), 'auto_gptq', 'auto_awq'
autoround.quantize_and_save(output_dir, format="auto_round")
```

## Deploying AutoRound Quantized Models in vLLM

```bash
vllm serve Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096
```

!!! note
     To deploy `wNa16` models on Intel GPU/CPU, please add `--enforce-eager` for now.

## Evaluating the Quantized Model with vLLM

```bash
lm_eval --model vllm \
  --model_args pretrained="Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enforce_eager=True" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size 128
```
