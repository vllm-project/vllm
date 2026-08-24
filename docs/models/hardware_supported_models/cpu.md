# CPU - Intel® Xeon®

!!! note "AMD Zen CPUs"
    On AMD Zen 4 / Zen 5 CPUs, AMD Zen optimizations are auto-enabled when the [`zentorch`](https://github.com/amd/ZenDNN-pytorch-plugin) package is installed. All models supported by vLLM on CPU are supported on AMD Zen as well; model compatibility does not change. This page reflects the current CPU reference validation matrix on Intel systems. See [AMD Zen optimizations](../../getting_started/installation/cpu.md#amd-zen-optimizations) for details.

## Validated Hardware

| Hardware |
| -------- |
| [Intel® Xeon® 6 Processors](https://www.intel.com/content/www/us/en/products/details/processors/xeon.html) |
| [Intel® Xeon® 5 Processors](https://www.intel.com/content/www/us/en/products/docs/processors/xeon/5th-gen-xeon-scalable-processors.html) |

## Deploy from a vLLM Recipe

The **Recipe** column below links to validated Xeon 6 configurations when
available. Use the Recipe conversion tool to generate `config.yaml` and
`env.sh`; see the
[Recipes conversion tool README](../../../tools/recipes/README.md) for usage.

Load the generated environment before starting vLLM:

```bash
source env.sh
vllm serve --config config.yaml
```

## Recommended Models

### Text-only Language Models

| Model | Architecture | Supported | Recipe |
| ------------------------------------ | ---------------------------------------- | --------- | ------ |
| openai/gpt-oss-20b | GptOssForCausalLM | ✅ | — |
| meta-llama/Llama-3.1-8B | LlamaForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/meta-llama/Llama-3.1-8B?hardware=xeon6) |
| meta-llama/Llama-3.1-8B-Instruct | LlamaForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/meta-llama/Llama-3.1-8B-Instruct?hardware=xeon6) |
| meta-llama/Llama-3.2-1B | LlamaForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/meta-llama/Llama-3.2-1B?hardware=xeon6) |
| meta-llama/Llama-3.2-1B-Instruct | LlamaForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/meta-llama/Llama-3.2-1B-Instruct?hardware=xeon6) |
| meta-llama/Llama-3.2-3B-Instruct | LlamaForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/meta-llama/Llama-3.2-3B-Instruct?hardware=xeon6) |
| meta-llama/Llama-3.3-70B-Instruct | LlamaForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/meta-llama/Llama-3.3-70B-Instruct?hardware=xeon6) |
| RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8 | LlamaForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/meta-llama/Llama-3.1-8B?hardware=xeon6&variant=w8a8_int8) |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 | LlamaForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/meta-llama/Llama-3.1-8B-Instruct?hardware=xeon6&variant=w8a8_int8) |
| RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8 | LlamaForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/meta-llama/Llama-3.2-1B-Instruct?hardware=xeon6&variant=w8a8_int8) |
| RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8 | LlamaForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/meta-llama/Llama-3.2-3B-Instruct?hardware=xeon6&variant=w8a8_int8) |
| RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8 | LlamaForCausalLM | ✅ | — |
| hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | LlamaForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/meta-llama/Llama-3.1-8B-Instruct?hardware=xeon6&variant=awq_int4) |
| AMead10/Llama-3.2-1B-Instruct-AWQ | LlamaForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/meta-llama/Llama-3.2-1B-Instruct?hardware=xeon6&variant=awq_int4) |
| AMead10/Llama-3.2-3B-Instruct-AWQ | LlamaForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/meta-llama/Llama-3.2-3B-Instruct?hardware=xeon6&variant=awq_int4) |
| TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ | LlamaForCausalLM | ✅ | — |
| TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ | LlamaForCausalLM | ✅ | — |
| ibm-granite/granite-3.2-2b-instruct | GraniteForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/ibm-granite/granite-3.2-2b-instruct?hardware=xeon6) |
| Qwen/Qwen3-1.7B | Qwen3ForCausalLM | ✅ | — |
| Qwen/Qwen3-4B | Qwen3ForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/Qwen/Qwen3-4B?hardware=xeon6) |
| Qwen/Qwen3-8B | Qwen3ForCausalLM | ✅ | — |
| Qwen/Qwen3-14B | Qwen3ForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/Qwen/Qwen3-14B?hardware=xeon6) |
| Qwen/Qwen3-14B-FP8 | Qwen3ForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/Qwen/Qwen3-14B?hardware=xeon6&variant=fp8) |
| Qwen/Qwen3-14B-AWQ | Qwen3ForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/Qwen/Qwen3-14B?hardware=xeon6&variant=awq) |
| Qwen/Qwen3-30B-A3B | Qwen3MoeForCausalLM | ✅ | — |
| Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 | Qwen3MoeForCausalLM | ✅ | — |
| Qwen/QwQ-32B | Qwen2ForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/Qwen/QwQ-32B?hardware=xeon6) |
| Qwen/QwQ-32B-AWQ | Qwen2ForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/Qwen/QwQ-32B?hardware=xeon6&variant=awq) |
| Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4 | Qwen2ForCausalLM | ✅ | — |
| RedHatAI/QwQ-32B-quantized.w8a8 | Qwen2ForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/Qwen/QwQ-32B?hardware=xeon6&variant=w8a8_int8) |
| zai-org/glm-4-9b-hf | GLMForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/zai-org/glm-4-9b-hf?hardware=xeon6) |
| google/gemma-7b | GemmaForCausalLM | ✅ | — |
| microsoft/Phi-4-reasoning | Phi3ForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/microsoft/Phi-4-reasoning?hardware=xeon6) |
| mistralai/Mistral-7B-Instruct-v0.2 | MistralForCausalLM | ✅ | — |
| TheBloke/Mistral-7B-Instruct-v0.2-AWQ | MistralForCausalLM | ✅ | — |

### Multimodal Language Models

| Model | Architecture | Supported | Recipe |
| ------------------------------------ | ---------------------------------------- | --------- | ------ |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | Llama4ForConditionalGeneration | ✅ | [Xeon 6](https://recipes.vllm.ai/meta-llama/Llama-4-Scout-17B-16E-Instruct?hardware=xeon6) |
| google/gemma-3-4b-it | Gemma3ForConditionalGeneration | ✅ | — |
| google/gemma-3-12b-it | Gemma3ForConditionalGeneration | ✅ | — |
| google/gemma-4-E4B-it | Gemma4ForConditionalGeneration | ✅ | [Xeon 6](https://recipes.vllm.ai/Google/gemma-4-E4B-it?hardware=xeon6) |
| google/gemma-4-E2B-it | Gemma4ForConditionalGeneration | ✅ | [Xeon 6](https://recipes.vllm.ai/Google/gemma-4-E2B-it?hardware=xeon6) |
| google/gemma-4-26B-A4B-it | Gemma4ForConditionalGeneration | ✅ | [Xeon 6](https://recipes.vllm.ai/Google/gemma-4-26B-A4B-it?hardware=xeon6) |
| microsoft/Phi-4-multimodal-instruct | Phi4MMForCausalLM | ✅ | [Xeon 6](https://recipes.vllm.ai/microsoft/Phi-4-multimodal-instruct?hardware=xeon6) |
| Qwen/Qwen2.5-VL-7B-Instruct | Qwen2VLForConditionalGeneration | ✅ | [Xeon 6](https://recipes.vllm.ai/Qwen/Qwen2.5-VL-7B-Instruct?hardware=xeon6) |
| Qwen/Qwen3-VL-30B-A3B-Instruct | Qwen3VLMoeForConditionalGeneration | ✅ | [Xeon 6](https://recipes.vllm.ai/Qwen/Qwen3-VL-30B-A3B-Instruct?hardware=xeon6) |
| openai/whisper-large-v3 | WhisperForConditionalGeneration | ✅ | [Xeon 6](https://recipes.vllm.ai/openai/whisper-large-v3?hardware=xeon6) |

✅ Runs and optimized.
