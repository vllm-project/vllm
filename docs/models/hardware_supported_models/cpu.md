# CPU - Intel® Xeon®

!!! note "AMD Zen CPUs"
    On AMD Zen 4 / Zen 5 CPUs, AMD Zen optimizations are auto-enabled when the [`zentorch`](https://github.com/amd/ZenDNN-pytorch-plugin) package is installed. All models supported by vLLM on CPU are supported on AMD Zen as well; model compatibility does not change. This page reflects the current CPU reference validation matrix on Intel systems. See [AMD Zen optimizations](../../getting_started/installation/cpu.md#amd-zen-optimizations) for details.

## Validated Hardware

| Hardware |
| -------- |
| [Intel® Xeon® 6 Processors](https://www.intel.com/content/www/us/en/products/details/processors/xeon.html) |
| [Intel® Xeon® 5 Processors](https://www.intel.com/content/www/us/en/products/docs/processors/xeon/5th-gen-xeon-scalable-processors.html) |

## Recommended Models

### Text-only Language Models

| Model | Architecture | Supported |
| ------------------------------------ | ---------------------------------------- | --------- |
| unsloth/gpt-oss-20b | GptOssForCausalLM | ✅ |
| meta-llama/Llama-3.1-8B-Instruct | LlamaForCausalLM | ✅ |
| meta-llama/Llama-3.2-1B | LlamaForCausalLM | ✅ |
| meta-llama/Llama-3.2-3B-Instruct | LlamaForCausalLM | ✅ |
| meta-llama/Llama-3.3-70B-Instruct | LlamaForCausalLM | ✅ |
| RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8 | LlamaForCausalLM | ✅ |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 | LlamaForCausalLM | ✅ |
| RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8 | LlamaForCausalLM | ✅ |
| RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8 | LlamaForCausalLM | ✅ |
| RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8 | LlamaForCausalLM | ✅ |
| hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | LlamaForCausalLM | ✅ |
| AMead10/Llama-3.2-1B-Instruct-AWQ | LlamaForCausalLM | ✅ |
| AMead10/Llama-3.2-3B-Instruct-AWQ | LlamaForCausalLM | ✅ |
| TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ | LlamaForCausalLM | ✅ |
| TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ | LlamaForCausalLM | ✅ |
| ibm-granite/granite-3.2-2b-instruct | GraniteForCausalLM | ✅ |
| Qwen/Qwen3-1.7B | Qwen3ForCausalLM | ✅ |
| Qwen/Qwen3-4B | Qwen3ForCausalLM | ✅ |
| Qwen/Qwen3-8B | Qwen3ForCausalLM | ✅ |
| Qwen/Qwen3-14B | Qwen3ForCausalLM | ✅ |
| Qwen/Qwen3-14B-AWQ | Qwen3ForCausalLM | ✅ |
| Qwen/Qwen3-30B-A3B | Qwen3MoeForCausalLM | ✅ |
| Qwen/QwQ-32B-AWQ | Qwen2ForCausalLM | ✅ |
| Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4 | Qwen2ForCausalLM | ✅ |
| RedHatAI/QwQ-32B-quantized.w8a8 | Qwen2ForCausalLM | ✅ |
| zai-org/glm-4-9b-hf | GLMForCausalLM | ✅ |
| google/gemma-7b | GemmaForCausalLM | ✅ |
| microsoft/Phi-4-reasoning | Phi3ForCausalLM | ✅ |
| TheBloke/Mistral-7B-Instruct-v0.2-AWQ | MistralForCausalLM | ✅ |

### Multimodal Language Models

| Model | Architecture | Supported |
| ------------------------------------ | ---------------------------------------- | --------- |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | Llama4ForConditionalGeneration | ✅ |
| google/gemma-3-4b-it | Gemma3ForConditionalGeneration | ✅ |
| google/gemma-3-12b-it | Gemma3ForConditionalGeneration | ✅ |
| google/gemma-4-E4B-it | Gemma4ForConditionalGeneration | ✅ |
| google/gemma-4-E2B-it | Gemma4ForConditionalGeneration | ✅ |
| google/gemma-4-26B-A4B-it | Gemma4ForConditionalGeneration | ✅ |
| microsoft/Phi-4-multimodal-instruct | Phi4MMForCausalLM | ✅ |
| Qwen/Qwen2.5-VL-7B-Instruct | Qwen2VLForConditionalGeneration | ✅ |
| openai/whisper-large-v3 | WhisperForConditionalGeneration | ✅ |

✅ Runs and optimized.
