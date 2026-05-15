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

| Model                                | Architecture                             | Supported |
| ------------------------------------ | ---------------------------------------- | --------- |
| meta-llama/Llama-3.1-8B-Instruct     | LlamaForCausalLM                         | ✅        |
| meta-llama/Llama-3.2-3B-Instruct     | LlamaForCausalLM                         | ✅        |
| ibm-granite/granite-3.2-2b-instruct  | GraniteForCausalLM                       | ✅        |
| Qwen/Qwen3-1.7B                      | Qwen3ForCausalLM                         | ✅        |
| Qwen/Qwen3-4B                        | Qwen3ForCausalLM                         | ✅        |
| Qwen/Qwen3-8B                        | Qwen3ForCausalLM                         | ✅        |
| zai-org/glm-4-9b-hf                  | GLMForCausalLM                           | ✅        |
| google/gemma-7b                      | GemmaForCausalLM                         | ✅        |

### Multimodal Language Models

| Model                                | Architecture                             | Supported |
| ------------------------------------ | ---------------------------------------- | --------- |
| Qwen/Qwen2.5-VL-7B-Instruct          | Qwen2VLForConditionalGeneration          | ✅        |
| openai/whisper-large-v3              | WhisperForConditionalGeneration          | ✅        |

✅ Runs and optimized.  
🟨 Runs and correct but not optimized to green yet.  
❌ Does not pass accuracy test or does not run.  
