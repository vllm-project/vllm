# CPU - Intel® Xeon®

## Supported Models

### Text-only Language Models

| Model                                | Architecture                             | Supported |
|--------------------------------------|-------------------------------------------|-----------|
| meta-llama/Llama-3.1 / 3.3           | LlamaForCausalLM                          | ✅        |
| meta-llama/Llama-4-Scout             | Llama4ForConditionalGeneration            | ✅        |
| meta-llama/Llama-4-Maverick          | Llama4ForConditionalGeneration            | ✅        |
| ibm-granite/granite (Granite-MOE)    | GraniteMoeForCausalLM                     | ✅        |
| Qwen/Qwen3                           | Qwen3ForCausalLM                          | ✅        |
| zai-org/GLM-4.5                      | GLMForCausalLM                            | ✅        |
| google/gemma                         | GemmaForCausalLM                          | ✅        |

### Multimodal Language Models

| Model                                | Architecture                             | Supported |
|--------------------------------------|-------------------------------------------|-----------|
| Qwen/Qwen2.5-VL                      | Qwen2VLForConditionalGeneration           | ✅        |
| openai/whisper                       | WhisperForConditionalGeneration           | ✅        |

✅ Runs and optimized.  
🟨 Runs and correct but not optimized to green yet.  
❌ Does not pass accuracy test or does not run.  
