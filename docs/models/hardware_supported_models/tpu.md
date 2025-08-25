# TPU

## Supported Models

### Text-only Language Models

| Model                                               | Architecture                   | Supported |
|-----------------------------------------------------|--------------------------------|-----------|
| mistralai/Mixtral-8x7B-Instruct-v0.1                | MixtralForCausalLM             | 🟨 |
| mistralai/Mistral-Small-24B-Instruct-2501           | MistralForCausalLM             | ✅ |
| mistralai/Codestral-22B-v0.1                        | MistralForCausalLM             | ✅ |
| mistralai/Mixtral-8x22B-Instruct-v0.1               | MixtralForCausalLM             | ❌ |
| meta-llama/Llama-3.3-70B-Instruct                   | LlamaForCausalLM               | ✅ |
| meta-llama/Llama-3.1-8B-Instruct                    | LlamaForCausalLM               | ✅ |
| meta-llama/Llama-3.1-70B-Instruct                   | LlamaForCausalLM               | ✅ |
| meta-llama/Llama-4-*                                | Llama4ForConditionalGeneration | ❌ |
| microsoft/Phi-3-mini-128k-instruct                  | Phi3ForCausalLM                | 🟨 |
| microsoft/phi-4                                     | Phi3ForCausalLM                | ❌ |
| google/gemma-3-27b-it                               | Gemma3ForConditionalGeneration | 🟨 |
| google/gemma-3-4b-it                                | Gemma3ForConditionalGeneration | ❌ |
| deepseek-ai/DeepSeek-R1                             | DeepseekV3ForCausalLM          | ❌ |
| deepseek-ai/DeepSeek-V3                             | DeepseekV3ForCausalLM          | ❌ |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8  | LlamaForCausalLM               | ✅ |
| RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w8a8 | LlamaForCausalLM               | ✅ |
| Qwen/Qwen3-8B                                       | Qwen3ForCausalLM               | ✅ |
| Qwen/Qwen3-32B                                      | Qwen3ForCausalLM               | ✅ |
| Qwen/Qwen2.5-7B-Instruct                            | Qwen2ForCausalLM               | ✅ |
| Qwen/Qwen2.5-32B                                    | Qwen2ForCausalLM               | ✅ |
| Qwen/Qwen2.5-14B-Instruct                           | Qwen2ForCausalLM               | ✅ |
| Qwen/Qwen2.5-1.5B-Instruct                          | Qwen2ForCausalLM               | 🟨 |

✅ Runs and optimized.  
🟨 Runs and correct but not optimized to green yet.  
❌ Does not pass accuracy test or does not run.  
