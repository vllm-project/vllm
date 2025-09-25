# TPU

## Supported Models

### Text-only Language Models

| Model                                               | Architecture                   | Supported | Issues |
|-----------------------------------------------------|--------------------------------|-----------|--------|
| Qwen/Qwen3-8B, 32B                                  | Qwen3ForCausalLM               | ✅ | |
| Qwen/Qwen2.5-1.5B, 7B, 14B, 32B                     | Qwen2ForCausalLM               | ✅ | |
| mistralai/Mixtral-8x7B-Instruct-v0.1                | MixtralForCausalLM             | ✅ | |
| mistralai/Mistral-Small-24B                         | MistralForCausalLM             | ✅ | |
| mistralai/Codestral-22B-v0.1                        | MistralForCausalLM             | ✅ | |
| meta-llama/Llama-3.*-8B, 70B                        | LlamaForCausalLM               | ✅ | |
| google/gemma-3-27b                                  | Gemma3ForConditionalGeneration | ✅ | |
| microsoft/Phi-3-mini-128k                           | Phi3ForCausalLM                | 🟨 | |
| google/gemma-3-4b                                   | Gemma3ForConditionalGeneration | ❌ | |
| mistralai/Mixtral-8x22B-Instruct-v0.1               | MixtralForCausalLM             | ❌ | |
| meta-llama/Llama-4-*                                | Llama4ForConditionalGeneration | ❌ | |
| microsoft/phi-4                                     | Phi3ForCausalLM                | ❌ | |
| deepseek-ai/DeepSeek-R1, V3                         | DeepseekV3ForCausalLM          | ❌ | |
| openai/gpt-oss-20b, 120b                            | GptOssForCausalLM              | ❌ | |
| moonshotai/Kimi-K2                                  | DeepseekV3ForCausalLM          | ❌ | |


### Multimodal Language Models
| Model                                               | Architecture                        | Supported | Issues |
|-----------------------------------------------------|-------------------------------------|-----------|--------|
| Qwen/Qwen2.5-VL-7B                                  | Qwen2_5_VLForConditionalGeneration  | ❌ | |


✅ Runs and optimized.  
🟨 Runs and correct but not optimized to green yet.  
❌ Does not pass accuracy test or does not run.  
