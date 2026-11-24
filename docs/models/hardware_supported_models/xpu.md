# XPU - Intel® GPUs

## Validated Hardware

| Hardware |
| -------- |
| [Intel® Arc™ Pro B-Series Graphics](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/workstations/b-series/overview.html) |

## Recommended Models

### Text-only Language Models

Note: Online FP8 = [Online Quantization](https://docs.vllm.ai/en/latest/features/quantization/online/) to FP8.

| Model                                           | Architecture                                     | Dtype                 |
| ----------------------------------------------- | ------------------------------------------------ | --------------------- |
| openai/gpt-oss-20b                              | GPTForCausalLM                                   | MXFP4                 |
| openai/gpt-oss-120b                             | GPTForCausalLM                                   | MXFP4                 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B        | LlamaForCausalLM                                 | BF16/Online FP8       |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B        | QwenForCausalLM                                  | BF16/Online FP8       |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | QwenForCausalLM                                  | BF16/Online FP8       |
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B       | LlamaForCausalLM                                 | BF16/Online FP8       |
| Qwen/Qwen2.5-72B-Instruct                       | Qwen2ForCausalLM                                 | BF16/Online FP8       |
| Qwen/Qwen3-14B                                  | Qwen3ForCausalLM                                 | BF16/Online FP8       |
| Qwen/Qwen3-32B                                  | Qwen3ForCausalLM                                 | BF16/Online FP8       |
| Qwen/Qwen3-30B-A3B                              | Qwen3ForCausalLM                                 | BF16/Online FP8       |
| Qwen/Qwen3-30B-A3B-FP8                          | Qwen3ForCausalLM                                 | FP8                   |
| Qwen/Qwen3-30B-A3B-GPTQ-Int4                    | Qwen3ForCausalLM                                 | Int4                  |
| Qwen/Qwen3-coder-30B-A3B-Instruct               | Qwen3ForCausalLM                                 | BF16/Online FP8       |
| Qwen/Qwen3-Next-80B-A3B-Instruct                | Qwen3NextForCausalLM                             | BF16/Online FP8       |
| Qwen/Qwen3-Next-80B-A3B-Thinking                | Qwen3NextForCausalLM                             | BF16/Online FP8       |
| Qwen/QwQ-32B                                    | QwenForCausalLM                                  | BF16/Online FP8       |
| deepseek-ai/DeepSeek-V2-Lite                    | DeepSeekForCausalLM                              | BF16/Online FP8       |
| deepseek-ai/DeepSeek-V4-Flash                   | DeepseekV4ForCausalLM                            | BF16/Online FP8       |
| meta-llama/Llama-3.1-8B-Instruct                | LlamaForCausalLM                                 | BF16/Online FP8       |
| microsoft/Phi-3.5-mini-instruct                 | Phi3ForCausalLM                                  | BF16/Online FP8       |
| THUDM/GLM-4-9B-chat                             | GLMForCausalLM                                   | BF16/Online FP8       |
| THUDM/CodeGeex4-All-9B                          | CodeGeexForCausalLM                              | BF16/Online FP8       |
| chuhac/TeleChat2-35B                            | LlamaForCausalLM (TeleChat2 based on Llama arch) | BF16/Online FP8       |
| 01-ai/Yi1.5-34B-Chat                            | YiForCausalLM                                    | BF16/Online FP8       |
| deepseek-ai/DeepSeek-Coder-33B-base             | DeepSeekCoderForCausalLM                         | BF16/Online FP8       |
| meta-llama/Llama-2-13b-chat-hf                  | LlamaForCausalLM                                 | FP16/Online FP8       |
| Qwen/Qwen1.5-14B-Chat                           | QwenForCausalLM                                  | BF16/Online FP8       |
| Qwen/Qwen1.5-32B-Chat                           | QwenForCausalLM                                  | BF16/Online FP8       |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic | LlamaForCausalLM                                 | FP8                   |
| thinkingmachines/Inkling-2-Mini (experimental)  | Inkling2LoopForConditionalGeneration             | NVFP4 + MXFP8         |

Note: Inkling-2-Mini on XPU is experimental and text-only: image inputs,
LoRA, DSpark speculative decoding and the FP8 KV cache are rejected (serve
it with `--language-model-only`). It was validated on Intel Arc Pro
B-series (B70) at TP1 with a BF16 KV cache. Its routed experts run NVFP4
weights with bf16 activations (W4A16; the checkpoint's
recipe quantizes them to NVFP4) and, on B-series, MXFP8 weights with bf16
activations (W8A16). Inkling-2-Small and Inkling-2-Small-MXFP8 are not
validated on XPU: only checkpoints cut to their first six layers were run.

### Multimodal Language Models

| Model                         | Architecture                         | Dtype           |
| ----------------------------- | ------------------------------------ | --------------- |
| OpenGVLab/InternVL3_5-8B      | InternVLForConditionalGeneration     | BF16/Online FP8 |
| OpenGVLab/InternVL3_5-14B     | InternVLForConditionalGeneration     | BF16/Online FP8 |
| OpenGVLab/InternVL3_5-30B-A3B | InternVLForConditionalGeneration     | BF16/Online FP8 |
| OpenGVLab/InternVL3_5-38B     | InternVLForConditionalGeneration     | BF16/Online FP8 |
| meta-models/Muse-Glimmer-30B  | MuseGlimmerForConditionalGeneration  | BF16/Online FP8 |
| Qwen/Qwen2-VL-7B-Instruct     | Qwen2VLForConditionalGeneration      | BF16/Online FP8 |
| Qwen/Qwen2.5-VL-72B-Instruct  | Qwen2VLForConditionalGeneration      | BF16/Online FP8 |
| Qwen/Qwen2.5-VL-32B-Instruct  | Qwen2VLForConditionalGeneration      | BF16/Online FP8 |
| Qwen/Qwen3-VL-32B-Instruct    | Qwen3VLForConditionalGeneration      | BF16/Online FP8 |
| Qwen/Qwen3.5-35B-A3B          | Qwen3_5MoeForConditionalGeneration   | BF16/Online FP8 |
| google/gemma-3-27b-it         | Gemma3ForConditionalGeneration       | BF16/Online FP8 |
| google/gemma-4-31B-it         | Gemma4ForConditionalGeneration       | BF16/Online FP8 |
| google/gemma-4-26B-A4B-it     | Gemma4ForConditionalGeneration       | BF16/Online FP8 |
| THUDM/GLM-4v-9B               | GLM4vForConditionalGeneration        | BF16/Online FP8 |
| openbmb/MiniCPM-V-4           | MiniCPMVForConditionalGeneration     | BF16/Online FP8 |

Qwen3.5 models on XPU use eager execution. MoE decode fusion is opt-in; see the
[configuration and supported deployments](https://github.com/vllm-project/vllm/blob/main/vllm/models/qwen3_5/xpu/specialized_decode/README.md).

### Embedding and Reranker Language Models

| Model                   | Architecture                   | Dtype           |
| ----------------------- | ------------------------------ | --------------- |
| Qwen/Qwen3-Embedding-8B | Qwen3ForTextEmbedding          | BF16/Online FP8 |
| Qwen/Qwen3-Reranker-8B  | Qwen3ForSequenceClassification | BF16/Online FP8 |
