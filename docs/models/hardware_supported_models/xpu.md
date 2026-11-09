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

### Qwen3.5 eager execution and MoE decode fusion

Qwen3.5 dense, MoE and MTP models on XPU use the dedicated eager implementation.
Breakable graphs are enabled by default for these architectures, and model-level
`torch.compile` is disabled. No environment variable is required to select this
execution path. ROCm and other fallback platforms retain their existing model
implementations.

Routed/shared expert decode fusion is **opt-in**. The default
`--additional-config '{"decode_backend":"auto"}'` uses stock MoE without
initializing a specialized executor, allocating its scratch or importing the
optional shared-expert provider. Dense and MTP models keep the default path.

The initial qualified deployment is two Arc Pro B70 GPUs (`device_id=0xE223`)
with FP16/block-2D capabilities, Qwen3.5-family MoE dimensions H=2048, routed and
shared I=512, E=256 and top-k=8, 40 layers, single-host mp, TP2/PP1/DP1/EP1, FP16 activations and online
`fp8_per_tensor` quantization. Routed/shared FP8-e4m3 weights use FP32 scales;
shared projections must use `XPUW8A16FP8LinearKernel`. Tested software includes
driver `1.15.39122+11`, Level-Zero V2 runtime `20.2.0`, PyTorch `2.14.0+xpu`,
and a companion build exposing `XpuFusedMoeWithSharedExpert`, whose tested
grouped-GEMM binary records oneAPI DPC++/C++ `2026.1.1` and GCC `13.3.1`.

Explicit opt-in fails startup with collected TP-rank reasons for an unsupported
deployment. Initially unqualified combinations include multimodal inputs,
speculative decoding, LoRA, prefix caching, sleep/wake, external KV/weight
transfer, prompt embeddings, alternate KV/SSM paths, offload, DBO, EPLB, nontrivial context/expert/sequence parallelism,
W8A8 shared projections and configured context lengths above 40960.
Default XPU functionality remains independent of this optional specialization.

For a qualified checkpoint and companion provider:

```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
  --tensor-parallel-size 2 --dtype float16 --quantization fp8_per_tensor \
  --language-model-only --no-enable-prefix-caching --max-model-len 40960 \
  --additional-config '{"decode_backend":"latency_specialized"}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}'
```

Graph-off execution uses
`--compilation-config '{"mode":0,"cudagraph_mode":"NONE"}'`.
Both modes use Model Runner V2 through `XPUModelRunner` and the standard hybrid
KV/SSM state. The model's optional ModelState plans MoE from CPU batch metadata.
Only ordinary decode with one query token per request qualifies:
`1 <= B = T_actual <= T_padded <= 8`, where T_padded includes graph padding.
Prefill, chunked prefill, mixed batches and larger decode batches select stock
before execution. One-token prompt tails cannot replay specialized decode
graphs. Runtime errors propagate without retrying through stock.

Both paths read the same canonical parameters and KV/SSM state; switching
paths requires no weight or state conversion. All row workspaces are prepared
during post-load before memory profiling and KV-cache sizing. Supported
in-place kernel-format reloads reuse executor/storage/workspaces; changes to
storage/layout or eligibility are rejected. Broader rebinding is unqualified.

Runtime diagnostics distinguish eligible planning, skip reasons, initialization
scratch and weight-storage sharing. Python launch counts include capture and
do not count native graph replay. Validation observes actual graph replay calls
and eligible batch plans, checks native capture launches, and observes bound
provider CPU operator events. Separate lightweight processes provide eager GPU
kernel traces. Enabling PTI GPU tracing in a model worker can time out later
whole-model graph replays on the tested driver, including on the default path,
even after stopping the GPU profiler; whole-model device tracing remains a gap.
Fused/stock accumulation and rounding may differ. Report numerical/model
quality, transition correctness, same-source serving performance and memory
costs for each claimed configuration; do not infer benefit for unmeasured
hardware or checkpoints.

### Embedding and Reranker Language Models

| Model                   | Architecture                   | Dtype           |
| ----------------------- | ------------------------------ | --------------- |
| Qwen/Qwen3-Embedding-8B | Qwen3ForTextEmbedding          | BF16/Online FP8 |
| Qwen/Qwen3-Reranker-8B  | Qwen3ForSequenceClassification | BF16/Online FP8 |
