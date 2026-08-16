# vLLM 源码阅读规划

## 项目规模概览

| 指标 | 数值 |
|------|------|
| 源码目录 | `vllm/` |
| Python 文件 | ~2563 个 |
| 总代码行 | ~729,000 行 |
| 核心架构 | V1（多进程：API Server → Engine Core → GPU Worker） |
| 主要语言 | Python + CUDA C++ (kernels) |

## 阅读路线总览

本规划将 vLLM 源码分为 **15 章**，按依赖关系从底层到上层排列。前 6 章构成理解 vLLM 的最小必要知识，后续章节可按兴趣选读。

### 基础层（第 1-3 章）

| 章 | 文件 | 主题 | 一句话摘要 |
|----|------|------|-----------|
| [01](01-config-and-bootstrap.md) | 01-config-and-bootstrap.md | 配置与启动 | VllmConfig 统一配置体系，从 CLI 到各子系统配置的加载链路 |
| [02](02-core-data-structures.md) | 02-core-data-structures.md | 核心数据结构 | SamplingParams、Request、Outputs 等贯穿整个推理流程的数据定义 |
| [03](03-platform-and-plugins.md) | 03-platform-and-plugins.md | 平台抽象与插件 | Platform 接口如何屏蔽 CUDA/ROCm/CPU 差异，插件系统如何扩展 vLLM |

### 引擎层（第 4-6 章）

| 章 | 文件 | 主题 | 一句话摘要 |
|----|------|------|-----------|
| [04](04-entrypoints.md) | 04-entrypoints.md | 入口与 API 层 | LLM 离线推理、OpenAI API Server、CLI 三大入口的请求接入流程 |
| [05](05-engine-core.md) | 05-engine-core.md | Engine Core 调度核心 | V1 引擎核心的主循环：接收请求 → 调度 → 分发 → 返回结果 |
| [06](06-kv-cache-management.md) | 06-kv-cache-management.md | KV Cache 管理 | Block 分配/回收、前缀缓存的哈希匹配、LRU 淘汰策略 |

### 执行层（第 7-9 章）

| 章 | 文件 | 主题 | 一句话摘要 |
|----|------|------|-----------|
| [07](07-executor-and-worker.md) | 07-executor-and-worker.md | Executor 与 Worker | 多进程/Ray 执行器如何管理 GPU Worker，Worker 的生命周期 |
| [08](08-model-runner.md) | 08-model-runner.md | GPU Model Runner | 输入张量准备、模型前向执行、CUDAGraph 捕获与回放 |
| [09](09-attention-backends.md) | 09-attention-backends.md | Attention 后端 | FlashAttention/FlashInfer/MLA 等注意力后端的选择与调度机制 |

### 模型层（第 10-11 章）

| 章 | 文件 | 主题 | 一句话摘要 |
|----|------|------|-----------|
| [10](10-model-layers.md) | 10-model-layers.md | 模型构建层 | Linear/Attention/RMSNorm 等基础层如何支持 TP 分片与量化 |
| [11](11-compilation-and-ir.md) | 11-compilation-and-ir.md | 编译与 IR 系统 | torch.compile 集成、vLLM IR 中间表示、Fusion Pass 优化流水线 |

### 高级特性（第 12-15 章，按兴趣选读）

| 章 | 文件 | 主题 | 一句话摘要 |
|----|------|------|-----------|
| [12](12-sampling-and-output.md) | 12-sampling-and-output.md | 采样与输出处理 | Sampler、Logprobs、Detokenizer、结构化输出约束 |
| [13](13-speculative-decoding.md) | 13-speculative-decoding.md | 投机解码 | Draft 模型提议、验证采样、N-gram/Eagle/MTP 等多种策略 |
| [14](14-multimodal.md) | 14-multimodal.md | 多模态支持 | 图片/视频/音频输入的处理流水线、Encoder Cache 管理 |
| [15](15-distributed.md) | 15-distributed.md | 分布式推理 | TP/PP/DP 并行状态管理、KV 传输（disagg prefill）、DP Coordinator |

## 阅读建议

1. **先通读 `docs/design/arch_overview.md`** 建立架构全景，再按章节顺序进入源码
2. **每章附带关键问题**，建议带着问题读，读完后在 `## 详细笔记` 区域记录发现
3. **每章控制在 2000-5000 行代码**，一次阅读会话可完成一章
4. **第 1-6 章是前置依赖**，建议按顺序阅读；第 7 章以后可按兴趣跳读
5. 源码中的 `vllm/v1/` 是 V1 架构（当前主线），优先阅读

## 官方文档对照

| 章节 | 对应官方文档 |
|------|-------------|
| 第 1 章 | `docs/design/arch_overview.md`, `docs/design/huggingface_integration.md` |
| 第 3 章 | `docs/design/plugin_system.md` |
| 第 4 章 | `docs/design/arch_overview.md` (Entrypoints) |
| 第 5 章 | `docs/design/arch_overview.md` (V1 Process Architecture) |
| 第 6 章 | `docs/design/prefix_caching.md` |
| 第 8 章 | `docs/design/cuda_graphs.md`, `docs/design/model_runner_v2.md` |
| 第 9 章 | `docs/design/attention_backends.md`, `docs/design/paged_attention.md` |
| 第 10 章 | `docs/design/fused_moe_modular_kernel.md` |
| 第 11 章 | `docs/design/torch_compile.md`, `docs/design/vllm_ir.md`, `docs/design/fusions.md` |
| 第 13 章 | `docs/features/speculative_decoding/` |
| 第 14 章 | `docs/design/mm_processing.md`, `docs/features/multimodal_inputs.md` |
| 第 15 章 | `docs/design/multiprocessing.md`, `docs/features/disagg_prefill.md` |

## 使用说明

- 使用 source-reader Agent 阅读具体章节：`帮我阅读第5章`
- 阅读后的笔记会自动填充到各章骨架文件的 `## 详细笔记` 区域
- 如果发现章节划分不合理，可在骨架文件底部添加 `## 反馈给规划Agent`

## 规划会话记录

| 日期 | Session ID | 进度说明 |
|------|-----------|---------|
| 2026-08-10 | `pending` | 初始规划，完成 15 章骨架 |
