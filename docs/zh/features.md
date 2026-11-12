# 特性地图

按问题寻找特性入口比按目录猜测更快。本页只做分类导航；兼容矩阵、支持范围和当前选项请以英文特性文档为准。

## 吞吐与延迟

- **Continuous batching / chunked prefill**：改善请求混合负载下的设备利用率和长 prompt 调度。先看[优化指南](../configuration/optimization.md#chunked-prefill)，再看[架构中的 Scheduler](../design/arch_overview.md#scheduler)。
- **Prefix caching**：复用相同 prompt 前缀的 KV Cache。入口：[用户文档](../features/automatic_prefix_caching.md)、[设计文档](../design/prefix_caching.md)。
- **Speculative decoding**：使用 draft/proposer 提议 token，再由 target 模型验证。入口：[Speculative Decoding](../features/speculative_decoding/README.md)。
- **CUDA Graph 与 `torch.compile`**：减少调度开销或优化计算图。入口：[CUDA Graph](../design/cuda_graphs.md)、[`torch.compile`](../design/torch_compile.md)。

## 显存与模型表示

- **KV Cache 管理与 offload**：影响可容纳上下文和请求数；查看[显存优化配置](../configuration/conserving_memory.md)及[KV Cache/前缀缓存设计](../design/prefix_caching.md)。
- **量化**：改变权重或 KV Cache 的表示方式，可能带来显存、吞吐和精度权衡。入口：[量化文档](../features/quantization/README.md)。
- **LoRA**：在基础模型上加载适配器。入口：[LoRA 文档](../features/lora.md)。

## 模型任务与输入类型

- **多模态输入**：图像、视频、音频等输入与模型架构相关。入口：[多模态特性](../features/multimodal_inputs.md)、[支持模型列表](../models/supported_models.md)、[处理设计](../design/mm_processing.md)。
- **Pooling 任务**：embedding、classification、scoring/reranking 等非生成任务。入口：[Pooling Models](../models/pooling_models/README.md)。
- **Structured outputs**：限制或解析生成结果以满足结构化格式要求。入口：[Structured Outputs](../features/structured_outputs.md)。
- **工具调用、推理内容与输出解析**：涉及模型模板和服务层 parser。可从 [Features](../features/README.md) 以及 `vllm/tool_parsers/`、`vllm/reasoning/` 开始。

## 分布式和部署

- **TP / PP / DP / EP**：模型切分或请求复制的不同维度。入口：[并行部署](../serving/parallelism_scaling.md)及各[部署文档](../deployment/)。
- **Disaggregated Prefill/Decode**：分离不同推理阶段的部署方式。入口：[Disaggregated Prefill](../features/disagg_prefill.md)。
- **插件**：扩展后端、接口或模型能力。入口：[插件系统设计](../design/plugin_system.md)。

## 如何判断一个特性从哪里开始读

1. 先读用户文档，确认场景、限制和公开开关。
2. 查[配置参数](configuration.md)，找对应配置类及来源。
3. 从[源码地图](source_map.md)定位实现层，搜索配置字段名或公开 API 名称。
4. 找对应测试目录，确认验证覆盖的是纯逻辑、模型集成、服务入口还是设备 kernel。

特性和硬件的兼容关系会变化；不要仅凭“文档中存在该特性”推断所有模型/后端都支持。总览矩阵见 [Features](../features/README.md)。