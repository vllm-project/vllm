# vLLM 中文项目导读

这组文档帮助读者建立 vLLM 的整体心智模型，并把概念映射到英文设计文档、配置参考和源码。它是阅读导航，不是英文文档的逐篇翻译；具体行为、参数默认值和 API 字段以链接指向的当前文档及代码为准。

## 怎么读

| 页面 | 适合解决的问题 |
| --- | --- |
| [架构与请求执行链](architecture.md) | 一个请求如何从入口走到 GPU，再返回结果？ |
| [关键概念](concepts.md) | Prefill、KV Cache、Continuous Batching、并行策略分别是什么？ |
| [特性地图](features.md) | 常见推理优化和功能的文档、源码入口在哪里？ |
| [接口与功能面](interfaces.md) | 离线 API、服务 API、CLI 和内部接口如何区分？ |
| [配置与参数](configuration.md) | 参数从哪里传入，如何找到类型、默认值和生效位置？ |
| [源码目录地图](source_map.md) | 应该从哪个包、子系统或测试目录开始读？ |
| [开发与贡献](development.md) | 如何按项目约定改代码、补测试和更新文档？ |

## 推荐路线

初次接触项目：先读[架构](architecture.md)，再读[关键概念](concepts.md)和[特性地图](features.md)。准备使用 vLLM 时，转到[接口](interfaces.md)与[配置](configuration.md)；准备改代码时，从[源码地图](source_map.md)找到组件，再按[开发与贡献](development.md)的路线验证。

从具体代码问题开始也可以直接沿调用链走：

1. 找到用户入口或配置字段。
2. 沿调用跳到负责决策或状态变更的模块，而不只停在转发/注册代码。
3. 找到相邻测试，确认可观察行为和覆盖层级。
4. 阅读对应设计文档，再决定是否需要深入 CUDA、Triton 或 Rust 实现。

## 项目是什么

vLLM 是面向大语言模型推理与服务的系统。它不只是模型 `forward()`：请求解析、批次调度、KV Cache 管理、并行执行、模型适配、设备 kernel、API 服务与性能工具共同构成完整路径。

仓库变化很快，且同一特性可能同时有用户文档、设计说明、配置定义和实现代码。遇到差异时，用户可见行为先看相应用户文档；参数类型和默认值看配置定义；内部流程看设计文档与当前实现；测试说明哪些行为被项目实际验证。

## 主要参考

- [vLLM 架构总览](../design/arch_overview.md)
- [vLLM V1 指南](../usage/v1_guide.md)
- [Hugging Face 集成](../design/huggingface_integration.md)
- [开发贡献指南](../contributing/README.md)
- [仓库级 Agent/贡献约定](../../AGENTS.md)