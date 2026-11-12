# 接口与功能面

vLLM 有多个面向不同使用者的入口。它们共享推理核心，但输入处理、生命周期和返回形式并不相同。

## 离线 Python API

`vllm.LLM` 适合在 Python 程序中直接加载模型并执行推理。生成模型常用 `generate` / `chat`；Pooling 模型使用 `embed`、`classify`、`score` 或 `encode` 等任务接口。先看[离线推理指南](../serving/offline_inference.md)，再查 [Python API Reference](../api/README.md)。

源码阅读入口：[LLM 类](../../vllm/entrypoints/llm.py)。从这里继续跟到 Engine 初始化、输入构造和 V1 核心，不要假设这个 facade 本身实现了 scheduler 或 GPU forward。

## 在线服务 API

`vllm serve` 启动 HTTP 服务。仓库提供多类兼容接口和 vLLM 自有接口，涵盖生成、聊天/Responses、embedding、音频、打分/重排、分类、metrics 等任务；具体支持取决于模型类型和服务配置。入口见[在线服务总览](../serving/online_serving/README.md)。

阅读接口时要区分三层：

1. **路由与协议模型**：请求、响应和错误的结构。
2. **Serving 实现**：如何把协议请求映射为 vLLM 内部输入与参数。
3. **Engine API**：实际执行推理的内部调用。

OpenAI/Anthropic 兼容面是对外部规范的实现；不要从内部 Engine API 推断它们的字段含义或默认值。具体端点及字段以各端点文档和对应上游规范为准：[OpenAI Compatible Server](../serving/online_serving/openai_compatible_server.md)、[Anthropic API](../serving/online_serving/README.md)。

## CLI

`vllm` 命令提供服务启动、聊天/补全客户端、benchmark 和运维工具等子命令。先用 `vllm --help` 查看本地版本，再看 [CLI Guide](../cli/README.md) 与具体命令参考。CLI 一般是配置解析和组件启动层，不是推理核心的实现位置。

源码入口：[CLI 主程序](../../vllm/entrypoints/cli/main.py)。

## 配置对象与内部 API

引擎配置由多类配置对象组合，主要入口包括 `EngineArgs`、`VllmConfig` 和各类子配置。内部使用者还可以通过 Engine 类或特定控制 API 管理请求和引擎。参考 [API Reference](../api/README.md)，并阅读 [Engine Arguments](../configuration/engine_args.md)。

## 选择入口

| 目标 | 建议入口 |
| --- | --- |
| 单机脚本、离线批处理 | `vllm.LLM` |
| 应用通过 HTTP 调用推理 | `vllm serve` 与对应在线 API |
| 启动、检查参数、运行基准 | `vllm` CLI |
| 添加/改造模型架构 | 模型实现接口与[模型开发指南](../contributing/model/README.md) |
| 接入新的硬件或扩展执行流程 | 插件文档及对应后端项目 |

## 延伸阅读

- [离线推理](../serving/offline_inference.md)
- [在线服务](../serving/online_serving/README.md)
- [CLI](../cli/README.md)
- [API Reference](../api/README.md)
- [模型支持列表](../models/supported_models.md)