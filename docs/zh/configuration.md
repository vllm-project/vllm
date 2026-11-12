# vLLM 配置参数

## `vllm serve` 参数

自动从当前版本 CLI 参数定义生成每个选项的名称、说明和默认值。下表对应 `vllm serve --help` 中的配置组。

| 参数组 | 配置内容 |
| --- | --- |
| 位置参数 | `model_tag` 指定要加载的模型；可省略并使用 CLI 定义的默认模型。 |
| `options` | 通用 CLI 选项，包括显示帮助、按配置组/参数名检索帮助等入口。 |
| `Frontend` | HTTP 服务监听、API 路由、鉴权、模型别名、Chat 模板、CORS、请求/响应行为和服务进程设置。 |
| `ModelConfig` | 模型与 Tokenizer 标识、模型配置覆盖、任务类型、上下文长度、数据类型、量化和模型行为选项。 |
| `LoadConfig` | 权重加载格式、下载目录、加载并行度、权重预取和加载过程设置。 |
| `AttentionConfig` | Attention 后端选择、实现参数及相关内核行为。 |
| `MambaConfig` | Mamba/状态空间模型的后端和缓存配置。 |
| `StructuredOutputsConfig` | 结构化输出后端与约束解码配置。 |
| `ParallelConfig` | Tensor/Pipeline/Data/Expert 等并行维度、分布式执行器、节点通信和 worker 配置。 |
| `CacheConfig` | KV Cache 的容量、数据类型、块大小、交换空间和前缀缓存配置。 |
| `OffloadConfig` | 权重或状态卸载到 CPU 等设备的配置。 |
| `MultiModalConfig` | 多模态输入上限、处理器缓存、媒体处理和 Encoder 并行配置。 |
| `LoRAConfig` | LoRA 适配器数量、秩、额外词表和加载配置。 |
| `LoggingConfig` | vLLM 日志级别、格式、统计输出和请求日志配置。 |
| `ObservabilityConfig` | 指标、Tracing、端到端延迟及可观测性设置。 |
| `SchedulerConfig` | 请求调度策略、并发上限、批处理 token 预算和抢占行为。 |
| `CompilationConfig` | 模型编译、CUDA Graph 捕获和编译缓存配置。 |
| `KernelConfig` | 内核选择、实现及预热行为配置。 |
| `VllmConfig` | 跨多个配置组的 vLLM 全局选项及组合配置。 |

--8<-- "gen:serve-args"

## Engine 参数

自动从 `EngineArgs` 和 `AsyncEngineArgs` 的参数定义生成，供在线服务及离线 `LLM(...)` 使用。

--8<-- "gen:engine-args"

## YAML 配置

`vllm serve --config config.yaml` 加载服务配置。字段名使用 CLI 长选项名称（不带 `--`）；命令行值优先于 YAML 值，未指定时使用默认值。

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
host: "0.0.0.0"
port: 8000
tensor-parallel-size: 2
max-model-len: 8192
gpu-memory-utilization: 0.9
```

## 环境变量

环境变量清单由项目定义自动生成，包含变量说明和默认值。`VLLM_HOST_IP`、`VLLM_PORT` 用于 vLLM 内部通信，不是 API Server 的监听地址和端口。

```python
--8<-- "vllm/envs.py:env-vars-definition"
```
