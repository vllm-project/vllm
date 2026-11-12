# 源码目录地图

仓库规模很大，建议从问题所属的边界进入，而不是按字母顺序浏览。以下是顶层目录的功能地图；子目录职责以源码和各自的 `AGENTS.md` 为准。

## Python 主包：`vllm/`

| 路径 | 大致职责 | 阅读提示 |
| --- | --- | --- |
| `vllm/entrypoints/` | Python/CLI/API 服务入口 | 从公开调用找到初始化与请求转发 |
| `vllm/engine/` | Engine facade、参数和生命周期封装 | 区分外层 API 与 V1 核心 |
| `vllm/v1/engine/` | V1 engine core 与进程控制 | 追踪 request 到 scheduler 的边界 |
| `vllm/v1/core/` | 调度、KV Cache 和请求状态 | 从 `core/sched/` 开始理解执行计划 |
| `vllm/v1/executor/` | worker 执行拓扑 | 查看本地、多进程或其他 executor 的调度 |
| `vllm/v1/worker/` | worker/model runner、输入准备与执行 | `gpu_model_runner.py` 是常见 GPU 路径的中心入口 |
| `vllm/config/` | 配置对象、校验、组合与默认值 | 参数来源和生效前规范化通常都在这里 |
| `vllm/model_executor/models/` | 模型架构实现和模型注册相关代码 | 先找 registry，再选一个结构相近的模型读 |
| `vllm/model_executor/layers/` | 复用的 Transformer 层、线性层等 | 区分通用层和特定模型结构 |
| `vllm/attention/` | Attention 层与 backend 抽象/实现 | 不同 backend 有独立的能力约束 |
| `vllm/multimodal/` | 多模态输入结构、解析和通用处理 | 与模型目录中的 processor/encoder 实现配合阅读 |
| `vllm/distributed/` | 进程组、通信、并行状态 | 先明确 TP/DP/PP 等并行维度再读 |
| `vllm/compilation/` | PyTorch 编译集成及图执行辅助设施 | 与 `vllm/v1/cudagraph_dispatcher.py` 联读 |
| `vllm/collective/`、`vllm/platforms/` | 集合通信抽象和硬件平台适配 | 具体文件随设备后端有所不同 |

## Native、kernel 与构建

- **`csrc/`**：C++/CUDA kernel、绑定和设备实现；相邻的 `csrc/attention/`、`csrc/moe/` 等子目录常对应热点算子。
- **`vllm/vllm_flash_attn/`、`vllm/triton_utils/` 等**：部分 kernel/backend 或 Python 集成代码分布在 Python 包内，不要默认所有 GPU kernel 都在 `csrc/`。
- **`rust/`**：Rust crate，含独立的工具链、依赖和构建流程。改 Rust 前阅读 `rust/AGENTS.md`。
- **`cmake/`、`CMakeLists.txt`、`setup.py`**：本地扩展和构建配置；修改 C++/CUDA 时遵循[增量编译指南](../contributing/incremental_build.md)。

## 测试与辅助目录

- **`tests/`**：测试结构大体对应功能域，常见目录有 `v1/`、`models/`、`multimodal/`、`kernels/`、`distributed/`、`entrypoints/`。先读 `tests/AGENTS.md`，按风险选择最轻且真实的测试路径。
- **`benchmarks/`**：性能基准脚本与 kernel benchmarks。性能结论应由代表性工作负载支撑，不要用功能测试代替性能测量。
- **`examples/`**：面向用户的示例程序。
- **`recipes/`**：模型、硬件或部署组合的配置与实践。
- **`tools/`**：仓库维护、构建、CI 等开发工具。
- **`docs/`**：用户指南、设计文档、API/CLI 参考和贡献说明。

## 从源码开始读的几条链

- **服务请求**：CLI/API entrypoint → renderer/协议实现 → Engine Core → Scheduler → executor/worker → Model Runner → model/layers/backend。
- **加一个模型**：model registry → 同类模型实现 → `interfaces.py` /模型接口 → 模型权重加载 → 对应 model tests。
- **改一个配置**：`vllm/config/` 字段定义 → 参数解析与归一化 → 使用该字段的模块 → config/功能测试 → 文档。
- **改一个 kernel**：Python op/dispatch → kernel 实现和注册 → correctness tests → 定向 benchmark → build/CI 配置。

整体执行图见[架构与请求执行链](architecture.md)；模型扩展见[模型开发指南](../contributing/model/README.md)。