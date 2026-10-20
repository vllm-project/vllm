# 代表性 KV 负载与核心优化提案

## Why

当前性能证据主要来自 GSM8K rollout，尚未验证真实 coding agent、长文档多问和长短请求混合场景下，应用信息、KV 搬运与调度协同的净收益。需要在本地固定的小样本上形成可复现的缓存压力，通过阶段计时和操作结果定位限制收益的机制，再实现并验证核心优化。

## What Changes

- 加入 SWE-bench Verified + mini-swe-agent、long-doc-qa、coding agent 与短聊天混合三类负载；固定任务、输入、种子、到达与资源预算，使用小样本和并发形成压力，不运行全量数据集。
- 复用现有控制客户端、服务/Store 管理、回放和分析入口；复用相邻 TokenMoE 项目的任务清单及容器适配经验，关闭与本实验无关的 MoE 路由采集，按当前模型重新采集和编码。
- 同时提供真实模型/工具执行和固定工作量回放。真实执行保留任务结果与失败；回放固定完整请求输入、输出 token 数和工具间隔，单独报告性能，不计算回放解题分数。
- 比较 Native APC、相同 connector 的无 hint 对照、无本地策略的显式应用指令、quota-only 和完整协同；只在诊断点扩展单项消融、oracle、预测误差及控制延迟实验。
- 用原生基线校准容量充足与压力点，冻结测量矩阵；在调优任务与保留验证任务上分别执行分析和复测，避免只挑选获益的任务、轮次或配置。
- 将收益分析落实到可验证的工程修改：测量传输与重算成本、完整缓存组准备、准备到准入的间隔、真实复用、无效搬运及长短请求干扰，按证据修改应用控制、可选策略或其引擎协作。
- 在至少一个保留压力场景争取完整协同相对 Native APC 的吞吐提升达到 25%、相对 quota-only 达到 10%、相对本轮修改前冻结的完整协同达到 5%；同时检查短聊天尾延迟和容量充足开销。目标、测量有效性、功能检查分别记录，未达到性能目标时优化验收保持未完成。
- 补充可复现命令、机器可读结果和分析报告，保留负面结果；与现有 `programmable-kv-cache` 变更的验证证据关联，继续服务于同一个 PR。

## Capabilities

### New Capabilities

- `kv-representative-workloads`: 三类有界工作负载、真实运行与固定回放、小样本压力校准、同条件对照、分任务/类别性能与正确性证据。
- `kv-experiment-driven-optimization`: 基于实验归因的核心优化、应用信息增益验证、无效搬运和公平性诊断、保留样本复测及明确的性能验收。

### Modified Capabilities

无。`openspec/specs/` 尚无已归档的能力规格；本变更依赖进行中的 `programmable-kv-cache`，不覆盖其控制、状态和执行契约，也不替代其中尚未完成的验收。

## Impact

- 主要影响 `benchmarks/programmable_kv_cache/`、对应实验文档和现有相邻测试；优先扩展公共入口，避免为每类负载复制服务管理和结果协议。
- 经证据确认后可修改 `vllm/v1/kv_hints/`、scheduler 或 KV manager 的协作点；继续由引擎审批资源、scheduler 管理队列、现有 connector 执行搬运。
- 实验使用本地 Qwen3.8-27B BF16、TP=2、GPU 0/3、`gpu_memory_utilization=0.5`，记录实际 KV 容量、缓存组、CPU/Store 资源与源码版本。性能实验不使用小额固定 KV 字节覆盖。
- 新增的可选运行依赖为 mini-swe-agent 及已有容器运行时；SWE 数据使用本地 Verified 副本，long-doc 和短聊天使用固定的小型本地输入。依赖通过 uv/venv 管理。
- 设计对齐 [PR #62382](https://github.com/vllm-project/vllm/pull/62382)、[RFC #57103 的更新边界](https://github.com/vllm-project/vllm/issues/57103#issuecomment-5814964446)及 [Agentic API #367](https://github.com/vllm-project/agentic-api/issues/367)。继续复用 `KvHintsEnvelope`、`SET_PRIORITY`/`COPY_TO`/`MOVE_TO` 和关联完成结果；默认机制模式、可选本地策略、无 public Pin、KVCR 不透明转发均保留。
- 本轮交付为实验及证据支持的优化；不引入新公共协议、上下文编辑、scheduler 替换、Mooncake 项目修改、真实 Dynamo/llm-d 接入或完整 RL trainer，也不单独发布新 PR。
