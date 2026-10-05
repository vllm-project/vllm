# Programmable KV Cache 后续工作

更新日期：2026-10-01。

本文记录首版之后要做的工作、前置依赖和验收标准。首版实现任务以
[OpenSpec tasks](../../openspec/changes/programmable-kv-cache/tasks.md) 为准；
本文不表示首版代码已经完成，也不把尚未完成的首版验收挪成“未来工作”。

## 已确认的首版边界

首版主线：

```text
轻量 orchestrator
  → 已合入的 KvHintsEnvelope / KvHintAction
  → vLLM transport + dispatch
  → vLLM 解析 native action 语义 / 可选策略消费 signal
  → 目标解析、资源审批、buffer/reference 管理
  → vLLM connector 调用 Mooncake Store
  → 实际完成与 residency 更新
  → 操作结果、增量关联事件、状态查询返回 orchestrator
```

同时保留现有 KVCR 路径：envelope 透传给 KVCR，由 KVCR 解释其 payload，
复用现有 `submit_hint`、query/deliver 和请求清理机制。

- **每个 envelope 只走一条解释路径**。首版不在同一 envelope 内混合 native
  和 KVCR action，不广播给两个解释方，也不失败后悄悄换解释方。
- 沿用已合入的协议字段；不新增自定义 hint 注册器或 handler 插件系统。
- vLLM 定义和执行 native Retain/Demote 等语义；Mooncake 校验自身存储状态和
  容量，执行其已有 API 支持的工作。首版不修改 Mooncake 项目。
- 两条路径需要 GPU 操作时都遵守 vLLM 的资源与生命周期控制；这不等于已经
  实现图中虚线标注的“未来 KVCR G1 directives”。
- 显式 session/continuation 状态命令与策略 signal 分开；默认没有本地策略时，
  仍可执行支持的显式命令。无人消费的 native signal 无副作用。
- 关联信息使用增量事件与状态查询。现有 full-report 只是当前请求命中前缀的
  补报，不是全 engine 状态快照。

以下仍然是**首版要求**，不能因困难移入后续清单：

| 首版要求 | 验收说明 |
| --- | --- |
| 完整轻量 orchestrator → vLLM → Mooncake 流程 | 必须经过实际入口、语义解析与真实 Store 调用；直接调用 adapter 或只有 mock 不算完整验收 |
| 无推理请求的 native 控制 | tool wait 期间可推进操作；不构造伪推理请求 |
| Mooncake 跨 engine 示例 | A 保存、B 提前加载，完成后观察 B 的实际 cache reuse |
| SessionPrefixIndex 与生命周期 | 共享前缀、分支、range、独立 Retain 声明、显式任务树结束 |
| 统一 GPU 资源管理 | 引用、预留、取消、部分失败、reset/restart、陈旧完成通知 |
| 可配置策略与 scheduler 协作 | quota/admission、request-score 示例；保留现有抢占算法 |
| 两条解释路径兼容 | native 不重复透传；KVCR payload 保持语义不变；原请求清理不退化 |
| 实验与正确性 | 同权重版本 rollout；权重更新后旧 KV 失效；公平对比和负收益场景 |
| 验证环境分层 | 本机 Qwen3.8-27B TP=2 调试；其他机器的 GPT-OSS-120B、DeepSeek-V4-Flash 全面评测仍需单独记录实际结果 |

## 已确认放到后续的工作

### F1. Dynamo → vLLM-native → Mooncake 实际集成

**状态：已确认后续接入。首版用轻量 orchestrator。**

依赖：首版 native action/payload、能力查询、操作结果和状态事件契约可运行。

- [ ] 在 `../` 下准备 Dynamo 对接仓库并固定兼容 revision；检查当时已有的
  hint 生产与 engine bridge，复用已有传输。
- [ ] 将轻量 orchestrator 中的 hint 发送流程接到真实 Dynamo worker 选择、
  session/continuation 及生命周期信息来源。
- [ ] 把原 envelope 字段和关联 ID 传到指定 vLLM engine；明确独立控制如何
  到达目标实例，以及 native/KVCR 路径如何选择。
- [ ] 让 Dynamo 消费能力、实际操作结果、关联/residency 增量；处理 DEFERRED、
  缺失目标、失败、重试与 engine epoch 变化。
- [ ] 验证 tool wait → Demote → 选择恢复 engine → Prefetch → readiness → resume。
- [ ] 对比轻量发送端与 Dynamo 的行为，分别报告 cache 收益和 router/control-loop
  开销，不能将所有差异归因于 KV 策略。

**完成标准：** 可复现地运行真实 Dynamo → vLLM → Mooncake；结果能追溯到原始
hint；压力、共享前缀、事件延迟和 worker 重启下仍有正确回退。记录必要的
Dynamo 改动与双方版本，不将轻量 orchestrator 示例称为 Dynamo 集成。

### F2. 同一 envelope 内混合 native 与 backend action

**状态：明确不属于首版；作为后续分发能力设计。**

依赖：首版整 envelope 单路径分发和两个 consumer 的能力/错误边界稳定。

- [ ] 与相关实现方确定每个 action 的唯一解释方选择规则；不能只凭同名
  `action_type` 推断两个 consumer 具有相同语义。
- [ ] 设计分组转发、action 原始顺序、跨组依赖和 message/action ID 关联。
- [ ] 定义混合 envelope 的部分失败、取消、重试及结果汇总；不默认承诺跨
  consumer 原子事务。
- [ ] 处理未知/歧义归属；防止 native 消费后被原 KVCR hook 再执行一遍。
- [ ] 验证拆分后 backend payload 的值与版本不变，并测量额外控制开销。

**完成标准：** 一个混合 envelope 中的每个 action 只被选定 consumer 解释，
结果可独立关联；失败、重试与取消不会重复执行或重复计费。协议字段是否需要
演进必须另行协调，不能把新字段暗中塞进首版。

## 图中已提出、但尚未承诺本次实现的后续方向

### F3. KVCR G1 directives 接入共享 GPU 操作

**状态：图中的虚线路径；需与 KVCR 团队明确具体 action 后立项。**

依赖：明确的 backend action/结果契约与可复用的 vLLM GPU resource operations。

- [ ] 确定首批 KVCR G1 action 及由谁解释；保持 KVCR 对其 payload 的解释权。
- [ ] 从 integration 调用 engine 的 GPU 操作，避免再经过 native payload 解析。
- [ ] 定义目标块与兼容性、capacity reservation、source/destination holds、
  多 rank 完成、取消和 failure 回报。
- [ ] 区分现有 framework source pin 回调与新增 GPU 操作；不要把内部传输引用
  当作已删除的公共 Pin 语义。

**完成标准：** 至少一个明确的 KVCR G1 action 能经 integration 正确执行；
并发 native 工作与 KVCR 工作不会重复占用容量、提前释放 buffer 或伪报 readiness。
现有 `submit_hint()` 透传本身不能满足这一验收。

### F4. KVCR 独立于推理请求的控制生命周期

**状态：候选扩展；当前 request-scoped 路径不能自动宣称支持。**

依赖：KVCR 能力和接口约定；不能仅给现有接口编造 request ID。

- [ ] 确认 KVCR 是否需要、以及如何支持 request-free hint 的提交与进度推进。
- [ ] 定义独立 operation handle、存活期限、取消、清理及结果关联。
- [ ] 明确无消费者/无后续请求时的资源回收和后端断连行为。

**完成标准：** 无推理请求时可推进支持的控制操作，结束后无泄漏；forwarded
与 completed/available 有明确区别。若后端不支持，应继续准确报告 unsupported。

## 需要另行决定的扩展，不属于已承诺路线

| 编号 | 候选方向 | 开始前需确定 | 完成标准 |
| --- | --- | --- | --- |
| F5 | gRPC 独立控制入口 | 对接方是否需要；与现有 Python/HTTP 控制的兼容面 | 复用相同 envelope 与状态记录，跨入口结果一致，明确取消/流控/engine targeting |
| F6 | SimpleCPUOffloadConnector 控制适配 | 是否确有使用需求及与通用 CPU 路径的维护成本 | 实际支持能力可发现；生命周期、引用和失败测试达到同等标准 |
| F7 | 更丰富的冷层 Retain/选择性删除能力 | 后端是否提供所需 priority/TTL/delete 语义；共享对象的声明归属 | 可验证地满足声明的保证；不以弱保留冒充强保证，不以全 Store reset 实现局部 Evict |
| F8 | 运行中更换策略/connector | 用户是否需要；状态交接、drain、rollback 和 worker 一致性方案 | 在有在途操作时切换也不泄漏、不误释放、不破坏有效声明 |

F7 如需要修改 Mooncake 本身，属于另一个需明确授权/协调的工作项，不在首版范围。
F8 不是当前“配置式插件”的隐含要求；首版更换插件需重启。

## 使用与更新规则

1. 每项后续工作落地时记录实现仓库/commit/PR、测试命令、实际结果及剩余限制。
2. 首版仍未确定的 payload、scope、retry、range/group 语义继续在 OpenSpec
   design/tasks 中解决，不因未定而自动归入未来。
3. 首版代码或实验失败继续保持未完成状态；不能通过改写本文宣称首版完成。
4. 新增工作项须注明是“已确认后续”“需要与上游协调”还是“候选待决定”。

## 关联文档

- [实现主线与双路径职责](../../openspec/changes/programmable-kv-cache/integration-paths.md)
- [设计](../../openspec/changes/programmable-kv-cache/design.md)
- [首版开发任务](../../openspec/changes/programmable-kv-cache/tasks.md)
- [能力覆盖与示例](../../openspec/changes/programmable-kv-cache/coverage.md)
- [原 RFC](%5BRFC%5Dprogrammable-kv-cache.md)
- [本地讨论记录](chats.md)

双路径依据还包括用户本次提供的 Karen/Zhuohang 讨论及
“Same typed envelope; different payload interpreters”图。该图中 shared dispatch
和 future KVCR G1 hooks 是设计要求/未来方向，不能据图声称已合入。

## 首版交付对账（2026-10-06 修正）

已实现的控制、逻辑索引、声明与 CPU/Mooncake 搬运有单元测试以及部分本地
Qwen 真实链路证据。具体证据与限制见 OpenSpec `coverage.md` 和
`validation.md`。策略/调度有针对性回归；其完整压力、收益和公平性矩阵未完成。

10 月 2 日的完成度统计高估了交付范围：空目标 rollout、未对接真实格式的
verl 脚本，以及仅检查文件存在的 coverage 脚本已移除。协议错误探针已补齐
断言，但伪造过期 epoch 不等于实际重启恢复。以下仍是首版未完成项：

- 物理 residency 查询/事件与完整闭环恢复；
- 多轮共享前缀压力、完整 branch/range/task-tree 示例与应用 rollout 控制器；
- verl Agent Loop 集成及 live weight-version 下旧 KV 的失效；
- native policy 到真实 Mooncake 完成/复用的验收；
- 策略对比、端到端开销、公平性及外部 GPT-OSS/DeepSeek 评测；
- 成功的 GitHub 查重和满足完整证据要求的 PR 描述。

这些内容保留在 OpenSpec 首版任务中。F1–F8 的后续边界不变。
