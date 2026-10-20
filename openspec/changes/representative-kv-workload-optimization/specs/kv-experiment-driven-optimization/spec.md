# 实验驱动的 KV 优化规格

## ADDED Requirements

### Requirement: Preserve the programmable-cache authority boundary

优化 SHALL 复用 `KvHintsEnvelope`、现有显式原语和关联结果，默认机制模式与可选本地策略保持独立。策略 SHALL 通过共同资源审批提出建议，scheduler 保持队列与抢占所有权，现有 adapter 执行搬运。优化 MUST 保留 KVCR 不透明转发、有效引用、共享容量去重、所有 rank/缓存组完成后发布及失败重算路径，禁止引入 public Pin、替换 scheduler、修改 Mooncake 项目或改变模型上下文语义。

#### Scenario: An explicit operation conflicts with policy advice

- **WHEN** 应用显式指令与本地策略建议不一致或资源不足
- **THEN** 引擎按原有契约执行、延期、拒绝或报告实际部分范围，策略不能静默改写应用命令

#### Scenario: A multi-group preparation is incomplete

- **WHEN** 一组或一个 rank 尚未完成，或目标 checkpoint 无法组成兼容前缀
- **THEN** 系统不声明完整可复用，继续保护在途缓冲区并按结果协调等待、取消或重算

### Requirement: Policies use only available application information

正常优化 SHALL 仅使用决策时已知的工具类型、生命周期、历史完成成本及当前引擎状态。预测 SHALL 记录输入依据和可获得时间。录制的未来工具耗时、输出长度、准入和结束时间 MUST NOT 作为正常策略的未来提示；oracle SHALL 单列。

#### Scenario: Replay contains the actual future pause

- **WHEN** 回放驱动知道某工具批次的真实耗时
- **THEN** 驱动只用它控制模拟就绪时间，正常策略仍接收基于过去样本的预测，只有标明 oracle 的诊断可使用该未来值

### Requirement: Attribute remaining bottlenecks before core changes

实验流程 SHALL 在实现优化前保存限制收益的证据、可证伪假设、预期影响及最小失败行为。分析 SHALL 覆盖工具与就绪后等待、控制、prefill/decode、非重叠传输、整组准备和清理，并明确区分累计耗时与墙钟区间。每轮 SHALL 聚焦一个主要机制，不能将纯参数搜索或增加压力作为核心优化完成证据。

#### Scenario: Copies complete but requests recompute

- **WHEN** 完成的准备没有转化为实际缓存命中
- **THEN** 分析按内容/缓存组/操作/会话关联发布、准入、覆盖缺口及淘汰证据，无法判定的原因保留 unknown，再据此选择修改

#### Scenario: Saving does not free additional allocatable blocks

- **WHEN** 存出对象原本已经是可回收的无引用缓存
- **THEN** 系统分别记录物理引用、可回收状态及映射变化，禁止仅按存出字节宣称增加了可分配容量

### Requirement: Demonstrate the contribution of application cooperation

实验 SHALL 比较显式指令、quota-only 和完整协同。D/C SHALL 共享可获得的应用信息、预测器及冷层资源，Q/C SHALL 固定相同 quota 参数。诊断 SHALL 包含单项策略、无预测或无准入协同、简单暂停即预取基线，并定点检查预测误差、错误目标/取消及控制延迟；这些条件及 oracle SHALL 与主性能结果分列。

#### Scenario: Full cooperation only matches quota

- **WHEN** 完整协同相对 Native 的收益全部由 quota 对照解释
- **THEN** 报告明确应用信息/搬运增量尚未建立，继续诊断其成本与复用，禁止用 Native 差值代替组合贡献

### Requirement: Bounded diagnostics and actionable transfer accounting

诊断 SHALL 记录实际存出/加载范围、字节、完整可用范围、重复/无消费者操作、实际命中及准入间隔，并保留 engine epoch、operation/transfer ID 的关联。诊断 SHALL 有界且可关闭，避免调度步全量扫描。系统 MUST 区分普通 GPU 命中、已加载且随后使用的内容和无法归因的命中。

#### Scenario: Existing GPU content already satisfies a request

- **WHEN** 请求命中已有 GPU 内容但没有证据关联到本轮加载
- **THEN** 该命中不被全部计入搬运贡献，字节、操作与复用统计保留各自口径

### Requirement: Verify core gains on preserved samples and frozen versions

系统 SHALL 保留本轮核心修改前的完整协同源码及配置 C0。核心候选 SHALL 由调优样本选定，冻结后在保留样本与相同工作量上比较 C0/C。使用保留结果继续调优时 SHALL 将其转为调优数据并分配新的预登记保留样本。公共执行路径变化 SHALL 触发受影响基线复测或明确的等同性验证。

#### Scenario: A candidate improves tuning tasks only

- **WHEN** 候选在调优组提高吞吐但保留组未显示净收益
- **THEN** 优化验收保持未完成，保留退化结果及原因，不能只用调优成绩交付核心提升

### Requirement: Separate validity correctness and performance acceptance

系统 SHALL 分别报告实验有效性、契约检查和性能目标状态。性能目标 SHALL 在至少一个保留压力点达到 C 相对 N 的中位固定工作吞吐提升至少 25%、相对 Q 至少 10%、相对 C0 至少 5%，各有至少三轮普通计时且至少两轮收益同向。所有有效轮次和其他场景 SHALL 一并报告。未达标时性能验收 MUST 保持未完成。

#### Scenario: Existing cooperation already beats Native

- **WHEN** C0 已有 Native 收益，但本轮修改没有达到相对 C0 的增量目标
- **THEN** 报告可确认已有能力在新负载的表现，本轮核心优化目标仍为未完成

#### Scenario: A fast run changes outputs or omits work

- **WHEN** 提升来自减少请求/输出 token、隐藏失败、提前结束排空或改变比较条件
- **THEN** 该轮不能满足性能目标，并保留其失效原因

### Requirement: Protect low-contention cost and short-request latency

性能验收 SHALL 同时满足容量充足场景 C 相对 N 的中位耗时退化不超过 5%，以及混合组短聊天 P95 就绪至完成延迟不超过同组 N 的 1.10 倍。系统 SHALL 处理相同计划请求，单列每类完成、失败、最大等待及预登记 SLO 内吞吐，并且不新增错误复用、提前发布、资源泄漏或非任务本身错误。

#### Scenario: Aggregate throughput rises while chat tails regress

- **WHEN** 固定工作吞吐提高但短聊天尾延迟超过门槛
- **THEN** 报告等待转移与分类指标，性能验收保持未完成，继续处理调度和准备占用的干扰

### Requirement: Reviewable optimization evidence and regression delivery

交付 SHALL 保留实验配置、输入/源码哈希、原始逐请求/任务数据、命令、分析及正负结论。核心修改 SHALL 在现有相邻 suite 添加最小行为回归，并运行受影响完整测试、必要真实 GPU/模型检查、pre-commit、mypy 与 OpenSpec 严格校验。新证据 SHALL 链接到原变更，不自动替代其未执行验收；本变更不创建独立 upstream PR。

#### Scenario: A candidate has no confirmed net benefit

- **WHEN** 修改只有单轮优势、收益无法复现或产生未解决的退化
- **THEN** 不将该候选标为已验证优化；撤回无证据的最终代码选项，保留原始实验和下一项具体待解决限制
