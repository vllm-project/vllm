# 代表性 KV 工作负载规格

## ADDED Requirements

### Requirement: Bounded local workload selection

实验系统 SHALL 支持 SWE-bench Verified + mini-swe-agent、long-doc-qa、coding agent 与短聊天混合三类负载。系统 SHALL 在执行前记录任务/文档/聊天 ID、哈希、选择规则、种子、调优/保留角色及运行预算。默认采用 4 个 SWE 调优任务、8 个保留任务，扩展最多 20 个不同任务；文档最多 32 个，混合组短聊天默认 128 条。系统 MUST NOT 因本变更自动运行全量数据集。

#### Scenario: A small benchmark is prepared

- **WHEN** 执行者创建本地实验协议
- **THEN** manifest 固定三类输入、有限规模候选和调优/保留分组，并可在不启动服务的 dry run 中检查展开后的运行矩阵

#### Scenario: An input exceeds context limits

- **WHEN** 完整输入与输出预算超过当前模型上下文
- **THEN** 系统明确报告该输入及预算，统一修订协议或保留失败，禁止静默截断、删除历史或按策略设置不同上下文

### Requirement: Real coding-agent and tool evidence

系统 SHALL 使用明确版本的 mini-swe-agent，通过本分支模型执行真实任务与工具，并保留模型消息、命令、退出码、工具结果、时间、补丁和任务终态。系统 SHALL 复用可用的本地任务及容器资源，在等资源条件下执行各对照。系统 MUST NOT 用相邻项目的 token ID、路由采集性能或历史模型速度替代当前分支的测量。

#### Scenario: A tool or task fails

- **WHEN** 命令失败、达到步数/时间预算、模型输出格式错误或任务未解出
- **THEN** 该任务保留在计划和结果中，区分任务结果与基础设施错误，禁止把失败任务从分母中移除

#### Scenario: A submitted patch is evaluated

- **WHEN** 真实小样本运行产生补丁
- **THEN** 系统在冻结的评分子集上记录实际测试结果，分别报告 submitted、resolved、失败和未评分状态

### Requirement: Long-document reuse with verifiable content

long-doc-qa SHALL 使用有来源及哈希的本地自然文本，固定每文档四个问题及至少一个可核验答案。相同文档问题 SHALL 共享逐 token 一致的文档前缀。系统 SHALL 支持独立文档交错、固定顺序种子和有界并发，并记录实际复用距离。合成文本与人工等待 MUST 单独标记。

#### Scenario: Questions revisit an evicted document

- **WHEN** 多个文档交错运行后再次访问某文档
- **THEN** 报告区分 GPU 命中、冷层加载和重算，同时保留问答正确性、等待和实际完成时间

### Requirement: Mixed-workload arrival and fairness evidence

混合负载 SHALL 同时运行 coding 会话与独立短聊天。短聊天 SHALL 使用固定开放到达表，并从计划就绪时刻计入延迟，包含客户端容量等待。系统 SHALL 单列各类 offered、completed、failed、等待和尾延迟，停止到达后继续处理所有计划请求并排空。

#### Scenario: Chat requests wait behind coding agents

- **WHEN** coding 占用资源导致聊天不能立即发送或准入
- **THEN** 等待进入聊天端到端延迟和 SLO 统计，全部计划请求仍计入完成与失败结果

### Requirement: Fixed-work replay preserves the full workload

系统 SHALL 将已验证的真实轨迹转换为固定完整请求输入、输出 token 数、依赖、初始到达、工具间隔和会话/类别元数据。每次回放 SHALL 执行真实模型，后续输入保持录制值，工具不重新执行。模型、tokenizer、模板、权重版本和工作量 SHALL 有可比指纹。系统 MUST NOT 将回放输出报告为新的任务评分。

#### Scenario: Fresh generation differs from the recording

- **WHEN** 一轮回放生成的文本与录制文本不同
- **THEN** 后续请求继续使用冻结的完整输入；若输出 token 数、请求清单或依赖不同，则该轮不能进入等工作量吞吐比较

#### Scenario: A recorded request cannot reach model execution

- **WHEN** 轨迹含上下文拒绝或其他前置错误
- **THEN** 原错误独立保留且不冒充成功推理，任何可执行回放子集均在对照前单独冻结并披露排除范围

### Requirement: Baseline-calibrated pressure and cache-fit controls

系统 SHALL 固定部署 GPU KV 预算，在 Native 上按预登记的有限候选增加独立工作集或并发，选择最小有效压力点及容量充足点。压力 SHALL 同时有工作集/有效容量证据，以及缓存周转重算/加载或容量延期、抢占、持续排队证据；单一占用峰值不足以判定。选择 MUST NOT 依据协同配置的收益排名。

#### Scenario: The bounded workload still fits

- **WHEN** 达到本地规模上限后仍未观测到规定压力
- **THEN** 报告压力未建立及限制条件，压力验收保持未完成，禁止用缩小固定 KV 字节预算伪装主部署压力

### Requirement: Comparable arms and complete timing

系统 SHALL 提供 N Native APC、B 无 hint 启用对照、D 显式指令无策略、Q quota-only、C 完整协同。三个压力点 SHALL 各执行 N/D/Q/C 至少三轮，容量充足点 SHALL 执行 N/B/Q/C 至少三轮。对照 SHALL 交错顺序、独立重启、验证一致的初始缓存状态，并记录全部配置。正式输入准备、控制、重试、任务清理和设备搬运排空 SHALL 纳入普通计时。

#### Scenario: A transfer continues after the last response

- **WHEN** 模型响应结束时仍有在途操作或待清理记录
- **THEN** 计时持续到实际设备访问结束和相关终态/清理完成，超时保持失败结果

#### Scenario: A connector adds host cache capacity

- **WHEN** D/C 使用 Native 没有的冷层容量
- **THEN** 报告明确记录额外 CPU/Store 资源，并提供相同冷层资源的 D/C 对照及相同 quota 参数的 Q/C 对照

### Requirement: Local resources provenance and measurement isolation

主性能实验 SHALL 使用本地 Qwen3.8-27B BF16、TP=2、GPU 0/3、APC、PCP=DCP=1、`gpu_memory_utilization=0.5`，记录实际缓存组和 KV 容量。实验 SHALL 固定 CPU/容器/Store 资源并记录源码、依赖、输入哈希及进程归属。profiling、回归和普通计时 SHALL 安排在独立窗口，所有 Python 与依赖操作遵守 uv/venv 约定。

#### Scenario: Measurement conditions change during a run

- **WHEN** 发现源码变化、额外 GPU 计算进程、重叠 profiling/回归或工作量不匹配
- **THEN** 原始轮次及原因保留，标为无效比较并在条件恢复后补测，禁止静默只保留最快样本
