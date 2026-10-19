# 与 RFC #59708 / PR #59714 的关系及示范 PR 建议

核对日期：2026-10-06。
我们的代码基线：`d49807b022`。
外部示例 PR 核对的 head：`7966f717dd4eb09461dd97102db1352c924bbf50`。
本文保存最初的接口对齐分析。
用户随后授权在 `rfc-hint` 分支实施；已交付的具体契约见 `vllm/v1/kv_hints/README.md`，执行证据见本目录 `validation.md` 的证据索引。
COPY/MOVE 的具体接口约定仍属实验接口，未宣称已获上游最终批准。

G1 指引擎本地 GPU 缓存，G2 指较冷的后端存储层；本文示例的 G2 使用 Mooncake Store。
以下“当前实现”和“尚缺”均对应核对日期，后续完成状态见开头链接。

## 结论

建议把示范 PR 定位为：**在 #59708 的公共动作与分发接口上，展示 Mooncake G1↔G2 搬运、引擎资源审批、异步完成和真实缓存复用。**

Karen 的 #59714 是 GPU 缓存优先级的参考实现；我们的价值在于补全它尚未实现的 G2 路径和执行生命周期。

RFC #59708 明确引用 #57103，并提出将基本操作转换为 Mooncake 调用。

两份提案都涉及分发、GPU 资源管理和保留优先级；冷层执行是可以协作的部分。
当前代码不能只换一个 PR 标题就称为兼容 #59708。

核对的公开资料：

- [RFC #59708](https://github.com/vllm-project/vllm/issues/59708)：未关闭；提出 `vllm.copy_to`、`vllm.move_to`、`vllm.set_priority`，动作版本 `1.0`。
- [PR #59714](https://github.com/vllm-project/vllm/pull/59714)：未合并草稿，8 个文件、 684 行新增、2 行删除。
  当前只实现 SET_PRIORITY 的 G1 Retain；并未实现 COPY/MOVE。
- [actions.py](https://github.com/vllm-project/vllm/blob/7966f717dd4eb09461dd97102db1352c924bbf50/vllm/v1/kv_hints/actions.py)
- [dispatch.py](https://github.com/vllm-project/vllm/blob/7966f717dd4eb09461dd97102db1352c924bbf50/vllm/v1/kv_hints/dispatch.py)
- [执行语义说明](https://github.com/vllm-project/vllm/blob/7966f717dd4eb09461dd97102db1352c924bbf50/vllm/v1/kv_hints/README.md)
- [我们在 #57103 的默认机制/可选策略边界](https://github.com/vllm-project/vllm/issues/57103#issuecomment-5814964446)

RFC 和 draft PR 都是讨论中的方案，不是已合并标准。
此处检查了公开正文、评论和全部 8 个文件的 diff；没有运行对方分支的测试。
其测试通过是 PR 作者的报告。

## 各方的接口与实现

| 层次 | Karen 的 RFC / 示例 | 我们的代码与 RFC |
| --- | --- | --- |
| 应用意图 | 编排端用公共基本操作表达 Retain/Prefetch/Demote | 轻量控制器、可选的本地策略产生意图；目前还有自定义的高层动作协议名称 |
| 公共执行面 | 类型化动作、按解释方注册分发、请求生命周期回调 | 已有消费/校验路径，需要对齐共同的接口和执行时机 |
| GPU 资源管理 | 明确由 vLLM 控制分配、发布、回收 | 已有容量预留与传输引用、取消、缓存代次、所有并行进程的完成屏障，可贡献给共同执行层 |
| 后端 | COPY/MOVE 经连接器转换为后端调用；KVCR 保留自有语义 | 已有 CPU/Mooncake 原生执行、真实 Store 调用和双引擎验证 |
| 反馈与逻辑状态 | 动作结果、过期目标与 SessionPrefixIndexer 是接口和开放问题的一部分 | 已有容量受限的 OperationRegistry、引擎代次、逻辑索引、关联事件和查询；物理驻留状态报告仍不完整 |
| 可选决策 | 默认编排端驱动；引擎生成提示消息属于开放问题 | 本地执行建议、建议合并、配额与请求准入示例已有代码；原生策略到真实 Mooncake 的完整验收仍待补 |

**两份提案都要求引擎审批 GPU 资源。**

RFC #59708 同样要求引擎管理 GPU 资源。

应共享这套执行机制，使用一套分配器和基本操作执行器。

## 必须对齐的实际差异

| 问题 | #59708 / #59714 | 我们当前实现 | 示范 PR 的处理建议 |
| --- | --- | --- | --- |
| 协议格式 | `vllm.*` 基本操作，版本 `1.0` | `retain/release/demote/prefetch/share/...`，版本 `1` | 主示例使用公共基本操作名称；旧高层动作作为明确的实验/兼容接口，不能无条件同名替换 |
| 控制入口 | 初期随真实推理请求进入；独立入口仍开放 | 请求附带与独立控制 Python/HTTP 共用控制器 | 主示例先对齐请求附带接口约定；独立入口作为已验证扩展展示，共用执行器 |
| 分发粒度 | 每个 `(action_type, action_version)` 一个 vLLM 原生解释方；跳过非 `vllm.*` 动作，保留原提示消息 | 调用上下文为整条消息选择原生或后端解释方，vLLM 原生消费后清空 request.kv_hints | 先统一唯一的入口与解释归属规则；主示例可使用各自独立提示消息，混合解释方消息另行明确 |
| Retain 时机 | 入站验证，成功结束后、释放请求引用前生效；abort/error 丢弃待执行动作 | 入口立即解释并安装声明，TTL 随即开始 | 生命周期阶段必须进入处理函数契约；不能把两种 TTL 起点说成一致 |
| Priority 参数 | 有符号整数；设置必须正 TTL；清除为 value=null；claim_id + revision | priority 0..100、TTL 可省略；owner_id + declaration_id；整条消息去重 | 对齐值域、声明标识、revision、清除与重试语义；不能只写一层字段别名 |
| 声明对象 | request_cached 对应当前物理缓存副本，reuse/reset 时清理 | namespace/hash/group 上的逻辑声明，可在副本重建后继续生效，并可 follow_growth | 保留两者区别；兼容公共物理副本生命周期语义，逻辑/增长声明应显式作为扩展讨论 |
| Target | 当前 PR 仅 `request_cached` / `local_g1`；远端与哈希目标未实现 | request/session/continuation/range/hash-group + 缓存命名空间 | COPY/MOVE 需要补共同目标与位置类型；外部不要依赖可复用的物理缓存块 ID |
| Result | 执行器内部的 KvHintResult；没有新的远端结果通道 | 持久化到有容量限制的结果表，支持查询与事件输出 | 复用关联标识与引擎代次；增加异步任务完成与部分目标结果的连接，勿并行发明两套结果权威 |
| `deferred` 含义 | 此 PR 表示等到请求完成再调用处理函数；尚未审批资源 | 我们的 DEFERRED 是终态，表示本次不具备执行条件 | 分开等待生命周期回调、资源不足延期和正在执行，避免调用方错误停止等待或重放 |
| Scheduler | GPU 缓存示例不增加请求排序与准入 | 可选配额与评分会影响等待请求选择 | 主示例默认关闭，单独说明扩展及其证据，不作为基本操作执行器的前提 |

两个直接合并风险：

1. `vllm/v1/engine/core.py` 的 `add_request` 会消费并清空 vLLM 原生提示消息，然后才调用调度器。
   PR #59714 的分发器挂在 `Scheduler.add_request`，因此直接 cherry-pick 后它可能收不到 vLLM 原生 hint。
   反过来后端整包路由若仍经过其 `vllm.*` 自动识别，又会破坏我们当前的整包解释归属约定。
2. 我们的 `RetentionFreeBlockQueue` 与对方的 `KVCachePriority.take_blocks` 都控制淘汰顺序。
   应共用一个实现，再接逻辑声明或公共优先级动作；不要叠加两个独立维护的淘汰堆。

对方保留原提示消息不等于已经完成所有混合解释方消息的处理约定。
需明确哪些动作到达后端、后端如何忽略已由 vLLM 原生消费的动作、以及两边的结果关联。
是否会重复执行取决于转发和结果关联规则，仍需确认。

还有一个 G2 特有的接口限制：#59714 的 `when` 在注册解释方时固定，而同一 `vllm.copy_to` 的 G1→Store 与 Store→G1 可能需要不同的生命周期阶段。
它不能直接为同一类型与版本注册两份解释方。
主例采用向冷层 MOVE、向 GPU COPY 可以先避免这个歧义；通用 COPY 的方向、目标与执行时机需由共同执行计划与处理函数契约明确，不能靠更换动作名称绕开。

## 高层缓存动作如何表达为基本操作

| 我们的意图 | 建议表达 | 不能丢失的边界 |
| --- | --- | --- |
| Retain | SET_PRIORITY(location=G1, value=p, claim, revision, ttl) | 仅淘汰优先级；不保证驻留 |
| Release | SET_PRIORITY(value=null, 同一声明, 更新 revision) | 只清除此声明；不是删除数据或覆盖其他声明 |
| Prefetch | COPY_TO(Store G2 → 当前引擎 G1) | 加载完成并注册后才报告 GPU 可复用 |
| Demote | MOVE_TO(当前引擎 G1 → Store G2) | Put 成功与 G1 回收是两个阶段；活跃引用会阻止回收 |
| Share 到 B | COPY_TO(A G1 → Store G2)，再在 B 执行 COPY_TO(Store G2 → B G1) | 现有 Share 只建立 Store 可访问性，不能冒充 B 已可复用或原生 GPU P2P |
| Evict | 三个基本操作中没有现成的等价表达 | 清除优先级不等于立即淘汰；不要把它隐藏成 MOVE_TO(null) |
| Associate / FinishTask | 独立逻辑状态管理 | 不应成为 COPY/MOVE/PRIORITY 的隐式副作用 |

上表是语义映射，不是已经实现的协议格式。
COPY/MOVE 的类型化动作参数和远端位置还需要与 Karen 对齐；不能把 RFC 中简化的 `(Worker, Tier)` 示例或数字 `target_blocks` 当作稳定的最终协议。

附带讨论中的 “guarantee retention and movement decisions under pressure” 建议改为保证资源/引用安全以及可观察的完成、部分完成和拒绝。
压力下 Retain 只是相对淘汰优先级，不能承诺 KV 一定保留；两份 RFC 的当前方向都不提供公共 Pin 保证。

## 建议的一份示范 PR

建议标题：

> [RFC][KV Hints] Execute COPY_TO/MOVE_TO through Mooncake Store

定位：补全 #59708 的冷层执行器与参考调用流程，并与 #59714 共用动作与分发接口。
继续保留一个完整 PR 的既定目标，可用清晰的提交和 README 分层展示。
当前研发分支相对 `0ee724bb7a` 有 90 个文件、约 1.32 万行新增（含测试和文档）；直接整包描述成一个简单冷层示例，会显著增加审阅负担。

整理方法：保留完整研发分支，在展示分支按下面顺序组织同一个 PR。
是否减少最终交付范围或拆分成其他 PR，仍需单独决定，不能在本次分析里把首版任务变成未来工作。
若示范分支基于尚未合并的 #59714，要固定依赖提交 SHA，并说明相对依赖提交的差异；不能把依赖代码算作我们的原创贡献。
也可以先向共同接口贡献所需扩展，再在其上展示 Mooncake；关键是共同维护一份动作与分发接口。

### A. 默认流程：随请求进入，真实 Mooncake 执行

1. 轻量编排端发真实请求 R1，附带 MOVE_TO(G1→Store G2)。
   按支持的缓存组边界选择目标范围，在成功完成回调解析实际可用源块。
2. 在释放请求引用 **之前**取得独立传输引用、登记传输任务，随后让请求正常结束；复用连接器元数据、Store Put 和并行进程完成通知。
3. Put 完成后，报告冷层副本可用；仅在引用允许时回收源副本。
   MOVE 的结果要明确源已回收与源仍被引用，复制成功不能当作搬运全部完成。
4. R2 是实际恢复请求，附带 COPY_TO(Store G2→本机 G1)。
   在进入模型执行前，执行器验证目标与容量，预留目标 GPU 缓存块，通过 Store Get 加载；R2 等待兼容的在途加载。
5. 所有必要并行进程和缓存组完成后注册 GPU 内容，R2 再命中前缀，校验输出一致。

**入口随请求不意味着传输必须同步完成。**
R1 结束后，已接受的后台传输仍须存活和推进，不能用伪推理请求来驱动。
现有传输引用、引擎存活检查和完成通知负责推进这部分后台工作。

**R2 才携带的 COPY 没有提前量。**
这个默认例子验证实际执行与缓存可复用状态，尚未测量工具等待期间提前预取的收益。
真正的提前准备需要附带提示消息的更早请求，或下面的独立入口扩展；不能混用两种实验的延迟结论。

### B. 同一执行器的扩展示例

- 用现有独立 Python/HTTP 控制，在工具等待期间提交同一种类型化基本操作，等待实际完成后才发送 R2。
  明确这是 #59708 尚开放的控制入口约定。
- 展示 A→Store→B 两步 COPY，分别观察源副本保留、Store 可访问性以及 B 的准备完成与实际缓存命中。
  保持由 B 审批 B 的 GPU 资源。
- 单独启用本地策略：它产生同一种基本操作执行建议，仍经过共同审批与后端处理函数。
  我们现有 ToolWaitPolicy 消费显式信号，并不是已经实现了从生成 token 自动检测工具调用；自动触发点和真实 G2 策略运行仍需补齐。

Dynamo/llm-d 接入仍按已确认范围保留为后续，发送端明确写轻量控制器。
不能把它标为已完成的 Dynamo 端到端集成。

### C. 如何复用现有代码

| 现有模块 | 在共同接口下的职责 |
| --- | --- |
| `kv_hints/native.py` | 与 `actions.py` 对齐公共基本操作参数格式；高层动作的组合方法可放在应用或示例层 |
| `kv_hints/controller.py` | 复用目标解析与资源审批，避免新增重复分发器 |
| `kv_hints/control.py` | 复用有容量限制的结果关联与去重、引擎代次、查询与事件；适配传输任务与生命周期阶段 |
| `kv_hints/resources.py`、`core/kv_cache_transfer.py` | 复用预留、引用、取消与缓存代次；由引擎权威审批 |
| `kv_hints/transfers.py` | 复用复制、写入、回收的分阶段完成及共享调用方记录；补基本操作的结果与目标范围 |
| `mooncake/store/control.py`、scheduler/worker | 复用真实 Put/Get、布局、存储键和元数据以及所有并行进程的完成通知，作为 Mooncake 处理函数的数据路径 |
| `state.py`、`request_observer.py` | 复用安全目标解析；会话寻址与索引是明确扩展，不宣称对方 PR 已定义完整格式 |
| `retention.py`、`core/kv_cache_retention.py` | 与共同的 GPU 优先级管理器统一；保留逻辑声明生命周期与物理副本生命周期的区别 |
| `policy.py`、`policy_runtime.py` | 可选动作生成器与建议合并器，默认关闭；输出公共基本操作，不直接调用 Mooncake 或操纵队列 |

`mooncake/store/actions.py` 可以作为薄的处理函数入口，但不必为了文件名复制已有 `control.py` 的数据路径。
注册 vLLM 原生执行器属于内部接线，不必扩张为一个让任意用户 hint 类型拥有语义解释权的公开插件 API。

### D. 示范必须证明的行为

- 请求成功、取消、达到长度上限、抢占、生成尾部范围、TTL 起点和重复 hint。
- COPY 处于 accepted/queued 时目标不可读；所有并行进程及缓存组完成后才发布。
- MOVE 保存失败时不删源；源仍被引用时报告部分完成；取消不提前释放工作进程的缓冲区。
- 恢复请求与 COPY 共享同一次加载；容量不足不会重复计费；普通重算回退仍可继续。
- 两个共享前缀调用方各自拥有声明和结果；撤销一方不破坏另一方。
- 原生动作只消费一次；后端动作参数的版本、值和顺序保持不变。
- 使用改成基本操作的真实控制器重新跑 Mooncake 与模型验证。
  当前历史记录验证的是旧高层缓存动作，不能换名字后直接宣称新接口已经端到端通过。

现有 `OperationResult` 多数目标范围字段是计数；公共 COPY/MOVE 的部分完成结果还需明确可归因的目标子集/失败位置，不能只给一个“完成了 N 块”就声称满足所有部分目标结果语义。
已完成的单次操作也不能当作之后持续驻留的保证。

## 已有证据与缺口

已有记录：真实 Mooncake GPU 缓冲区测试 42 项；Qwen TP=2 的显式 Demote/Prefetch 回收并恢复 4 个副本、命中 784 tokens；双引擎 A/Store/B 也观察到 B 的 784-token 实际复用；Mooncake 的两组 128 题均为 124/128。
详见 `validation.md` 所列的早期阶段记录及压缩原始数据。
这些记录提供了实际执行器的测试证据。

尚缺：`vllm.copy_to/move_to` 原语接线、新的请求生命周期约定与共同优先级契约、原生策略到真实 G2 的验收、完整 rollout/verl/公平性实验以及物理驻留状态报告。
首版 40/52 的缺口仍然保留；示范 PR 应如实标为参考实现或草稿。

## 建议与 Karen 对齐的五个决定

1. 将 COPY/MOVE 的类型化动作、目标、位置、处理函数与传输任务接口放在同一个公共接口；以谁的分支作为演示基底，如何避免重复分发与优先级状态。
2. COPY 的准入与加载就绪回调，以及 MOVE 的完成回调、传输引用、源副本部分回收契约。
3. 逐动作解释方与整条消息路由的关系，以及后端转发已消费动作的规则。
   主例是否先要求单一解释方的提示消息。
4. 对齐“等待生命周期回调”和“资源不足需重试”两种 deferred，并确定 G2 完成结果如何关联原始消息与动作、向编排端回报。
5. 独立控制和引擎生成提示消息是否作为同一示范 PR 的可选扩展，与默认请求附带基线分开标识；不把默认 G2 示例绑定到配额与评分实验。

可直接用于讨论的英文表述：

> We can make our reference PR the G2 companion to #59714, using the shared `vllm.copy_to` / `vllm.move_to` contracts and dispatcher.
> The default example would remain orchestrator-driven and request-attached.
> Our existing Mooncake code can supply engine-owned reservations and transfer holds, Store Put/Get, all-rank completion and publication before actual prefix reuse.
> We would like to converge on request lifecycle hooks, target/location schemas and async completion reporting, especially the distinction between waiting for a lifecycle hook and resource deferral.
> Request-free submission and local-policy producers can be clearly labeled opt-in extensions over the same execution machinery.

## 相关工作与发布前检查

公开 REST 检索找到 #59714（GPU 缓存优先级）、#58198（无推理请求的 CPU 预取）以及相关的 [#38514](https://github.com/vllm-project/vllm/pull/38514)（按上下文保留）。
另一个 [#55220](https://github.com/vllm-project/vllm/pull/55220) 是 Mooncake 对象语义提交清单，不是同一个提示动作执行器示例，但需关注其跨布局与存储对象的完整性约束。
搜索 `vllm.copy_to` 未找到其他未合并 PR，不等于证明没有重复工作。

本次按仓库要求尝试 `gh issue view 57103 --comments`、按 issue body 与 KV hint 关键词查询未合并 PR；CLI 仍未认证，命令未成功。
本次正文、评论和 diff 通过公开 GitHub REST API 获取。
实际发布前仍需完成最新查重、与重叠工作确认复用边界、人类逐行审阅/测试和 AI 辅助声明。
本次没有发布 PR 或发送讨论消息。
