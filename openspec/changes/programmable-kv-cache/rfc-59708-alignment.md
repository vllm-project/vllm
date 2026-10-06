# 与 RFC #59708 / PR #59714 的关系及示范 PR 建议

核对日期：2026-10-06。我们的代码基线：`d49807b022`。
外部示例 PR 核对的 head：`7966f717dd4eb09461dd97102db1352c924bbf50`。
本文保存最初的接口收敛分析。用户随后授权在 `rfc-hint` 分支实施；已交付的具体契约
见 `vllm/v1/kv_hints/README.md`，执行证据见本目录 `validation.md` 的 RFC 对齐小节。
COPY/MOVE 的 concrete profile 仍属实验接口，未宣称已获上游最终批准。

## 结论

建议把示范 PR 定位为：**在 #59708 的公共 action/dispatch 接口上，展示
Mooncake G1↔G2 搬运、engine 资源审批、异步完成和真实缓存复用。**

Karen 的 #59714 是 G1 priority 的参考实现；我们的价值在于补全它尚未实现的
G2 路径和执行生命周期。#59708 明确引用 #57103，并提出 Mooncake lowering。
因此不是替代关系，但也不是完全不重叠：两边都涉及 dispatch、G1 资源权和
retention。当前代码不能只换一个 PR 标题就称为兼容 #59708。

公开源：

- [RFC #59708](https://github.com/vllm-project/vllm/issues/59708)：open；提出
  `vllm.copy_to`、`vllm.move_to`、`vllm.set_priority`，action version `1.0`。
- [PR #59714](https://github.com/vllm-project/vllm/pull/59714)：open draft，8 个文件、
  684 行新增、2 行删除。当前只实现 SET_PRIORITY 的 G1 Retain；并未实现 COPY/MOVE。
- [actions.py](https://github.com/vllm-project/vllm/blob/7966f717dd4eb09461dd97102db1352c924bbf50/vllm/v1/kv_hints/actions.py)
- [dispatch.py](https://github.com/vllm-project/vllm/blob/7966f717dd4eb09461dd97102db1352c924bbf50/vllm/v1/kv_hints/dispatch.py)
- [执行语义说明](https://github.com/vllm-project/vllm/blob/7966f717dd4eb09461dd97102db1352c924bbf50/vllm/v1/kv_hints/README.md)
- [我们在 #57103 的默认机制/可选策略边界](https://github.com/vllm-project/vllm/issues/57103#issuecomment-5814964446)

RFC 和 draft PR 都是讨论中的方案，不是已合并标准。此处检查了公开正文、评论和
全部 8 个文件的 diff；没有运行对方分支的测试。其测试通过是 PR 作者的报告。

## 三者分别贡献什么

| 层次 | Karen 的 RFC / 示例 | 我们的代码与 RFC |
| --- | --- | --- |
| 应用意图 | orchestrator 用公共 primitives 表达 Retain/Prefetch/Demote | 轻量 controller、可选的本地策略产生意图；目前还有自己的 semantic action wire names |
| 公共执行面 | typed actions、按 consumer 注册分发、request lifecycle hooks | 已有消费/校验路径，需要对齐共同的接口和执行时机 |
| G1 资源权 | 明确由 vLLM 控制分配、发布、回收 | 已有 reservation/holds、取消、generation、全 rank barrier，可贡献给共同执行层 |
| 后端 | COPY/MOVE 经 connector lowering；KVCR 保留自有语义 | 已有 CPU/Mooncake 原生执行、真实 Store 调用和双 engine 验证 |
| 反馈与逻辑状态 | action outcome / stale targets / SessionPrefixIndexer 是接口和开放问题的一部分 | 已有 bounded OperationRegistry、epoch、逻辑索引、关联事件和查询；物理 residency 报告仍不完整 |
| 可选决策 | 默认 orchestrator 驱动；engine-generated hints 属于开放问题 | 本地 proposal/composer、quota/admission 示例已有代码；原生策略到真实 Mooncake 的完整验收仍待补 |

**统一资源审批不是我们独有、对方缺失的一层。** #59708 同样要求 engine 保有
G1 权限。应共享这套执行机制，而不是在 primitive executor 前后放两套 allocator。

## 必须对齐的实际差异

| 问题 | #59708 / #59714 | 我们当前实现 | 示范 PR 的处理建议 |
| --- | --- | --- | --- |
| Wire contract | `vllm.*` primitives，version `1.0` | `retain/release/demote/prefetch/share/...`，version `1` | 主示例使用公共 primitive 名称；旧 semantic action 作为明确的实验/兼容接口，不能无条件同名替换 |
| 控制入口 | 初期随真实 inference request 进入；独立入口仍开放 | request-attached 与 request-free Python/HTTP 共用 controller | 主示例先对齐 request-attached profile；独立入口作为已验证扩展展示，共用执行器 |
| Dispatch 粒度 | 每个 `(action_type, action_version)` 一个 native consumer；跳过非 `vllm.*` action，保留原 envelope | invocation 为整包选择 native/backend，native 消费后清空 request.kv_hints | 先统一唯一的入口与 ownership 规则；主示例可使用各自独立 envelope，mixed-envelope 另行明确 |
| Retain 时机 | 入站验证，成功结束后、释放请求引用前生效；abort/error 丢弃待执行动作 | 入口立即解释并安装声明，TTL 随即开始 | lifecycle phase 必须进入 handler 契约；不能把两种 TTL 起点说成一致 |
| Priority 参数 | 有符号整数；设置必须正 TTL；清除为 value=null；claim_id + revision | priority 0..100、TTL 可省略；owner_id + declaration_id；整包 dedup | 对齐值域、claim 身份、revision、清除与重试语义；不能只写一层字段别名 |
| 声明对象 | request_cached 对应当前物理缓存副本，reuse/reset 时清理 | namespace/hash/group 上的逻辑声明，可在副本重建后继续生效，并可 follow_growth | 保留两者区别；兼容公共 copy-scoped 语义，逻辑/增长声明应显式作为扩展讨论 |
| Target | 当前 PR 仅 `request_cached` / `local_g1`；远端与 hash target 未实现 | request/session/continuation/range/hash-group + namespace | COPY/MOVE 需要补共同 target/location 类型；外部不要依赖可复用的物理 block ID |
| Result | executor-local KvHintResult；没有新的远端结果通道 | 持久化到有界 registry，并可 query/events | 复用 correlation/epoch；增加异步 job 完成与 partial scope 的连接，勿并行发明两套结果权威 |
| `deferred` 含义 | 此 PR 表示等到请求完成再调用 handler；尚未审批资源 | 我们的 DEFERRED 是终态，表示本次不具备执行条件 | 分开 lifecycle queued、resource deferred 和 running，避免调用方错误停止等待或重放 |
| Scheduler | G1 示例不增加 request ranking/admission | 可选 quota/score 会影响等待请求选择 | 主示例默认关闭，单独说明扩展及其证据，不作为 primitive executor 的前提 |

两个直接合并风险：

1. `vllm/v1/engine/core.py` 的 `add_request` 会消费并清空 native envelope，然后才
   调用 scheduler。#59714 的 dispatcher 挂在 `Scheduler.add_request`，因此直接
   cherry-pick 后它可能收不到 native hint。反过来 backend 整包路由若仍经过其
   `vllm.*` 自动识别，又会破坏我们当前的整包 ownership 约定。
2. 我们的 `RetentionFreeBlockQueue` 与对方的 `KVCachePriority.take_blocks`
   都控制 eviction ordering。应收敛成一个权威实现，再接逻辑声明或公共 priority
   action；不要叠加两个 independently-maintained eviction heaps。

对方保留原 envelope 不等于已经完成所有 mixed-owner 保证。需明确哪些 action
到达 backend、backend 如何忽略已由 native 消费的 action、以及两边的结果关联。
这是待收敛的契约，不据此直接断言当前 KVCR 一定会重复执行。

还有一个 G2 特有的接口限制：#59714 的 `when` 在注册 consumer 时固定，而同一
`vllm.copy_to` 的 G1→Store 与 Store→G1 可能需要不同的 lifecycle phase。它不能
直接为同一 type/version 注册两份 consumer。主例采用 MOVE-out / COPY-in 可以
先避免这个歧义；通用 COPY 的方向、目标与执行时机需由共同 planner/handler
契约明确，不能靠更换 action 名称绕开。

## Semantic actions 如何表达为 primitives

| 我们的意图 | 建议表达 | 不能丢失的边界 |
| --- | --- | --- |
| Retain | SET_PRIORITY(location=G1, value=p, claim, revision, ttl) | 仅 eviction bias；不保证驻留 |
| Release | SET_PRIORITY(value=null, 同一 claim, 更新 revision) | 只清除此声明；不是删除数据或覆盖其他声明 |
| Prefetch | COPY_TO(Store G2 → 当前 engine G1) | 注册为可复用内容后才 ready |
| Demote | MOVE_TO(当前 engine G1 → Store G2) | Put 成功与 G1 回收是两个阶段；活跃引用会阻止回收 |
| Share 到 B | COPY_TO(A G1 → Store G2)，再在 B 执行 COPY_TO(Store G2 → B G1) | 现有 Share 只建立 Store 可访问性，不能冒充 B 已 ready 或原生 GPU P2P |
| Evict | 三个 primitives 中没有现成的等价表达 | 清 priority 不等于立即 evict；不要把它隐藏成 MOVE_TO(null) |
| Associate / FinishTask | 独立逻辑状态管理 | 不应成为 COPY/MOVE/PRIORITY 的隐式副作用 |

上表是语义映射，不是已经实现的 wire schema。COPY/MOVE 的 typed payload 和
remote location 还需要与 Karen 收敛；不能把 RFC 中简化的 `(Worker, Tier)` 示例
或数字 `target_blocks` 当作稳定的最终协议。

附带讨论中的 “guarantee retention and movement decisions under pressure” 建议
改为保证资源/引用安全以及可观察的完成、部分完成和拒绝。压力下 Retain 只是
priority bias，不能承诺 KV 一定保留；两份 RFC 的当前方向都不提供 public Pin。

## 建议的一份示范 PR

建议标题：

> [RFC][KV Hints] Execute COPY_TO/MOVE_TO through Mooncake Store

定位：补全 #59708 的 G2 executor/reference path，并与 #59714 共用 action/dispatch
接口。继续保留一个完整 PR 的既定目标，可用清晰的 commits 和 README 分层展示。
当前研发分支相对 `0ee724bb7a` 有 90 个文件、约 1.32 万行新增（含测试和文档）；
直接整包描述成一个简单 G2 example，会显著增加审阅负担。

整理方法：保留完整研发分支，在展示分支按下面顺序组织同一个 PR。是否减少最终
交付范围或拆分成其他 PR，仍需单独决定，不能在本次分析里把首版任务变成未来工作。
若示范分支基于尚未合并的 #59714，要固定 dependency SHA，并说明 stacked diff；
不能把依赖代码算作我们的原创贡献。也可以先向共同接口贡献所需扩展，再在其上展示
Mooncake；关键是共同维护一份 action/dispatch surface。

### A. 默认流程：随请求进入，真实 Mooncake 执行

1. 轻量 orchestrator 发真实请求 R1，附带 MOVE_TO(G1→Store G2)。按支持的 group
   边界选择 scope，在成功完成 hook 解析实际可用源块。
2. 在释放 request references **之前**取得独立 transfer holds、登记 job，随后让
   请求正常结束；复用 connector metadata、Store Put 和 rank completion。
3. Put 完成后，报告 cold availability；仅在引用允许时回收源副本。MOVE 的结果
   要明确 source-reclaimed 与 source-still-referenced，不能把 copy success 当作
   完整 move success。
4. R2 是实际恢复请求，附带 COPY_TO(Store G2→本机 G1)。在进入模型执行前，执行器
   验证目标与容量，预约目的地，通过 Store Get 加载；R2 等待兼容的在途 load。
5. 所有必要 rank/group 完成后注册 GPU 内容，R2 再命中前缀，校验输出一致。

**入口随请求不意味着传输必须同步完成。** R1 结束后，已接受的后台 transfer
仍须存活和推进，不能用伪 inference request 来驱动。这恰是我们现有 lease、
connector liveness 和 completion 代码的价值。

**R2 才携带的 COPY 没有提前量。** 这个默认例子证明 execution/readiness，不证明
工具等待期间的 proactive prefetch 收益。真正的提前准备需要另一个更早的真实请求
piggyback，或下面的独立入口扩展；不能混用两种实验的延迟结论。

### B. 同一执行器的扩展示例

- 用现有独立 Python/HTTP 控制，在 tool wait 期间提交同一种 typed primitive，
  等待实际完成后才发送 R2。明确这是 #59708 尚开放的 ingress profile。
- 展示 A→Store→B 两步 COPY，分别观察 source preservation、Store availability 和
  B 的 readiness/cache hit。保持由 B 审批 B 的 GPU 资源。
- 单独 opt-in 本地策略：它产生同一种 primitive proposal，仍经过共同审批与
  backend handler。我们现有 ToolWaitPolicy 消费显式 signal，并不是已经实现了
  从生成 token 自动检测 tool call；自动触发点和真实 G2 策略运行仍需补齐。

Dynamo/llm-d 接入仍按已确认范围保留为后续，发送端明确写 lightweight controller。
不能把它标为已完成的 Dynamo E2E。

### C. 如何复用现有代码

| 现有模块 | 在共同接口下的职责 |
| --- | --- |
| `kv_hints/native.py` | 与 `actions.py` 收敛公共 primitive schema；semantic recipes 可放 application/example 层 |
| `kv_hints/controller.py` | 抽出 common resolution/approval 接线，避免再做一套竞争 dispatcher |
| `kv_hints/control.py` | 复用 bounded correlation、dedup、epoch、query/events；适配 job 与生命周期阶段 |
| `kv_hints/resources.py`、`core/kv_cache_transfer.py` | 复用 reservation、references、cancel、generation；由 engine 权威审批 |
| `kv_hints/transfers.py` | 复用 copy/save/reclaim 分阶段完成和共享 interests；补 primitive outcome/scope |
| `mooncake/store/control.py`、scheduler/worker | 复用真实 Put/Get、layout/key、metadata 与全 rank 回报，作为 Mooncake handler 的数据路径 |
| `state.py`、`request_observer.py` | 复用安全 target resolution；session addressing/index 是明确扩展，不宣称对方 PR 已定义完整格式 |
| `retention.py`、`core/kv_cache_retention.py` | 与共同 G1 priority owner 统一；保留逻辑 lifetime 与 copy lifetime 的区别 |
| `policy.py`、`policy_runtime.py` | 可选 producer/composer，默认关闭；输出公共 primitive，不直接调用 Mooncake 或操纵队列 |

`mooncake/store/actions.py` 可以作为薄 handler 入口，但不必为了文件名复制已有
`control.py` 的数据路径。注册 native executors 是内部 wiring，不必扩张为一个
让任意用户 hint 类型拥有语义解释权的公开插件 API。

### D. 示范必须证明的行为

- 请求成功、abort、length limit、preemption、生成尾部范围、TTL 起点和重复 hint。
- COPY accepted/queued 不产生可读 destination；全 rank/group 完成后才发布。
- MOVE 保存失败时不删源；源仍被引用时报告部分完成；取消不提前释放 worker buffer。
- Resume 与 copy 共享同一 load；容量不足不会重复计费；正常 recompute/fallback 可继续。
- 两个共享前缀调用方各自拥有声明和结果；撤销一方不破坏另一方。
- Native 只消费一次；backend payload 的版本/值/顺序保持 opaque。
- 使用改成 primitive 的真实 controller 重新跑 Mooncake 与模型验证。当前历史记录
  验证的是旧 semantic actions，不能换名字后直接宣称新接口已经端到端通过。

现有 `OperationResult` 多数 scope 字段是计数；公共 COPY/MOVE 的 partial result
还需明确可归因的目标子集/失败位置，不能只给一个“完成了 N 块”就声称满足所有
partial-scope 语义。已完成的单次操作也不能当作之后持续驻留的保证。

## 已有证据与缺口

已有记录：真实 Mooncake GPU buffer suite 42 项；Qwen TP=2 的显式 Demote/Prefetch
回收并恢复 4 个副本、命中 784 tokens；双 engine A/Store/B 也观察到 B 的 784-token
实际复用；Mooncake 的两组 128 题均为 124/128。详见 `validation.md` 的 October 1
记录及压缩原始数据。这是很强的 executor 基础，而不是还只有接口草图。

尚缺：`vllm.copy_to/move_to` 原语接线、新 request-lifecycle profile、共同 priority
契约、原生策略到真实 G2 的验收、完整 rollout/verl/公平性实验以及物理 residency
报告。首版 40/52 的缺口仍然保留；示范 PR 应如实标为 reference/draft。

## 建议与 Karen 收敛的五个决定

1. 将 COPY/MOVE 的 typed action、target/location 和 handler/job 接口放在同一个
   公共 surface；以谁的分支作为演示基底，如何避免重复 dispatch/priority 状态。
2. COPY 的 admission/load-ready hook，以及 MOVE 的 completion hook、transfer holds、
   partial source reclamation 契约。
3. per-action consumer 与 whole-envelope route 的关系，以及 backend 转发已消费
   action 的规则。主例是否先要求单一 owner 的 envelope。
4. 对齐“等待 lifecycle hook”和“资源不足需重试”两种 deferred，并确定 G2 完成结果
   如何关联原始 message/action、向 orchestrator 回报。
5. request-free 和 engine-generated hints 是否作为同一示范 PR 的 opt-in extensions，
   与默认 request-attached 基线分开标识；不把默认 G2 示例绑定到 quota/score 实验。

可直接用于讨论的英文表述：

> We can make our reference PR the G2 companion to #59714, using the shared
> `vllm.copy_to` / `vllm.move_to` contracts and dispatcher. The default example
> would remain orchestrator-driven and request-attached. Our existing Mooncake
> code can supply engine-owned reservations and transfer holds, Store Put/Get,
> all-rank completion and publication before actual prefix reuse. We would like
> to converge on request lifecycle hooks, target/location schemas and async
> completion reporting, especially the distinction between waiting for a lifecycle
> hook and resource deferral. Request-free submission and local-policy producers
> can be clearly labeled opt-in extensions over the same execution machinery.

## 相关工作与发布前检查

公开 REST 检索找到 #59714（G1 priority）、#58198（CPU request-free prefetch）以及
相关的 [#38514](https://github.com/vllm-project/vllm/pull/38514)（context-aware retention）。
另一个 [#55220](https://github.com/vllm-project/vllm/pull/55220) 是 Mooncake semantic
commit manifests，不是同一个 hint executor 示例，但需关注其跨 layout/object 完整性边界。
搜索 `vllm.copy_to` 未找到其他 open PR，不等于证明没有重复工作。

本次按仓库要求尝试 `gh issue view 57103 --comments`、按 issue body 与 KV hint
关键词查询 open PR；CLI 仍未认证，命令未成功。本次正文、评论和 diff 通过公开
GitHub REST API 获取。实际发布前仍需完成最新查重、与重叠工作确认复用边界、
人类逐行审阅/测试和 AI assistance 披露。本次没有发布 PR 或发送讨论消息。
