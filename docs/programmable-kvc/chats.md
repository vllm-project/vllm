# #feat-session-aware 频道聊天记录

> 频道由 @Maroon Ayoub 于 8月15日创建

---

## 8月15日

### Maroon Ayoub（早上 7:05）：频道目的

hey all - we should port the multiple small-forum conversations into this channel as pretty much everyone wants to set context, this effort focuses on evolving the inference stack for agentic-inference - particularly focusing on session-centric orchestration which fits the latter and unlocks optimizations - we can drop some RFCs and reading resources in a thread here.

> **Francisco Javier Arceo**（早上 7:22）回复： Just noting that we should coordinate with #sig-agentic-api as well

---

### Maroon Ayoub（早上 7:05）：工作状态

the current status is that @Karen Chung got the plumbing in, and we're tackling two fronts next:

1. KVEvents
2. Offloading/Retention APIs

---

### Karen Chung（早上 7:08）

Thanks for starting the channel @vMaroon!!
Let's also use this forum to align on the boundary of responsibilities between vLLM engine and orchestrator (Dynamo, llmd) regarding KV block <-> session information.
Mostly questions for @Zhewen Li and @Yifan Qiao whether vLLM is open to accepting new indexing structures, etc.

> **Karen Chung**（早上 7:11）回复： I'll return with more concrete questions later, but here's the most high-level vLLM RFC from our end related to this in the meantime: <https://github.com/vllm-project/vllm/issues/51428>. @vMaroon/me/@ishan and others had a conversation earlier this week re: the two design choices in the `Phase 1: Session-addressable KV cache` section of this RFC.
> The proposed designs depend on either vLLM or Dynamo owning a `SessionPrefixIndex`.
>
> **Yifan Qiao**（早上 6:43）回复： Thanks for the RFC and the nice writeup!
> I know I said I preferred to have KVEvent as the interface between vLLM engine<> orchestrator, but after reading the RFC, the proposed `SessionPrefixIndex` inside vLLM also looks quite appealing to me, so I am now supportive to have it inside vLLM.
> With that being said, I feel orchestrator can still mainly a similar data structure separately, which is a global view vs. per-engine view of sessions with vLLM-internal `SessionPrefixIndex`.
> And the two `SessionPrefixIndex`s (inside engine and orchestrator) can still talk with KVEvent.
> Does this make any sense to you?
> cc @Nick Hill @Trong Dao Le @Woosuk Kwon too FWIW as we also discussed sth similar before.
>
> **Maroon Ayoub**（下午 1:29）回复： For offloading we *need* such a structure to enable session-addressable KV 搬运 APIs, if the core can have it then we can *maybe* make use of it for a more vertical view or boundary APIs (cc @Danny Harnik @Or Ozeri @Chang Guo)
>
> **ishan**（晚上 11:02）回复： Can you explain a bit more on
> > *maybe* make use of it for a more vertical view or boundary APIs
>
> **ishan**（晚上 11:21）回复：
> > Thanks for the RFC and the nice writeup!
> > I know I said I preferred to have KVEvent as the interface between vLLM engine<> orchestrator, but after reading the RFC, the proposed `SessionPrefixIndex` inside vLLM also looks quite appealing to me
>
> This is really great to hear @Yifan Qiao.
> Happy to share some of the approaches I took in SGL to ensure that we have somewhat uniform APIs across the board
>
> **Yifan Qiao**（凌晨 5:44）回复： Thanks!
> Yep that will be great!
>
> **Karen Chung**（早上 6:17）回复：
> > With that being said, I feel orchestrator can still mainly a similar data structure separately, which is a global view vs. per-engine view of sessions with vLLM-internal `SessionPrefixIndex`.
> > And the two `SessionPrefixIndex`s (inside engine and orchestrator) can still talk with KVEvent.
> > Does this make any sense to you?
>
> @Yifan Qiao Totally.
> On Dynamo's end we are aligning towards having both data structures: (1) building a global Dynamo SessionPrefixIndexer (#13279, also linked in vLLM RFC #51428) and (2) contributing a similar indexer structure to vLLM engines such that the vLLM scheduler has visibility into it.
>
> For (1) we would also require vLLM KVEvents to emit session information, which @vMaroon has been pushing on (vllm RFC: <https://docs.google.com/document/d/1K0i0W5-n9-xuPbMeoWWBjh-SRtBf4YdAzBCfFViq8wg/edit?usp=sharing>). For (2) I'm still thinking about what the cleanest design is which would also satisfy our offloading team's (working on KVCC project) desiderata.
> We'll get back to you on more design specifics once we have them, though it will indeed closely resemble @ishan's SGL feature work.
>
> **Maroon Ayoub**（晚上 6:45）回复：
> > **ishan** Can you explain a bit more on *maybe* make use of it for a more vertical view or boundary APIs
>
> As in since we need the equivalent view for offloading tiers, we should find alignment and reuse if possible (offloading team input required)

---

## 8月18日

### Francisco Javier Arceo（发布于 #sig-core）

🌊 Hey folks, I drafted an RFC on how vLLM and the #sig-agentic-api should evolve together to support agentic inference workloads.
The MVP now supports Codex and Claude Code with a richer feature set than what comes out of core vLLM today.
The RFC lays out options for tighter integration with vLLM.
I'd appreciate feedback on the framing, scope, and options:

<https://github.com/vllm-project/vllm/issues/52567>

---

### Francisco Javier Arceo（凌晨 12:56）

FYI folks, sharing here now that #sig-agentic-api has an official MVP

---

## 8月18日—19日（KVEvents session_id 设计讨论，续）

### 在 Maroon 早上 7:05 消息的回复串中

**Karen Chung**（凌晨 2:51）

re: KvEvents -- what's our current consensus on <https://github.com/vllm-project/vllm/pull/51381> (`session_id` plumbing on kv events)?
To me it looks like a reasonable lightweight change, regardless of whether we add stuff like continuation_id on top of it later.
Any design discussions to resolve before it gets reviewed?

**Maroon Ayoub**（凌晨 2:54）

this goes in the direction outlined in the doc, where the semantics of the events change slightly and the router is responsible for invoking some of the publishing using the reporting-mode flag.
It's a good first step, but whether it is temporary or not depends on the possibilities/semantics we get when we have vLLM-internal tracking that can be utilized to report session-set per event

**Karen Chung**（凌晨 12:16）

Yea makes sense.
If I understand correctly, the two knobs for session reporting from KV events are

1. Incremental vs full
2. Single-session vs session-set (for blocks which are possibly assoc.
   with multiple) per event

And #51381 for now would enable `{incremental, full} x single-session` but not session-set; having a session prefix indexer in vLLM would unblock session-set.
I think merging #51381 would be still helpful before we build on the internal tracking piece, wdyt

It would unblock some of the orchestrator router works 🙂

**Maroon Ayoub**（凌晨 12:17）

maybe we can go for `x session-set` right away and build the semantics assuming that, which would contain support for individual sessions.
We fall-back to requesting full-reporting if something is missing

the latter can remain a robustness policy or get deprecated if useless later

**Karen Chung**（凌晨 12:19）

For full-reporting, are you mostly seeing it as a resilience/fault tolerance piece?
Or do you see other scenarios where it's useful for orchestrator/router

In my mind router would only need full reporting at startup or re-startup.
Curious what you had in mind

**Maroon Ayoub**（凌晨 12:21）

without internal index, it does play a critical role because KVEvents would not automatically fire for every session, they fire only when the blocks are new.
With this, full-reporting is critical for associating sessions with blocks without content-derived session-id mgmt

with internal index, every event can be a standalone source of truth of associated sessions, or well, only if we make it as such (e.g., fire when the **session index** updates as opposed to only when a block is written)

an alternative is a separate event per added session, which can work with individual session in event

technically in both directions it can play the first role, but in the 2nd we have an option of making it only resilience/fault tolerance

**Karen Chung**（凌晨 12:27）

> because KVEvents would not automatically fire for every session, they fire only when the blocks are new.

I see.
You're referring to scenarios when a block has cache hits but doesn't emit Stored events, right?
But this isn't a problem if the orchestrator/router can compute that cache hit itself, right?
Then router doesn't need vLLM to report the session affiliation that it already knows

> maybe we can go for `x session-set` right away and build the semantics assuming that

yea, maybe its as simple as changing session id from `str` to `set(str)` 🙂

**Maroon Ayoub**（凌晨 12:28）

i think this is the last thing to converge on, let me put a clearer msg after the llm-d community call + llm-d-router call in the next 1.5 hrs

or we can hijack the latter and discuss there 🤣

10 AM ur time

**Karen Chung**（凌晨 12:30）

haha would love to be an impostor but have a few meetings this morning.
Let's discuss when we're both freed

@ishan for viz as well

---

### Maroon Ayoub（凌晨 1:47）— 方案归纳

sorry took some time, I think we have 2 options:

1. events contain individual session-id
   1. **without** engine-internal tracking: the router would need to use `full report mode` upon detecting an "unmapped session".
      This is due to the current semantics of publishing an admission event only when a block is admitted (first writer session)
      1. if a mapping is missing, the router needs to assume that another session may have a shared prefix, therefore it needs the engine to re-produce all the existing blocks with the live session's tag - otherwise the engine wont publish the shared prefix
         1. an alternative is for the router to fix the mappings itself assuming shared block-hashes
   2. **with** engine-internal tracking: whenever a session mapping is appended to a block, we publish an individual admission event
2. events contain session-id set:
   1. **without** engine-internal tracking: practically not possible/practical unless im missing something
   2. **with ..**: same as 1.b., but each event is a full source of truth

intuitively I lean for 1.b with 1.a as fallback/recovery.
but need to run some mental trajectories

**Maroon Ayoub**（凌晨 2:02）

thoughts?

**Maroon Ayoub**（凌晨 2:07）

TL;DR IMO we can continue with the PR

**ishan**（凌晨 2:39）

> 1.b with 1.a as fallback/recovery

this tracks and plays nicely with sessionindexer at orchestrator level iiuc right?
pls sanity check me @Karen Chung

**Karen Chung**（凌晨 3:23）

> b.
> **with** engine-internal tracking: whenever a session mapping is appended to a block, we publish an individual admission event

I think it's nice to have ("admission" event for new session mapping on an existing block) for sure, but technically we don't need this from the orchestrator POV right.
At least for Dynamo router, orchestrator would be able to reconcile the fact that the block already existed with a different session-id, so vLLM telling it about that could be redundant.

But no objections with pushing PR forward with the current shape (1a) and revisiting additions in 1b direction once we have engine-internal session block tracking

For `session set` I'm similarly not sure if vllm needs to tell orchestrator "hey i know you sent me this request with session id 123, but btw its also associated with session id 012 based on some previous request".
Maybe it's nice to have for implementing policies directly inside vLLM scheduler level though

---

## 8月19日

### Yifan Qiao（中午 12:20）

@pisceskkk and his colleagues also have some ideas on agentic hints; @pisceskkk feel free to invite more people to here and #sig-agentic-api for discussions with a broader audience!

---

### Tingzhu Fang（下午 3:42）

Hello everyone, recently our team has been conducting some work related to agent hint.
We have also obtained some results based on agent testing and have written an RFC summarizing our work: <https://github.com/vllm-project/vllm/issues/52113>. You can read it first, and then we will discuss it together.
We hope to receive your opinions.
Thank you.

> **Karen Chung**（晚上 9:59）回复： Cool, this is very directionally aligned with our RFC <https://github.com/vllm-project/vllm/issues/51428> — kv hints + in-engine block-session prefix 🙂
>
> **Francisco Javier Arceo**（晚上 11:27）回复： hey all, jumping in from the #sig-agentic-api side.
> I'm one of the maintainers, and we've been working on the API and control-plane side of this same problem, so very excited to see these!
>
> i just opened a draft RFC for server-managed `messages` prompt-cache state: <https://github.com/vllm-project/agentic-api/pull/195>
>
> i'm directionally aligned with both proposals.
> `agentic-api` can normalize protocol state into trusted session and lifecycle hints, llm-d can handle placement, and vllm can resolve physical KV reuse.
> i've cross-linked both RFCs so we can keep the designs coordinated and i'd love to collaborate on landing these together for agentic workloads!

---

## 8月20日

### Maroon Ayoub（凌晨 1:42）

thoughts on <https://github.com/vllm-project/vllm/pull/38514>? we can turn it session-addressable, but feels like all gpu-retention logic is a derivative of this (cc @Hyunkyun Moon(Moreh))

> **[PR #38514] Context-Aware KV-Cache Retention API (#37003)** Purpose: Implementation of the Context-Aware KV-Cache Retention API proposed in #37003.
> Adds a token-range retention directive that lets the orchestrator annotate requests with per-range eviction priorities, enabling efficient KV-cache management for agentic workloads where tool-call pauses cause false eviction of reusable KV blocks.
> Changes: `vllm/v1/core/priority_eviction_queue.py` (new): Min-heap based priority eviction queue with lazy TTL expiry
>
> **Maroon Ayoub**（凌晨 1:48）回复： @Karen Chung @ishan @Yifan Qiao @Francisco Javier Arceo @Tingzhu Fang
>
> **Maroon Ayoub**（凌晨 1:48）回复： the PR covers GPU mgmt only, I think offloading needs its own
>
> **ishan**（凌晨 1:49）回复： This is similar to the TRTLLM feature right?
>
> **Maroon Ayoub**（凌晨 1:50）回复： yep
>
> **Maroon Ayoub**（凌晨 1:50）回复： went for about exact API to make it easier to go through but that didnt help 😂
>
> **Maroon Ayoub**（凌晨 1:53）回复： though it needs to be detached from the request path and into a standalone API
>
> **Maroon Ayoub**（凌晨 1:53）回复： e.g., have flavors of:
>
> 1. attaching to a request
> 2. providing hashes (suboptimal)
> 3. providing session-id
>
> **ishan**（凌晨 2:38）回复： First thought - I see this as a bit separate (but yes still related to) from the 会话 aware 搬运 work.
> Like instead of a DEREF/EVICT hint or something you could use these APIs to accomplish something like that.
>
> I will think a bit more about this over the weekend.
> Have been going back and forth on getting this into other FW for same reason

---

## 8月22日

### Karen Chung（上午 10:42）

Hey all, Dynamo team (cc @ishan) has converged on a concrete shape for the **"KvHint envelope" protocol** (see prev.
discussed RFC #51428 for context) in vLLM.
We kept the protocol flexible (broadly compatible w/ orchestrators) and expressive (can extend to express any action type, e.g. fetch/demote/etc we envision).
This should align well with direction of @vMaroon @Tingzhu Fang's works too.

Here's the RFC #53421 which also links the companion PR #53423 🙂 @Nick Hill @Yifan Qiao would appreciate feedback / review 🙏

> **Karen Chung**（凌晨 5:04）回复： Relatedly, Dynamo team is working on KVCR (KV Cache Runner) which will consume this for hint-driven KV cache policies.
> The project is public here: <https://github.com/ai-dynamo/kvcr> and there's a relevant vLLM PR #53624 (which will rebase onto the KvHint PR #53423). 🙂
>
> **Karen Chung**（早上 7:17）回复： @Zhewen Li @Nick Hill could I get another review on <https://github.com/vllm-project/vllm/pull/53423> 🥺 added the grpc plumbing Zhewen requested

---

## 8月底

### Maroon Ayoub（下午 5:37）

<https://github.com/vllm-project/vllm/pull/51381#top> @Nick Hill @Yifan Qiao can u pls take a look at this

> **Yifan Qiao**（凌晨 4:32）回复： Sorry for the delay!
> I just approved it.
> There seems to be a conflict

---

## 9月3日

### Maroon Ayoub（凌晨 12:40）

x-post from #feat-v1-cpu-offloading — fixes for the simple offloading connector on kvevents production

### Maroon Ayoub（发布于 #feat-v1-cpu-offloading）

hey @Yifan Qiao @Nick Hill - can we pls get this through?
<https://github.com/vllm-project/vllm/pull/54325> cc @Flex @Robert Shaw

I'll read the file to get the full content and then restructure it.

Let me organize this chat log into a more readable format.

---

## #feat-session-aware 频道记录

> **频道创建：** 2026年8月15日，由 @Maroon Ayoub 创建

---

## 1. 频道成立与目标定义（8月15日）

**Maroon Ayoub** 发起频道，说明频道方向：

> 这个频道聚焦于推进面向 agentic inference 的推理栈演进——特别是 session-centric orchestration，它契合 agentic 场景并能解锁一系列优化。
> 我们会在这里汇集 RFC 和阅读资源。

**当前进展：** @Karen Chung 已完成基础管道搭建，下一步聚焦两个方向：

1. **KVEvents**（KV 缓存事件机制）
2. **Offloading/Retention APIs**（KV 缓存卸载/保留接口）

**Francisco Javier Arceo** 补充：需要与 #sig-agentic-api 频道协调。

---

## 2. KVEvents 中的 session_id 设计讨论（8月15日—8月20日）

围绕 PR [#51381](https://github.com/vllm-project/vllm/pull/51381)（在 KV events 上传递 `session_id`）展开的核心设计讨论。

### 2.1 问题背景

KVEvents 当前只在缓存块首次写入时触发。
如果多个会话共享前缀缓存块，后续会话不会触发事件，导致 router 无法建立完整的会话→缓存块映射。

### 2.2 两个设计维度

Karen Chung 梳理了两个正交维度：

| 上报方式 | 单会话 | 会话集合 |
| ------------ | ------------------ | -------------------------------------- |
| **增量上报** | 仅报告变化 | 仅报告变化，但包含完整会话集合 |
| **全量上报** | 重新上报所有缓存块 | 重新上报所有缓存块 + 完整会话集合 |

### 2.3 Maroon 的方案归纳

#### 方案 1：事件包含单个 session-id

- **1.a 无引擎内部追踪：** router 在检测到"未映射会话"时需要触发 `full report mode`，让引擎重新上报所有已有缓存块。
  或者 router 自行通过缓存块 hash 推断映射。
- **1.b 有引擎内部追踪：** 每当一个会话映射被追加到某个缓存块时，引擎发布一条独立的 admission event。

#### 方案 2：事件包含 session-id 集合

- **2.a 无引擎内部追踪：** 无法维护该集合。
- **2.b 有引擎内部追踪：** 同 1.b，但每个事件自身是完整的 source of truth。

**倾向结论：** 采用 **1.b**（引擎内部追踪 + 单会话事件），以 **1.a** 作为回退/恢复机制。
PR #51381 可以继续推进。

### 2.4 Karen 的补充观点

从编排端角度看，1.b 是 nice-to-have 但非必须——Dynamo router 自身可以推导出缓存块已存在但属于不同会话的情况，不一定需要 vLLM 主动通知。
`session set` 同理，vLLM 不一定需要告诉编排端一个缓存块同时关联了哪些会话，除非是在 vLLM scheduler 层面直接实现策略。

---

## 3. vLLM 引擎内 SessionPrefixIndex 的边界讨论（8月15日—8月22日）

由 Karen Chung 发起，核心问题：**vLLM 引擎与编排端（Dynamo/llm-d）之间的职责边界是什么？**
**vLLM 是否应接受新的索引结构？**

### 3.1 相关 RFC

- vLLM RFC [#51428](https://github.com/vllm-project/vllm/issues/51428)：提出 `SessionPrefixIndex` 的两种设计选择——放在 vLLM 内部 vs 放在 Dynamo 内部。

### 3.2 Yifan Qiao 的立场转变

> 我原本倾向于用 KVEvent 作为 vLLM 引擎与编排端之间的唯一接口。
> 但读完 RFC 后，在 vLLM 内部放置 `SessionPrefixIndex` 也很有吸引力，我现在支持这个方向。

他同时提出编排端仍可维护自己的类似数据结构（全局视图 vs 引擎内的局部视图），两者通过 KVEvent 通信。

### 3.3 达成的方向

Karen Chung 确认 Dynamo 团队的方向是**两层都建**：

1. **全局层：** Dynamo SessionPrefixIndexer（[Dynamo #13279](https://github.com/ai-dynamo/dynamo/issues/13279)）
2. **引擎层：** 向 vLLM 引擎贡献类似的 indexer 结构，使 vLLM scheduler 对此可见

全局层需要 vLLM KVEvents 携带会话信息（Maroon 在推进的 [KvEvents RFC](https://docs.google.com/document/d/1K0i0W5-n9-xuPbMeoWWBjh-SRtBf4YdAzBCfFViq8wg/edit?usp=sharing)）。
引擎层的设计还在细化，需满足 offloading 团队（[KVCC 项目](https://github.com/ai-dynamo/dynamo/issues/11673)）的需求，会参考 @ishan 在 SGL 中的相关工作。

Maroon 补充：offloading 场景**必须**有这样的结构来实现 session-addressable KV 搬运 APIs。
如果 vLLM core 有了，offloading 层可以直接复用。

---

## 4. Agentic Hint 相关 RFC（8月19日—8月20日）

### 4.1 Tingzhu Fang 团队的 RFC

[vLLM #52113](https://github.com/vllm-project/vllm/issues/52113)：Agent hint 相关工作及基于 agent 测试的成果总结。

Karen Chung 回复：方向上与 RFC #51428（KV hints + 引擎内 block-session 前缀索引）高度一致。

### 4.2 Francisco Javier Arceo 的 sig-agentic-api 侧 RFC

[vLLM #52567](https://github.com/vllm-project/vllm/issues/52567)：vLLM 与 #sig-agentic-api 如何协同演进以支持 agentic inference 工作负载。
MVP 已支持 Codex 和 Claude Code，功能比原生 vLLM 更丰富。

他同时发布了 server-managed `messages` prompt-cache state 的草案 RFC：[agentic-api PR #195](https://github.com/vllm-project/agentic-api/pull/195)。

> 方向上与两个提案一致。
> `agentic-api` 可以将协议状态规范化为可信的会话和生命周期 hints，llm-d 处理调度，vLLM 解析物理 KV 复用。

---

## 5. Context-Aware KV-Cache Retention API（8月20日）

Maroon Ayoub 提出讨论 PR [#38514](https://github.com/vllm-project/vllm/pull/38514)：

**功能：** 让编排端为请求标注 token-range 级别的驱逐优先级，解决 agentic 场景中 tool-call 暂停导致可复用 KV 缓存块被误驱逐的问题。
实现基于最小堆 + 懒惰 TTL 过期。

**讨论要点：**

- 该 PR 仅覆盖 GPU 管理，offloading 需要单独处理
- API 设计参考了 TRT-LLM 的同类功能
- 需要从请求路径中解耦，成为独立 API，支持三种调用方式：
  1. 附加到请求
  2. 提供 hash（次优）
  3. 提供 session-id
- @ishan 认为这与 session-aware 搬运工作有关但相对独立，可作为 DEREF/EVICT hint 的替代实现方式

---

## 6. KvHint Envelope 协议（8月22日—9月初）

### 6.1 RFC 与 PR

Karen Chung 宣布 Dynamo 团队已收敛出 **KvHint 提示消息协议**的具体形态：

- RFC [#53421](https://github.com/vllm-project/vllm/issues/53421)
- 配套 PR [#53423](https://github.com/vllm-project/vllm/pull/53423)

设计原则：**灵活**（兼容各种编排端）+ **可扩展**（可表达 fetch/demote 等各类操作）。

### 6.2 KVCR（KV Cache Runner）

Dynamo 团队正在开发 [KVCR](https://github.com/ai-dynamo/kvcr)，用于消费 KvHint 执行 hint-driven KV cache 策略。
关联 vLLM PR [#53624](https://github.com/vllm-project/vllm/pull/53624)（将 rebase 到 KvHint PR #53423 之上）。

---

## 7. Review 请求与推进（8月底—9月初）

- **PR #51381**（session_id on KV events）：Maroon 请求 @Nick Hill @Yifan Qiao review。
  Yifan 已 approve，存在 conflict 需解决。
- **PR #53423**（KvHint 提示消息）：Karen 请求 @Zhewen Li @Nick Hill 再次 review，已补充 Zhewen 要求的 gRPC plumbing。
- **PR #54325**（simple offloading 连接器的 KVEvents 修复）：Maroon 在 #feat-v1-cpu-offloading 频道请求 review。

---

## 附录：关键链接汇总

| 类型 | 编号/链接 | 描述 |
| ------------ | ------------------------------------------------------------ | ------------------------------------------ |
| vLLM RFC | [#51428](https://github.com/vllm-project/vllm/issues/51428) | Session-addressable KV cache 总 RFC |
| vLLM RFC | [#52113](https://github.com/vllm-project/vllm/issues/52113) | Agent hint RFC（Tingzhu Fang 团队） |
| vLLM RFC | [#52567](https://github.com/vllm-project/vllm/issues/52567) | vLLM + sig-agentic-api 协同 RFC |
| vLLM RFC | [#53421](https://github.com/vllm-project/vllm/issues/53421) | KvHint 提示消息协议 RFC |
| vLLM PR | [#38514](https://github.com/vllm-project/vllm/pull/38514) | Context-Aware KV-Cache Retention API |
| vLLM PR | [#51381](https://github.com/vllm-project/vllm/pull/51381) | session_id plumbing on KV events |
| vLLM PR | [#53423](https://github.com/vllm-project/vllm/pull/53423) | KvHint 提示消息实现 |
| vLLM PR | [#53624](https://github.com/vllm-project/vllm/pull/53624) | KVCR 相关 vLLM 改动 |
| vLLM PR | [#54325](https://github.com/vllm-project/vllm/pull/54325) | offloading 连接器 KVEvents 修复 |
| Dynamo Issue | [#13279](https://github.com/ai-dynamo/dynamo/issues/13279) | Dynamo SessionPrefixIndexer |
| Dynamo Issue | [#11673](https://github.com/ai-dynamo/dynamo/issues/11673) | KVCC 项目 |
| 外部项目 | [KVCR](https://github.com/ai-dynamo/kvcr) | KV Cache Runner |
| Agentic API | [PR #195](https://github.com/vllm-project/agentic-api/pull/195) | server-managed messages prompt-cache state |
| Google Doc | [KvEvents RFC](https://docs.google.com/document/d/1K0i0W5-n9-xuPbMeoWWBjh-SRtBf4YdAzBCfFViq8wg/edit?usp=sharing) | KvEvents 语义变更 RFC |
