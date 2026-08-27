# vLLM V1 KV-Aware Scheduling & Prefix Cache Management 项目改造设计说明

> **文档用途**：这是项目级设计说明，不是直接照抄的 patch 指令。后续 Coding Agent 必须先基于已经冻结的 vLLM **实际 base commit** 阅读源码，核对本文涉及的类、函数、调用链和状态语义，再输出精确到文件/函数/测试用例的实施方案；确认后再改代码。
>
> **目标**：一周内、单卡 RTX 4090、尽量只使用 vLLM 原有测试和 benchmark 工具，完成三个相互关联的工程改造：
>
> 1. 轻量级 Scheduler Policy 解耦
> 2. Recompute-Aware Preemption & Re-admission
> 3. Waiting-Queue-Informed Prefix Cache Eviction
>
> 可选第 4 点：Reclaimable-KV-Aware Victim Scoring。前三项全部稳定后再考虑。

---

## 0. Agent 执行规则

### 0.1 第一阶段不要写代码

先确认：

- `git rev-parse HEAD`
- 当前项目分支
- `git status` 是否干净
- V1 Scheduler 的真实调度循环
- RequestQueue / Request 的真实排序方式
- `_preempt_request` 的状态变化
- preemption 后 KV block 的释放与 recomputation 路径
- `KVCacheManager.allocate_slots` 的真实调用链
- `BlockPool.get_new_blocks` 与 free-block eviction order
- prefix cache lookup 是否有副作用
- 已有测试里可直接复用的 case

如果本文和实际 base commit 不一致，以 **base commit** 为准；不要为了匹配本文去更新 upstream。

### 0.2 不做大重构

本项目不是实现完整 Scheduler Plugin Framework，也不是完整复刻 FastServe/Preble。

目标是：

> 抽出少量“策略决策点”，让 policy 决定“选谁”，而 Scheduler Core 继续拥有 allocation、free、request state mutation、batching、KV bookkeeping 等 correctness-critical 操作。

核心逻辑尽量控制在约 **300–600 行 + 测试**。如果明显膨胀，先缩 scope。

### 0.3 默认行为必须可保留

需要能够明确比较：

```text
baseline
vs
optimized
```

新策略关闭时：

- FCFS 不应被意外改变
- Priority 不应被意外改变
- Prefix Cache 默认 LRU 不应被意外改变
- 已有 eviction/block order 测试不应被无意破坏

---

# 1. 项目定位

推荐项目名：

**KV-Aware Scheduling and Prefix Cache Management for vLLM V1**

中文：

**基于 vLLM V1 的 KV Cache 感知调度与 Prefix Cache 管理优化**

这个项目应该是一条完整链路，而不是三个 patch：

```text
Request / Waiting Queue
          │
          ▼
   Scheduler Policy
          │
   ┌──────┴──────────────┐
   │                     │
   ▼                     ▼
Re-admission         Victim Selection
   │                     │
   └──────────┬──────────┘
              ▼
        KV Allocation
              │
        allocation fails
              │
              ▼
          Preemption
              │
              ▼
     KV blocks / cache state
              │
              ▼
      Prefix Cache Eviction
              │
              ▼
      Future Prefix Reuse
```

面试时可以从一次 request 的完整生命周期讲 Scheduler → KVCacheManager → BlockPool。

---

# 2. 项目约束

- 单卡 RTX 4090 24GB
- 代码落地几天，一周内形成可写简历版本
- 不做新科研数据集
- 不开发全新 benchmark framework
- 不做多 GPU 系统
- 不改 CUDA kernel
- 不做大规模 C++ 重写
- 不完整复刻 MLFQ / Preble routing
- 不为了凑点做无关新模型适配
- 可以下载一个 2B/3B/7B 正常模型
- 可以使用现有 pytest
- 可以使用 `benchmark_prefix_caching.py`
- 可以使用 `vllm bench serve --dataset-name prefix_repetition`
- 可以通过限制 KV cache 大小主动制造 memory pressure

优先级：

1. correctness
2. 设计可解释
3. 测试和 commit 完整
4. 最后才是性能数字

---

# 3. 当前 vLLM V1 代码事实：Agent 必须复核

本文撰写时（2026-08-27）观察到的 main 行为如下，但实现必须以用户冻结的 commit 为准。

## 3.1 Scheduler

重点文件：

```text
vllm/v1/core/sched/scheduler.py
vllm/v1/core/sched/request_queue.py
vllm/v1/request.py
```

当前结构中：

- `SchedulingPolicy` 至少有 `FCFS` / `PRIORITY`
- Scheduler 持有 `waiting`、`skipped_waiting`、`running`
- queue 由 `create_request_queue(self.policy)` 创建
- Priority queue 使用 heap
- `Request.__lt__` 主要按：
  1. user priority
  2. arrival time
  3. request id
- Request 已有 `num_preemptions`

Priority 模式下，KV allocation 失败后的 victim 路径当前存在类似：

```python
preempted_req = max(
    self.running,
    key=lambda r: (r.priority, r.arrival_time),
)
```

这说明 victim 选择主要知道 priority / arrival time，却不知道：

- 已经计算了多少 token
- 抢占后潜在重算多少
- 真正能释放多少独占 KV block

这就是项目第二点的入口。

## 3.2 KVCacheManager

重点：

```text
vllm/v1/core/kv_cache_manager.py
```

关注：

```text
get_computed_blocks(...)
allocate_slots(...)
free(...)
get_blocks(...)
get_block_ids(...)
```

`allocate_slots()` 负责判断请求所需 KV block 能否分配，空间不足时可以返回 `None`，Scheduler 再决定如何处理。

因此边界应保持：

> Scheduler 决定“谁运行/谁被抢”；KVCacheManager 决定“KV 能不能分配”。

## 3.3 BlockPool / Prefix Cache

重点：

```text
vllm/v1/core/block_pool.py
vllm/v1/core/kv_cache_utils.py
```

当前 BlockPool：

- 预创建 KV blocks
- `free_block_queue` 同时承载 free blocks / cached eviction candidates 的顺序
- cached block hash table 支持 prefix lookup
- `get_new_blocks()` 从 free queue 取 block
- 如果拿到 cached block，会通过正式 eviction 路径清理 hash/event/metrics
- `free_blocks()` 根据 block 属性恢复 eviction order

项目第三点应该只改变：

> “必须复用 cached free block 时，优先选择哪个 block”

而不是重写 refcount 或 cache correctness。

---

# 4. 总体架构

```text
                         Scheduler Core
                    correctness/state mutation
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
        Scheduling Decision          KVCacheManager
              Policy                      │
                 │                        ▼
      ┌──────────┴──────────┐          BlockPool
      │                     │             │
      ▼                     ▼             ▼
Re-admission            Preemption    Prefix Eviction
Ordering                Victim         Preference
```

规则：

- Policy 返回决策
- Scheduler Core 执行决策
- Policy 不直接 free block
- Policy 不直接改 RequestStatus
- Policy 不改 refcount
- Prefix eviction policy 不改变 request admission order
- retained prefix 不是 hard pin

---

# 5. 改动一：Mini Scheduler Policy Abstraction

## 5.1 动机

scheduler.py 很大。如果每种新策略都继续塞：

```python
if policy == A:
    ...
elif policy == B:
    ...
```

会把策略与复杂 correctness 路径绑死。

本项目只抽自己真正需要的最小决策面。

## 5.2 最小接口目标

至少覆盖两个问题：

### A. 谁被抢占？

概念接口：

```python
select_preemption_victim(
    running_requests,
    current_request,
    context,
) -> Request
```

### B. preempted request 回 waiting 后怎么排序？

概念接口：

```python
get_waiting_order_key(request)
```

具体接口名由 Agent 看 base commit 后决定，不要机械照搬。

推荐概念结构：

```text
DefaultSchedulingDecisionPolicy
└── 完整保持当前行为

RecomputeAwareSchedulingDecisionPolicy
├── recompute-aware victim selection
└── preempted-request re-admission preference
```

不做：

- 动态插件包 discovery
- entry point
- out-of-tree registry
- 大量生命周期 callback
- 每 token callback

## 5.3 配置

必须能切换 baseline / optimized。

建议 Agent 找最小 typed config 入口，概念上类似：

```text
preemption:
- default
- recompute_aware

prefix eviction:
- lru
- waiting_queue_aware
```

优先复用当前 `SchedulerConfig`；不优先用散乱环境变量。

## 5.4 验收

- 默认路径与 base commit 等价
- 新 policy 只决定“选谁”
- request state mutation 仍由 Scheduler 完成
- unit test 可以直接验证 policy 行为

---

# 6. 改动二：Recompute-Aware Preemption

## 6.1 问题

同一 user-priority tier：

```text
A.num_computed_tokens = 128
B.num_computed_tokens = 3072
```

若必须抢一个，在其他条件相同的情况下，抢 A 通常比抢 B 的潜在恢复代价更低。

当前简单 priority/arrival-time victim selection 无法表达这一点。

## 6.2 第一版 cost estimator

保持简单：

```text
recompute_cost(request) ≈ request.num_computed_tokens
```

优点：

- 已有字段
- 不需要 profiling
- 不需要数据集
- 不需要预测模型
- deterministic
- 面试容易解释

P0 不做复杂预测。

## 6.3 user priority 是硬约束

绝不能出现：

> 高 user priority request 因为 recompute cost 小，就被低优先级 request 抢掉。

正确做法是分层：

```text
Step 1: 先确定 user priority 最差的 candidate tier
Step 2: 只在该 tier 内比较 recompute cost
Step 3: 相同 cost 再用稳定 tie-break
```

概念：

```text
victim tier = worst user-priority tier
victim = argmin(recompute_cost) within tier
```

priority 数值方向必须由 Agent 从当前源码确认。

## 6.4 P0 不使用 raw block count

不要第一版写：

```text
score = blocks / tokens
```

因为 request 持有的 blocks 可能共享，raw block count 不等于真正 reclaimable blocks。Hybrid KV group 也会让 accounting 更复杂。

P0 只做 token-based recompute cost。

---

# 7. Re-admission：被抢过的 request 优先恢复

## 7.1 问题

```text
A prefill 2000 tokens
→ 运行
→ 被抢
→ 回 waiting
→ 同 priority 新请求不断先执行
→ A 的可复用 cache 继续被挤掉
→ A 恢复时大量 recompute
```

项目应该在**同一 user priority tier 内**给 preempted request resume preference。

## 7.2 推荐 ordering

```text
1. user priority
2. 是否 preempted/resume
3. arrival time
4. stable request id
```

即：

```text
same user priority:
preempted > never-run
```

但：

```text
higher user priority new request
>
lower user priority preempted request
```

## 7.3 num_preemptions

第一版推荐使用：

```text
request.num_preemptions > 0
```

作为 binary resume signal。

这样语义比直接按“被抢次数越多越优先”更温和。

如果 base commit / 现有测试更适合直接把 `num_preemptions` 用作 tie-break，Agent 必须说明 fairness/starvation 影响。

注意 heap invariant：不要在对象仍处于 heap 中时改变影响比较结果的字段而不重建 heap。

---

# 8. 改动三：Waiting-Queue-Informed Prefix Cache Eviction

## 8.1 当前 LRU 的盲点

普通 Prefix Cache LRU 只知道：

> 哪个 cached free block 最久没用。

不知道：

> waiting queue 里谁马上就要用它。

例子：

```text
Original eviction order:
[A, B, C, D, E]

Waiting near head:
X needs B
Y needs D
```

普通 LRU 可能很快淘汰 B/D，导致 X/Y 几轮后 admission 时重新 prefill。

## 8.2 新策略

从 waiting queue 的前 N 个请求获取短期 reuse signal：

```text
Retained = cached prefix blocks
           likely needed by near-head waiting requests
```

例如：

```text
Original: [A, B, C, D, E]
Retained: {B, D}

Temporary preference:
[A, C, E, B, D]
```

B/D **不是 pinned**。

如果容量仍然不够：

```text
A,C,E 用完
→ 仍可淘汰 B,D
```

因此：

> 新策略不得让原本可成功的 allocation 因 retention 而失败。

---

# 9. Queue-Informed LRU correctness invariants

## 9.1 Retention lookup 必须 read-only

不允许：

- 增加 `ref_cnt`
- 从 free queue 移除 block
- 改 LRU 顺序
- 改 access metadata
- 修改 request block table
- 修改 request status
- 计入真实 prefix hit stats
- 发送真实 cache-access event
- 触发 remote KV transfer

它只回答：

> “这个 waiting request 如果近期 admission，本地现在有哪些连续 cached prefix blocks 可复用？”

## 9.2 Retained != pinned

必须：

```text
优先 non-retained
→ 不够
→ fallback retained
```

## 9.3 不改变 request scheduling order

只影响 cached free block eviction preference。

## 9.4 正式 eviction 仍走原路径

hash cleanup、events、metrics、refcount 等必须继续经过 base commit 原有 BlockPool eviction/accounting 路径。

---

# 10. Retained Set 候选范围

不要扫完整 waiting queue。

推荐：

```text
N ≈ max_num_running_reqs
```

或一个不超过它的 candidate window。

理由：

- near-head request 才是短期需求
- 限制 Scheduler CPU overhead
- Priority queue 内部 heap list 不是自然排序，Agent 必须通过真正“按 scheduling order 遍历”的接口取候选

---

# 11. Lazy retained-set construction

理想路径：

```text
allocation
    │
    ▼
有足够 uncached free blocks？
    │
   YES ──► 原 fast path，不扫描 waiting
    │
   NO
    ▼
即将牺牲 cached free blocks
    │
    ▼
此时才构建 retained set
```

同一次 `allocate_slots()` transaction 内最多构建一次，然后 lower-level allocations 复用。

Agent 勘察后可比较：

### 方案 A：Scheduler 预计算并传 `retained_block_ids`

简单，但不是 lazy。

### 方案 B：传 lazy resolver/context

更接近 RFC；真正需要 cached eviction 时才 resolve。

### 方案 C：KVCacheManager transaction context

Scheduler 只提供候选/hint，Manager 在本次 allocation 内维护临时 retained set。

原则：

> 选 base commit 上最小、最安全的实现，不为了“架构漂亮”过度设计。

---

# 12. 只读 Prefix Probe

可能需要增加概念 helper：

```python
peek_cached_prefix_blocks(request)
```

要求：

```text
Input: waiting Request
Output: 本地连续可复用 prefix block IDs / read-only view
Side effects: NONE
```

不要直接假设现有 `get_computed_blocks()` 完全纯读。

Agent 必须检查：

- coordinator 的 prefix lookup
- prefix stats
- KV events
- connector path
- hybrid KV group

若现有 API 已保证纯读，直接复用；否则做最小 helper。

---

# 13. 三个改动如何形成一个故事

```text
KV Memory Pressure
        │
        ▼
allocate_slots() fails
        │
        ▼
Recompute-aware victim
        │
少牺牲昂贵计算历史
        │
        ▼
request preempted
        │
        ▼
Re-admission preference
        │
尽早恢复已做过工作
        │
        ▼
waiting queue 暴露近期 demand
        │
        ▼
Queue-informed Prefix Eviction
        │
优先保留即将复用 prefix
        │
        ▼
减少重复 prefill/recompute
```

项目总思想：

> 让 Scheduler 与 Prefix Cache Manager 利用 request execution history 和 near-future queue demand 做轻量协同，而不是各自只看局部状态。

---

# 14. 与论文 / 社区工作的关系

## FastServe

不完整复刻 MLFQ。

吸收：

> 细粒度 preemptive serving 中“执行历史与恢复成本应参与调度”的思想。

项目落地为：

- recompute-aware victim
- preempted request re-admission

## Preble

不做多 GPU routing。

吸收：

> cache locality / future reuse 应影响系统决策。

项目缩小为：

```text
future prefix demand
→ local prefix block retention/eviction preference
```

## vLLM 当前 RFC

本项目与两个方向高度一致：

1. Scheduler Plugin Framework：policy 做 decision，core 保持 correctness。
2. Waiting-Queue-Informed LRU：waiting queue 作为 near-term demand，retained 不是 pin，lookup 只读，allocation 必须 fallback。

---

# 15. Optional Bonus：Reclaimable-KV-Aware Victim Scoring

**前三项全部完成后再做。**

## 15.1 raw block count 不等于 reclaimable count

如果 A 看似持有 100 blocks，其中 80 与其他 request 共享，抢 A 后真正变 free 的可能很少。

所以不能：

```text
reclaimable = len(get_block_ids(request))
```

## 15.2 可选 helper

概念：

```python
get_reclaimable_block_count(request) -> int
```

必须考虑：

- ref_cnt
- shared prefix
- KV groups
- null/padding block
- 当前 coordinator 语义

## 15.3 可选 score

可以考虑：

```text
preemption efficiency
≈ reclaimable_blocks / (1 + recompute_cost)
```

但更推荐先做：

```text
满足所需释放量
→ 在可行 candidates 中最小化 recompute cost
```

更容易解释和测试。

---

# 16. 预计触碰文件

以实际 commit 为准：

```text
vllm/v1/core/sched/scheduler.py
vllm/v1/core/sched/request_queue.py
vllm/v1/request.py                  # 尽量少动
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/block_pool.py
vllm/config/scheduler.py            # 若需 typed config

可能新增：
vllm/v1/core/sched/policy.py

测试：
tests/v1/core/test_scheduler.py
tests/v1/core/test_prefix_caching.py
以及当前 repo 中已有的 request_queue 测试
```

---

# 17. 测试方案

不开发新的 benchmark framework。

## 17.1 Scheduler / policy unit tests

### Test A：default compatibility

新 feature 关闭时，与 base commit 行为一致。

### Test B：strict user priority

```text
A: 更高 user priority, cold
B: 更低 user priority, preempted
```

A 仍必须优先。

### Test C：same-tier re-admission

```text
A: same priority, preempted, arrival later
B: same priority, cold, arrival earlier
```

optimized 模式下 A 先恢复。

### Test D：recompute-aware victim

```text
A: same worst priority, num_computed_tokens=128
B: same worst priority, num_computed_tokens=2048
```

victim 应为 A。

### Test E：priority beats recompute cost

即便低优先级 request 重算成本很高，也不能去抢更高 user priority request。

## 17.2 BlockPool / Prefix Cache tests

### Test F：eviction order

```text
LRU:      [A, B, C, D, E]
retained: {B, D}
```

期望：

```text
A, C, E, B, D
```

### Test G：fallback

需要的 blocks 多于 non-retained 时，仍可淘汰 retained，不得 allocation fail。

### Test H：retained 不是 pin

容量不足时 retained 最终可以正式 eviction。

### Test I：read-only probe

probe 前后验证：

- ref_cnt
- free queue order
- cached hash mapping
- stats/events（当前测试设施允许时）
- request block state

不变。

### Test J：fast path

如果 uncached free blocks 已足够，retention resolver 不应被调用。

## 17.3 Scheduler + KV integration

构造：

```text
waiting X 有 cached prefix B
其他 request 的 allocation 产生 cached-block pressure
```

Baseline：

```text
B 被普通 LRU 提前淘汰
```

Optimized：

```text
B 暂时 retained
X 后续仍有更长 local prefix hit
```

优先在已有 `test_scheduler.py` / `test_prefix_caching.py` 完成。

---

# 18. GPU Benchmark

Unit tests 稳定后再跑。

## 18.1 只用 vLLM 已有工具

优先：

```bash
vllm bench serve   --dataset-name prefix_repetition   ...
```

以及：

```bash
python benchmarks/benchmark_prefix_caching.py ...
```

fixed-prompt 模式即可，不需要新数据集。

## 18.2 模型

- debug：2B/3B
- final：4090 能稳定运行的 7B 左右模型
- 模型不是项目贡献点
- 不主动引入 AWQ/GPTQ 等额外变量

## 18.3 制造 KV pressure

通过 base commit 已有 KV cache memory/config 选项缩小 cache budget：

```text
24GB 4090
→ 主动限制 KV cache 可用量
→ 触发 preemption / cached eviction
```

参数名由 Agent 根据 base commit 的 CLI/config 核实。

## 18.4 对比

最低只需要：

```text
Baseline:
default preemption + default LRU

Optimized:
recompute-aware preemption/re-admission
+ waiting-queue-informed LRU
```

有时间再做 ablation：

```text
A baseline
B only scheduler
C only cache
D both
```

---

# 19. 结果记录

优先复用现有 stats：

- request throughput
- token throughput
- TTFT
- TPOT/ITL（现有 bench 有则记录）
- prefix cache hit
- preemption count
- recomputed tokens（已有统计可得时）

不要为了一个数字开发大型 instrumentation。

**真实测量前禁止写虚构提升百分比。**

---

# 20. 一周执行顺序

## Day 0

- 冻结 commit
- 环境 baseline
- scheduler/prefix tests
- 小模型启动

## Day 1

- 代码勘察
- Mini Policy abstraction
- 默认行为测试

## Day 2

- Recompute-aware victim selection
- strict priority tests

## Day 3 上午

- Re-admission ordering
- heap/queue invariant tests

## Day 3 下午～Day 4

- BlockPool deterministic queue-aware eviction
- 再接 waiting queue

## Day 5

- regression
- targeted pytest
- lint/type checks（项目范围）

## Day 6

- 4090 built-in benchmark
- baseline vs optimized

## Day 7

- README
- architecture
- result table
- commit cleanup
- 简历/面试梳理

**Day 4 后原则上不再加新 feature。**

---

# 21. Git Commit 规划

建议：

```text
feat(scheduler): add lightweight scheduling decision policy

feat(scheduler): add recompute-aware preemption selection

feat(scheduler): prioritize preempted requests on re-admission

feat(kv-cache): add waiting-queue-informed prefix retention

test(scheduler): cover recompute-aware scheduling decisions

test(prefix-cache): cover queue-aware eviction ordering
```

每个逻辑点：

```text
git diff
→ targeted tests
→ review
→ commit
→ push
```

下一步前 working tree 保持干净。

---

# 22. P0 验收标准

## Architecture

- [ ] policy 与 correctness mutation 边界清楚
- [ ] 默认路径可用
- [ ] 不需要 CUDA/C++ 改造

## Preemption

- [ ] user priority 为硬约束
- [ ] 同 tier 可按 recompute cost 选 victim
- [ ] preempted request 有清晰 re-admission 语义
- [ ] 不破坏 heap/request queue invariant

## Prefix Cache

- [ ] waiting queue 提供短期 reuse signal
- [ ] prefix probe read-only
- [ ] retained 不是 pinned
- [ ] 不足时 fallback
- [ ] allocation feasibility 不变
- [ ] eviction accounting 仍走官方路径

## Tests

- [ ] scheduler targeted tests
- [ ] prefix caching targeted tests
- [ ] 新 deterministic tests
- [ ] default compatibility tests

## GPU

- [ ] 单 4090 模型正常启动
- [ ] built-in prefix workload 跑通
- [ ] baseline / optimized 有可复现命令

---

# 23. 明确不做

Agent 不得自行扩 scope：

- 完整 Scheduler Plugin Framework
- MLFQ
- SJF/EWSJF 全套
- VTC fairness accounting
- P/D disaggregation
- multi-GPU router
- TP/PP 改造
- KV CPU offload
- KV compression
- PagedAttention CUDA kernel
- FlashAttention/FlashInfer kernel
- speculative decoding 新算法
- 新模型 native implementation
- 新 dashboard
- 新科研 dataset
- 自研 load generator

---

# 24. 风险与降级

## Policy abstraction 太大

降级为只抽：

```text
select_preemption_victim
+
waiting ordering helper
```

## Lazy retention wiring 太复杂

降级顺序：

```text
完整 lazy context
→ 每 allocate_slots 构建一次
→ 每 scheduler step 构建一次
```

不要退化成“每分配一个 block 扫一次 waiting queue”。

## Hybrid/Connector 过于复杂

P0 先保证：

- 单 GPU
- decoder-only
- local prefix cache

新策略默认关闭；unsupported path 可明确 fallback 原 LRU。不要为了全兼容拖垮一周项目。

## Benchmark 没差异

先检查：

- 是否真的发生 preemption
- 是否真的发生 cached eviction
- waiting request 是否真的复用 prefix

没有 pressure 就看不到策略收益。

---

# 25. 面试必须理解的问题

1. 一轮 vLLM scheduler iteration 做什么？
2. `allocate_slots` 为什么失败？
3. preemption 后 request 的 KV 怎么处理？
4. 为什么 `num_computed_tokens` 能当第一版 recompute proxy？
5. 为什么 user priority 必须高于 recompute cost？
6. preempted resume 与新 request admission 有何不同？
7. cached block 为什么可以 `ref_cnt == 0`？
8. free block 与 cached free block 有何区别？
9. retained prefix 为什么不能 hard pin？
10. waiting queue 为什么能作为 near-future reuse signal？
11. read-only probe 为什么不能 touch refcount/LRU/metrics？
12. shared prefix 为什么让 raw block count != reclaimable blocks？
13. 为什么 eviction policy 不应改变 allocation feasibility？
14. 项目与 FastServe 的关系？
15. 项目与 Preble 的关系？
16. 为什么没有完整复刻论文？
17. 单 4090 怎么制造 KV pressure？
18. 为什么无需改 PagedAttention CUDA kernel？

---

# 26. 最终简历叙事模板

真实 benchmark 前不要填数字。

### Scheduler architecture

> 基于 vLLM V1 解耦调度决策与 correctness-critical 状态管理，设计轻量级 scheduling decision policy，为抢占与 cache-aware 策略提供可扩展决策接口，并保持默认 FCFS/Priority 路径兼容。

### Recompute-aware scheduling

> 设计 recomputation-cost-aware preemption/re-admission 策略，在严格用户优先级约束下利用 request 已计算 token 估计恢复代价，减少高重算成本请求被无差别抢占及重复 prefill。

### Prefix cache management

> 实现 waiting-queue-informed prefix cache eviction，利用近期 waiting requests 的 prefix reuse 信号动态调整 cached free block 淘汰顺序，在不 pin KV block、不改变 allocation feasibility 的情况下提高即将调度请求的 prefix 保留概率。

最后真实测量后再追加结果。

---

# 27. Agent 读完本文后必须先生成 IMPLEMENTATION_PLAN.md

**不要直接改代码。**

`IMPLEMENTATION_PLAN.md` 至少包含：

## A. Base commit inspection

- commit hash
- branch
- changed/untracked files
- vLLM version metadata

## B. Current code map

精确列：

```text
class
function
call site
state mutation
```

重点：

- waiting/running queue
- victim selection
- `_preempt_request`
- `num_preemptions`
- allocation failure
- prefix lookup
- cached block eviction

## C. Gap analysis

逐项列：

```text
本文假设
vs
当前 base commit 实际实现
```

## D. Proposed exact patch

精确到：

```text
文件
类
函数
新增接口
参数
返回值
配置
```

先不生成完整代码。

## E. Correctness invariants

说明如何保护：

- priority
- heap
- refcount
- free queue
- prefix hash
- event
- metrics
- fallback

## F. Tests

每个 test 写清：

```text
文件
测试名
initial state
operation
expected state
```

## G. Commit sequence

拆成小 patch。

## H. Estimated code size / risk

如果核心改动明显超过约 600 行，说明原因并提出缩小方案。

用户确认实施方案后，再开始代码修改。

---

# 28. 参考资料

## vLLM Source

- Scheduler  
  https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py
- Request Queue  
  https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/request_queue.py
- Request  
  https://github.com/vllm-project/vllm/blob/main/vllm/v1/request.py
- KV Cache Manager  
  https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py
- Block Pool  
  https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py
- Prefix Caching Design  
  https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md
- Scheduler Tests  
  https://github.com/vllm-project/vllm/blob/main/tests/v1/core/test_scheduler.py
- Prefix Cache Tests  
  https://github.com/vllm-project/vllm/blob/main/tests/v1/core/test_prefix_caching.py

## Relevant vLLM RFC / Issues

- Extensible Scheduler Plugin Framework #51608  
  https://github.com/vllm-project/vllm/issues/51608
- Waiting-Queue-Informed LRU #48485  
  https://github.com/vllm-project/vllm/issues/48485
- Preempted request re-admission bug #41951  
  https://github.com/vllm-project/vllm/issues/41951
- Priority preemption discussion #40004  
  https://github.com/vllm-project/vllm/issues/40004
- Cache-affinity request ordering #42185  
  https://github.com/vllm-project/vllm/issues/42185

## Papers

- FastServe  
  https://arxiv.org/abs/2305.05920
- Preble  
  https://arxiv.org/abs/2407.00023

## Built-in Benchmark

- `vllm bench serve` / `prefix_repetition`  
  https://docs.vllm.ai/en/latest/cli/bench/serve/

---

# 29. 最终四条原则

> **不重新发明 vLLM，只改真实存在的决策缝隙。**

> **不为了科研感做复杂预测，优先复用 vLLM 已有 request / queue / KV metadata。**

> **Scheduler Policy 做 decision，Core 保持 correctness；Waiting Queue 给 future demand，Cache Manager 不越权调度。**

> **项目价值不在代码量，而在三个改动能否形成一条完整、可测试、可解释的推理引擎故事。**
