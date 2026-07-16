# 调度器与 KV Cache 管理深度解析

## 概述

调度器 (`vllm/v1/core/sched/scheduler.py`) 是 vLLM 的"大脑"，负责决定每一步哪些请求参与 GPU 计算。KV Cache 管理器 (`vllm/v1/core/kv_cache_manager.py`) 是调度器的"右手"，负责为每个请求分配和回收 GPU 显存中的 KV cache block。两者协同工作，实现了 vLLM 高吞吐的核心——**连续批处理 (continuous batching)** 和 **PagedAttention**。

---

## 1. 调度器核心数据结构

```python
class Scheduler:
    # 请求池：所有活跃请求
    self.requests: dict[str, Request] = {}

    # 两个队列
    self.waiting: RequestQueue    # 等待调度的请求（新来的 / 被抢占的）
    self.running: list[Request]   # 正在运行的请求

    # 关键约束
    self.max_num_running_reqs      # 最大并发请求数
    self.max_num_scheduled_tokens  # 每步最大 token 预算
    self.max_model_len             # 模型最大序列长度
```

调度器维护两个核心队列：
- **waiting**：优先级队列，按策略排序（FCFS 或其他）
- **running**：当前正在生成的请求列表

---

## 2. `schedule()` 方法 —— 一步调度的完整流程

每一步 EngineCore 调用 `scheduler.schedule()`，返回一个 `SchedulerOutput` 给 ModelRunner 执行。

### 2.1 调度 RUNNING 请求（decode 阶段优先）

```
首先处理已经在运行的请求（它们正在 decode 或还在 chunked prefill）
```

对每个 running 请求：
1. 计算 `num_new_tokens`：本步需要计算的 token 数
   - 对于 decode：通常是 1（或包含 spec decode 的投机 token）
   - 对于 chunked prefill：是本次 chunk 的大小
2. 受 `token_budget` 限制，不超过剩余预算
3. 调用 `kv_cache_manager.allocate_slots(request, num_new_tokens)` 分配新 block
4. 如果分配失败 → **抢占 (preemption)**：从队尾开始踢出低优先级请求，释放其 block

关键代码路径：
```python
# scheduler.py:524
new_blocks = self.kv_cache_manager.allocate_slots(
    request, num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
if new_blocks is None:
    # 尝试抢占其他请求来腾出空间
    while self.running:
        preempted_req = self.running.pop()  # 从队尾开始抢占
        self._preempt_request(preempted_req)
        # 再次尝试分配...
```

### 2.2 调度 WAITING 请求（prefill 新请求）

```
如果没有发生抢占，且 token 预算还有剩余，才调度新请求
```

对每个 waiting 请求：
1. 调用 `kv_cache_manager.get_computed_blocks(request)` —— 查找 prefix cache 命中
2. 计算实际需要计算的 token 数（总 token - 已缓存 token）
3. 受 `token_budget` 和 `max_num_running_reqs` 限制
4. 调用 `kv_cache_manager.allocate_slots(request, num_new_tokens, new_computed_blocks=...)` 
5. 分配失败 → 本轮不再调度新请求（`break`）
6. 分配成功 → 请求进入 running 队列

### 2.3 核心设计哲学

来自源码注释（scheduler.py:393）：
> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has num_computed_tokens and num_tokens_with_spec. At each step, the scheduler tries to assign tokens so that each request's num_computed_tokens can catch up its num_tokens_with_spec.

这意味着 prefill 和 decode 在调度器眼里没有本质区别，都是"还有多少 token 没算完"的问题。这使得 **chunked prefill**（把长 prompt 分多步计算）和 **continuous batching**（每步混合 prefill 和 decode）自然而然地实现。

---

## 3. KV Cache 管理器

### 3.1 架构层次

```
KVCacheManager
  └── coordinator (KVCacheCoordinator / HybridKVCacheCoordinator)
       └── BlockPool
            ├── blocks: list[KVCacheBlock]           # 所有物理 block
            ├── free_block_queue: FreeKVCacheBlockQueue  # 空闲 block 双向链表
            └── cached_block_hash_to_block: BlockHashToBlockMap  # prefix cache 哈希表
```

### 3.2 Block 的生命周期

```
              ┌──────────┐
              │  Free     │  ← 初始状态 / 被 evict 后
              └─────┬────┘
                    │ allocate
                    ▼
              ┌──────────┐
              │ Allocated │  ← 正在被某个请求使用, ref_cnt > 0
              └─────┬────┘
                    │ request 完成 or 被抢占 → free()
                    ▼
              ┌──────────┐
              │  Cached   │  ← ref_cnt == 0, 但保留 hash, 可用于 prefix cache
              └─────┬────┘
                    │ 空间不够时 → evict（从链表头淘汰）
                    ▼
              ┌──────────┐
              │  Free     │
              └──────────┘
```

每个 `KVCacheBlock` 有：
- `block_id`: 物理 block 编号（对应 GPU 显存中的固定区域）
- `ref_cnt`: 引用计数（几个请求在使用这个 block）
- `block_hash`: 内容哈希（用于 prefix caching）

### 3.3 核心操作

#### `get_computed_blocks(request)` — Prefix Cache 查找

当新请求到来时，调度器先问 KV Cache 管理器："这个请求的 prompt 有多少已经缓存了？"

```python
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
    # 用 request.block_hashes（prompt 内容的分块哈希）查找缓存
    computed_blocks, num_computed_tokens = self.coordinator.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )
    return computed_blocks, num_computed_tokens
```

原理：将 prompt token 按 block_size 分块，对每块计算 hash，在 `cached_block_hash_to_block` 中查找最长的连续前缀匹配。

#### `allocate_slots(request, num_new_tokens, ...)` — 分配新 block

```
Blocks 布局：
----------------------------------------------------------------------
| < computed > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                              |            < to be allocated >           |
----------------------------------------------------------------------
```

步骤：
1. `remove_skipped_blocks()` — 释放滑动窗口外的 block（如 sliding window attention）
2. `get_num_blocks_to_allocate()` — 计算需要多少新 block
3. 检查 `free_blocks >= needed + watermark + reserved`，否则返回 None
4. `allocate_new_computed_blocks()` — 将 prefix cache 命中的 block 挂到请求上（增加 ref_cnt）
5. `allocate_new_blocks()` — 从 free queue 分配新 block

#### `free(request)` — 释放请求的 block

当请求完成或被抢占时调用。逆序释放 block（尾部先释放），这样 prefix 部分的 block 最后释放，有更大概率保留在缓存中供后续请求复用。

---

## 4. 调度器与 KV Cache 的交互时序

```
schedule() 被调用
│
├─ 遍历 running 请求
│   ├─ 计算 num_new_tokens
│   ├─ kv_cache_manager.allocate_slots(req, num_new_tokens)
│   │   ├─ 成功 → 加入本步 batch
│   │   └─ 失败 → 抢占低优先级请求
│   │       ├─ _preempt_request() → kv_cache_manager.free(victim)
│   │       └─ 重试 allocate_slots()
│   └─ 记录 req_to_new_blocks[req_id] = new_blocks
│
├─ 遍历 waiting 请求
│   ├─ kv_cache_manager.get_computed_blocks(req) → prefix cache hit
│   ├─ 计算实际 num_new_tokens (总 tokens - cached tokens)
│   ├─ kv_cache_manager.allocate_slots(req, num_new_tokens, new_computed_blocks=...)
│   │   ├─ 成功 → 请求加入 running
│   │   └─ 失败 → break, 本步不再调度新请求
│   └─ 记录 req_to_new_blocks, num_scheduled_tokens
│
├─ 构建 SchedulerOutput
│   ├─ new_reqs_data: 新请求的完整信息（token_ids, block_ids）
│   ├─ cached_reqs_data: 已有请求的增量更新（new block_ids）
│   └─ num_scheduled_tokens: 每个请求本步计算的 token 数
│
└─ _update_after_schedule()
    └─ 推进每个请求的 num_computed_tokens
```

---

## 5. 抢占机制 (Preemption)

当 GPU 显存不够（block 分配失败）时触发抢占：

```python
def _preempt_request(self, request: Request, timestamp: float) -> None:
    # 1. 释放请求的所有 KV cache block
    self._free_request_blocks(request)
    # 2. 释放 encoder cache
    self.encoder_cache_manager.free(request)
    # 3. 重置请求状态
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0  # 下次需要重新 prefill
    # 4. 放回 waiting 队列头部（优先重新调度）
    self.waiting.prepend_request(request)
```

注意：被抢占的请求 `num_computed_tokens` 清零，意味着重新调度时需要完全重新 prefill（除非 prefix cache 还保留着它的 block）。这是 vLLM 选择的权衡——简单且 prefix cache 能部分弥补。

---

## 6. Prefix Caching 工作原理

这是 KV Cache 管理中最精巧的部分：

### 6.1 Block Hash 计算

每个请求的 token 序列被分为 block_size 大小的块，每块计算一个哈希（包含 token 内容 + 前一块的哈希，形成链式哈希）。

### 6.2 缓存查找

新请求到达时，从第一个 block 开始，逐块在哈希表中查找：
- 命中 → 复用已有 block（ref_cnt++），跳过这些 token 的计算
- 未命中 → 后续所有 block 都需要新分配

### 6.3 缓存淘汰

当需要分配新 block 但 free queue 为空时，从 free queue 头部淘汰（LRU 策略）：
- 只淘汰 ref_cnt == 0 的 block（没有活跃请求在用）
- 淘汰时从哈希表移除对应条目

### 6.4 效果

对于有共同 system prompt 的 chat 场景（如 OpenAI 兼容 API），大量请求共享前缀的 KV cache block，显著减少重复计算和显存占用。

---

## 7. 关键设计决策总结

| 设计点 | 选择 | 原因 |
|--------|------|------|
| Prefill/Decode 统一调度 | 不区分阶段，只看"还差几个 token" | 自然支持 chunked prefill 和 continuous batching |
| Block 粒度管理 | 固定大小 block (通常 16 tokens) | PagedAttention 的基础，减少碎片 |
| 抢占策略 | 释放全部 block + 清零 computed tokens | 实现简单，prefix cache 能部分恢复 |
| 缓存淘汰 | LRU (链表头部) | O(1) 淘汰，适合在线服务 |
| Running 优先 | 先调度 running，再调度 waiting | 保证已开始的请求能持续推进 |
| Watermark | 保留一部分空闲 block 不分配给新请求 | 避免频繁抢占的抖动 |

---

## 8. 后续深入方向

- BlockPool 的双向链表和 LRU 具体实现 (`kv_cache_utils.py`)
- KVCacheCoordinator 如何处理多 KV cache group（如 hybrid model）
- Sliding window attention 下 block 的 skip/remove 逻辑
- P/D 分离 (Prefill-Decode Disaggregation) 中 KV 传输与调度的协同
- Speculative Decoding 对 block 分配的影响（lookahead tokens）
