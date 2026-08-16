# 第6章：KV Cache 管理

> 一句话：KV Cache 管理是 vLLM 的核心创新（PagedAttention），本章覆盖 Block 分配/回收、前缀缓存的哈希匹配、LRU 淘汰、以及多层 KV Cache 类型支持。

## 涉及文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/v1/core/kv_cache_utils.py` | ~2335 | KV Cache 工具函数：哈希计算、Block 管理数据结构（KVCacheBlock） |
| `vllm/v1/core/kv_cache_manager.py` | ~885 | KVCacheManager：顶层管理器，协调多个 SingleTypeKVCacheManager |
| `vllm/v1/core/single_type_kv_cache_manager.py` | ~1942 | 单类型 KV Cache 管理：allocate / free / get_computed_blocks |
| `vllm/v1/core/block_pool.py` | ~830 | BlockPool：Block 对象池 + 双向链表 Free Queue |
| `vllm/v1/core/kv_cache_coordinator.py` | ~903 | KV Cache 协调器：混合 KV Cache 类型（如 Hybrid Model）的协调 |

## 关键问题（带着这些问题读）

1. KVCacheBlock 的哈希是如何计算的？parent hash + block tokens + extra hash 的链式结构如何保证唯一性？
2. 前缀缓存命中时，`get_computed_blocks()` 的查找路径是什么？命中后如何"touch"避免被淘汰？
3. Free Queue 的双向链表设计为什么比 Python deque 更优？O(1) 移动中间元素的场景是什么？
4. Block 淘汰（eviction）时，为什么 freed blocks 按倒序加入 free queue？
5. Hybrid 模型（如 Jamba，混合 Attention + Mamba）的 KV Cache 如何协调？

## 调用链概览

```
新请求到达:
  Scheduler.schedule()
    → KVCacheManager.get_computed_blocks(request)
      → 哈希匹配 → 返回已计算的 block 列表
    → KVCacheManager.allocate_slots(request, num_new_tokens)
      → touch computed blocks (增加 ref_cnt, 从 free queue 移除)
      → pop free queue head → 如果是 cached block 则 evict
      → 如果 block 写满 → 加入 cache_blocks 哈希表

请求完成:
  KVCacheManager.free(request)
    → 减少 ref_cnt → ref_cnt=0 的 block 加入 free queue 尾部（倒序）
```

## 官方文档参考

- `docs/design/prefix_caching.md` — 前缀缓存的完整设计文档，包含数据结构、操作流程和端到端示例
- `docs/design/paged_attention.md` — PagedAttention 原始论文的内核实现说明（历史文档）
- `docs/design/hybrid_kv_cache_manager.md` — 混合 KV Cache 管理器设计

## 详细笔记

> （实际阅读后填充）
