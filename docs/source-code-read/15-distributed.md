# 第15章：分布式推理

> 一句话：vLLM 支持 TP/PP/DP 三种并行策略，通过 parallel_state 管理进程组通信，并支持 disaggregated prefill（KV Transfer）实现 prefill/decode 分离部署。

## 涉及文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/distributed/parallel_state.py` | ~2350 | 并行状态管理：初始化 TP/PP 进程组、all-reduce/all-gather 封装 |
| `vllm/distributed/kv_events.py` | ~560 | KV 事件系统：disagg prefill 场景下的 KV Cache 传输事件 |
| `vllm/v1/engine/coordinator.py` | ~473 | DP Coordinator：数据并行模式下的负载均衡与同步 |
| `vllm/v1/executor/ray_executor.py` | ~628 | Ray Executor：跨节点分布式执行（多机推理） |
| `vllm/v1/worker/dp_utils.py` | ~225 | DP 工具：数据并行 Worker 间的同步操作 |

## 关键问题（带着这些问题读）

1. Tensor Parallelism 的进程组是如何初始化的？`init_distributed_environment()` 做了什么？
2. TP 中 all-reduce 和 all-gather 分别在模型的哪些位置调用？（ColumnParallel 的 all-gather vs RowParallel 的 all-reduce）
3. Pipeline Parallelism 的微批次调度如何实现？PP rank 之间传递的是什么数据？
4. DP Coordinator 如何做负载均衡？请求是在哪一层被路由到不同的 DP rank？
5. Disaggregated Prefill 中，prefill 实例计算完的 KV Cache 如何通过 NIXL/NVLink 传给 decode 实例？

## 调用链概览

```
TP 场景 (以 RowParallelLinear 为例):
  input → 每个 TP rank 拿自己的分片权重
    → matmul(input, weight_shard)
    → all_reduce(partial_output) across TP group
    → 每个 rank 得到相同的完整输出

DP 场景:
  API Server
    → DP Coordinator.route(request) → 选择负载最低的 DP rank
    → EngineCore[rank_i] 处理该请求
    → 结果返回给 API Server

Disagg Prefill:
  Prefill Instance:
    → 执行 prefill → 计算 KV Cache
    → KV Events → NIXL/NVLink 传输 KV blocks
  Decode Instance:
    → 接收 KV Cache blocks
    → 直接进入 decode 阶段
```

## 官方文档参考

- `docs/design/multiprocessing.md` — Python 多进程方法选择
- `docs/features/disagg_prefill.md` — Disaggregated Prefill 使用指南
- `docs/serving/data_parallel_deployment.md` — 数据并行部署
- `docs/serving/parallelism_scaling.md` — 并行扩展策略

## 详细笔记

> （实际阅读后填充）
