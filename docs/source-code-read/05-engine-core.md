# 第5章：Engine Core 调度核心

> 一句话：Engine Core 是 V1 架构的心脏，运行一个 busy loop 持续调度请求、管理 KV Cache、协调 GPU Worker 执行，本章深入其主循环的每一步。

## 涉及文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/v1/engine/core.py` | ~2479 | EngineCore：主循环、调度、与 Worker 的交互 |
| `vllm/v1/engine/core_client.py` | ~1874 | EngineCoreClient：API Server 与 EngineCore 的 ZMQ 通信 |
| `vllm/v1/engine/async_llm.py` | ~1154 | AsyncLLM：异步引擎包装，驱动 EngineCore 并处理输出流 |
| `vllm/v1/engine/output_processor.py` | ~836 | OutputProcessor：将模型输出转换为 RequestOutput |
| `vllm/v1/engine/input_processor.py` | ~500 | InputProcessor：请求预处理（tokenize、校验） |
| `vllm/v1/engine/detokenizer.py` | ~362 | Detokenizer：增量 detokenize，支持流式输出 |

## 关键问题（带着这些问题读）

1. EngineCore 的主循环（`run_busy_loop`）每一轮做了哪些事？调度 → 执行 → 输出的具体步骤？
2. Scheduler 的调度策略是什么？如何决定哪些请求进入 prefill、哪些继续 decode？
3. API Server 进程与 Engine Core 进程之间通过 ZMQ 传递了哪些消息类型？序列化方式？
4. Engine Core 如何处理请求取消（abort）和超时？
5. 多 Engine Core（DP > 1）场景下，Coordinator 如何分配请求？

## 调用链概览

```
EngineCore.run_busy_loop():
  while True:
    1. 从 ZMQ socket 接收新请求 / 取消信号
    2. Scheduler.schedule() → 决定本步要处理的请求集合
       → 分配 KV Cache slots (调用 KVCacheManager)
    3. Executor.execute_model(scheduled_requests)
       → Worker.execute_model() → GPU forward pass
    4. 处理模型输出 → 更新请求状态
    5. 通过 ZMQ socket 发送输出给 API Server
```

## 官方文档参考

- `docs/design/arch_overview.md` — V1 Process Architecture、Engine Core Process
- `docs/design/arch_overview.md` — LLM Engine 一节

## 详细笔记

> （实际阅读后填充）
