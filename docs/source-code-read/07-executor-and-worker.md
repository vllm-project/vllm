# 第7章：Executor 与 Worker

> 一句话：Executor 负责管理 GPU Worker 进程的生命周期和通信，Worker 负责在单个 GPU 上加载模型和执行推理，两者构成 V1 的执行层。

## 涉及文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/v1/executor/abstract.py` | ~380 | Executor 抽象基类：定义 execute_model / initialize 等接口 |
| `vllm/v1/executor/multiproc_executor.py` | ~1101 | MultiprocExecutor：多进程执行器，spawn Worker 子进程 |
| `vllm/v1/executor/uniproc_executor.py` | ~196 | UniprocExecutor：单进程执行器，Worker 在主进程内 |
| `vllm/v1/worker/gpu_worker.py` | ~1399 | GPUWorker：单 GPU 工作者，模型加载、内存分析、KV Cache 初始化 |
| `vllm/v1/worker/worker_base.py` | ~358 | WorkerBase：Worker 抽象接口定义 |

## 关键问题（带着这些问题读）

1. Engine Core 如何选择使用 MultiprocExecutor 还是 UniprocExecutor？选择依据是什么？
2. MultiprocExecutor 如何启动 Worker 子进程？数据（权重、KV Cache）是如何在进程间共享的？
3. GPUWorker 的初始化顺序是什么？`init_device` → `load_model` → `determine_available_memory` → `initialize_cache` 的流程？
4. Worker 的 `execute_model()` 接收什么输入、返回什么输出？输入输出是如何跨进程传递的？
5. Ray Executor 与 MultiprocExecutor 的区别在哪？什么场景必须用 Ray？

## 调用链概览

```
EngineCore 初始化:
  → Executor.create(vllm_config)
    → MultiprocExecutor.__init__()
      → spawn N 个 Worker 子进程
      → Worker.init_device() → 绑定 GPU
      → Worker.load_model() → 加载权重
      → Worker.determine_available_memory() → 探测可用显存
      → Worker.initialize_cache() → 分配 KV Cache

每步推理:
  EngineCore → Executor.execute_model(scheduler_output)
    → 广播输入给所有 Worker
    → Worker.execute_model() → ModelRunner.execute_model()
    → 收集输出 → 返回给 EngineCore
```

## 官方文档参考

- `docs/design/arch_overview.md` — Worker、GPU Worker Processes 一节
- `docs/design/multiprocessing.md` — Python 多进程方法选择的权衡

## 详细笔记

> （实际阅读后填充）
