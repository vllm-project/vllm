# 第8章：GPU Model Runner

> 一句话：GPUModelRunner 是模型前向执行的直接驱动者，负责将调度输出转换为 GPU 张量、管理 CUDAGraph 捕获/回放、以及协调 attention 元数据。

## 涉及文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/v1/worker/gpu_model_runner.py` | ~7967 | 核心文件：输入准备、模型执行、CUDAGraph 管理、warmup |
| `vllm/v1/worker/gpu_input_batch.py` | ~1155 | InputBatch：维护当前批次的 token IDs、位置、采样参数等 |
| `vllm/v1/worker/block_table.py` | ~442 | BlockTable：管理请求到物理 KV Cache block 的映射表 |
| `vllm/v1/cudagraph_dispatcher.py` | ~200+ | CudagraphDispatcher：根据批次特征选择 FULL/PIECEWISE/NONE 模式 |
| `vllm/forward_context.py` | ~200+ | ForwardContext：前向传播的全局上下文（attention metadata、CG 模式） |

## 关键问题（带着这些问题读）

1. `execute_model()` 的完整流程？从 SchedulerOutput 到 SamplerOutput 经历了哪些步骤？
2. 输入张量（input_ids、positions、block_table）是如何从 InputBatch 准备的？prefill 和 decode 有何不同？
3. CUDAGraph 的捕获时机是什么？`_dummy_run()` 如何模拟真实推理来捕获图？
4. CudagraphDispatcher 如何根据 BatchDescriptor（num_tokens、uniform）选择运行模式？
5. 为什么这个文件有近 8000 行？它的主要职责边界在哪？

## 调用链概览

```
Worker.execute_model(scheduler_output):
  → GPUModelRunner.execute_model(scheduler_output)
    1. _update_states(scheduler_output)  # 更新 InputBatch
    2. _prepare_inputs()                 # 构建 input_ids, positions, attn_metadata
    3. CudagraphDispatcher.dispatch()    # 决定 CG 运行模式
    4. set_forward_context(...)          # 设置全局上下文
    5. model(input_ids, positions, ...)  # 执行模型前向
       → CUDAGraphWrapper: 捕获或回放
    6. Sampler(hidden_states)            # 采样下一个 token
    → 返回 ModelRunnerOutput
```

## 官方文档参考

- `docs/design/cuda_graphs.md` — CUDAGraph 模式详解、CudagraphDispatcher 设计
- `docs/design/arch_overview.md` — Model Runner 一节

## 详细笔记

> （实际阅读后填充）
