# 第11章：编译与 IR 系统

> 一句话：vLLM 深度集成 torch.compile，通过自定义的 IR 中间表示将算子语义与实现分离，并在编译流水线中执行 Fusion Pass 优化，最终生成高效的 Triton/CUDA 内核。

## 涉及文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/compilation/backends.py` | ~1339 | 编译后端：Inductor 集成、缓存管理、Piecewise 编译 |
| `vllm/compilation/compiler_interface.py` | ~810 | 编译器接口：VllmBackend、PassConfig、编译流水线入口 |
| `vllm/compilation/decorators.py` | ~780 | @support_torch_compile 装饰器：标记可编译的模型 forward |
| `vllm/compilation/cuda_graph.py` | ~361 | CUDAGraphWrapper：CUDAGraph 捕获与回放封装 |
| `vllm/config/compilation.py` | ~1570 | CompilationConfig：编译级别、cudagraph_mode、fusion 开关 |
| `vllm/ir/` | ~1000+ | vLLM IR：@register_op、register_impl、IrOp 分发机制 |

## 关键问题（带着这些问题读）

1. 编译流水线的 6 个阶段（Dynamo Tracing → AOTAutograd → IR Fusion → IR Lowering → Clone Cleanup → Inductor）分别做什么？
2. vLLM IR 的 `@register_op` 如何将算子语义与平台实现分离？`dispatch()` 如何按优先级选择实现？
3. `maybe_inplace` 的设计目的是什么？它如何在编译时保证内存安全（clone 插入与消除）？
4. Piecewise 编译如何将 attention op 作为 splitting_ops 切分计算图？这对 CUDAGraph 有什么影响？
5. 编译缓存的 hash key 包含哪些因素？如何保证缓存安全性？

## 调用链概览

```
编译流水线:
  model.forward (被 @support_torch_compile 装饰)
    → torch.compile(model.forward, backend=VllmBackend)
      1. [Dynamo] → FX Graph (含 vllm_ir.* ops)
      2. [Pre-grad] Inplace Functionalization → maybe_inplace → default
      3. [AOTAutograd] → 函数化
      4. [Post-grad] IR Fusion Passes → 算子融合 (e.g., AllReduce+RMSNorm)
      5. [Post-grad] IR Lowering → vllm_ir.* → 具体实现 (e.g., torch.ops._C.*)
      6. [Post-grad] Clone Cleanup → 消除冗余 clone
      7. [Inductor] → Triton/C++ codegen → 编译产物
```

## 官方文档参考

- `docs/design/torch_compile.md` — torch.compile 集成的完整走读（缓存、动态形状、图编译）
- `docs/design/vllm_ir.md` — vLLM IR 设计文档（算子声明、实现注册、编译流水线）
- `docs/design/fusions.md` — 各种 Fusion Pass 的功能、配置和性能收益
- `docs/design/cuda_graphs.md` — CUDAGraph 模式与 Piecewise 编译的交互

## 详细笔记

> （实际阅读后填充）
