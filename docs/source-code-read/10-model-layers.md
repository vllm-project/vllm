# 第10章：模型构建层

> 一句话：vLLM 对 nn.Module 做了深度定制——Linear 层内建 TP 分片与量化权重加载、FusedMoE 层封装专家路由与矩阵乘、RMSNorm/RoPE 等常用组件构成模型的乐高积木。

## 涉及文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/model_executor/layers/linear.py` | ~1669 | 列并行/行并行 Linear：内建 TP 分片、权重加载、量化适配 |
| `vllm/model_executor/layers/vocab_parallel_embedding.py` | ~581 | 词表并行 Embedding：将词表按 TP 维度分片 |
| `vllm/model_executor/layers/fused_moe/layer.py` | ~446 | FusedMoE 顶层接口：调度 expert 路由 + 矩阵乘的融合实现 |
| `vllm/model_executor/layers/fused_moe/modular_kernel.py` | ~1713 | 模块化 MoE 内核：Prepare → Expert → Finalize 三段式框架 |
| `vllm/model_executor/layers/layernorm.py` | ~325 | RMSNorm：通过 vLLM IR 调度高性能内核 |
| `vllm/model_executor/layers/rotary_embedding/base.py` | ~324 | RoPE 基类：旋转位置编码及其各种变体的基础 |

## 关键问题（带着这些问题读）

1. ColumnParallelLinear / RowParallelLinear 如何在初始化时就对权重做分片？`weight_loader` 回调的作用？
2. 量化（FP8、AWQ、GPTQ）如何与 Linear 层结合？QuantizationConfig 在哪个时机介入？
3. FusedMoE 的三段式框架（Prepare → Expert → Finalize）如何解耦 All2All 通信与计算？
4. RMSNorm 调用 `ir.ops.rms_norm()` 是什么？它与直接调用 `torch.ops._C.rms_norm` 有什么区别？
5. 模型定义（如 `models/llama.py`）如何组合这些层？`vllm_config` 和 `prefix` 参数的作用？

## 调用链概览

```
以 Llama 为例:
  LlamaForCausalLM.__init__(vllm_config, prefix)
    → LlamaModel → [LlamaDecoderLayer × N]
      → LlamaAttention:
          → QKVParallelLinear (ColumnParallel)  # Q/K/V 投影
          → RowParallelLinear                    # O 投影
          → RotaryEmbedding                      # RoPE
          → Attention(backend)                   # 注意力计算
      → LlamaMLP:
          → MergedColumnParallelLinear           # gate + up
          → SiluAndMul                           # 激活
          → RowParallelLinear                    # down
      → RMSNorm                                  # 层归一化
```

## 官方文档参考

- `docs/design/fused_moe_modular_kernel.md` — FusedMoE 模块化内核的设计文档
- `docs/design/arch_overview.md` — Class Hierarchy 中关于权重分片和量化的设计选择

## 详细笔记

> （实际阅读后填充）
