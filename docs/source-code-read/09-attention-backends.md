# 第9章：Attention 后端

> 一句话：vLLM 支持 FlashAttention、FlashInfer、FlashMLA 等多种 attention 后端，通过统一的 AttentionBackend 接口实现自动选择和 CUDAGraph 兼容性检查。

## 涉及文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/v1/attention/backend.py` | ~1122 | AttentionBackend 抽象基类：定义 build/forward 接口、CG 兼容性声明 |
| `vllm/v1/attention/selector.py` | ~230 | 后端选择器：按优先级自动选择兼容的 attention 后端 |
| `vllm/model_executor/layers/attention/attention.py` | ~808 | Attention 层：模型代码中调用的统一 Attention 接口 |
| `vllm/model_executor/layers/attention/mla_attention.py` | ~2894 | MLA Attention（DeepSeek）：Multi-head Latent Attention 实现 |
| `vllm/config/attention.py` | ~177 | AttentionConfig：后端选择、flash_attn_version 等配置 |

## 关键问题（带着这些问题读）

1. 后端自动选择的优先级规则是什么？Standard Attention 和 MLA Attention 的优先级表有何不同？
2. AttentionBackend 需要实现哪些核心方法？prefill 和 decode 是否使用不同的后端？
3. `validate_configuration()` 校验了哪些条件？（dtype、head_size、compute capability 等）
4. MLA Attention 与标准 MHA 在 KV Cache 布局上有什么区别？为什么 MLA 需要单独的后端？
5. AttentionCGSupport 枚举（ALWAYS/UNIFORM_BATCH/NEVER）如何影响 CUDAGraph 模式降级？

## 调用链概览

```
初始化时:
  GPUModelRunner._initialize_attn_backend()
    → AttentionSelector.select(model_config, cache_config, ...)
      → 按优先级遍历后端 → validate_configuration()
      → 返回第一个兼容的后端类

每步推理时:
  model forward → Attention.forward(q, k, v)
    → backend.forward(q, k, v, kv_cache, attn_metadata)
      → FlashAttention / FlashInfer / FlashMLA 内核
```

## 官方文档参考

- `docs/design/attention_backends.md` — 后端优先级表、功能支持矩阵
- `docs/design/paged_attention.md` — PagedAttention CUDA 内核的底层实现原理

## 详细笔记

> （实际阅读后填充）
