# 第12章：采样与输出处理

> 一句话：Sampler 将模型 logits 转换为 token 选择，StructuredOutput 约束输出符合 JSON Schema 等格式，Detokenizer 将 token 序列增量还原为文本。

## 涉及文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/v1/sample/sampler.py` | ~436 | Sampler：temperature scaling → top-p/top-k 过滤 → 采样/greedy |
| `vllm/v1/sample/metadata.py` | ~55 | SamplingMetadata：批次采样参数的打包格式 |
| `vllm/v1/structured_output/__init__.py` | ~490 | StructuredOutputManager：管理结构化输出后端的调度 |
| `vllm/v1/structured_output/utils.py` | ~561 | 结构化输出工具：JSON Schema 解析、grammar 构建 |
| `vllm/logits_process.py` | ~200+ | LogitsProcessor：repetition_penalty、presence_penalty 等 logits 修改器 |
| `vllm/v1/engine/detokenizer.py` | ~362 | Detokenizer：增量 detokenize 支持 streaming |

## 关键问题（带着这些问题读）

1. Sampler 的执行顺序是什么？logits 从模型输出到最终 token 经历了哪些变换？
2. 结构化输出（guided decoding）如何在每步采样前修改 logits？xgrammar / outlines / guidance 三个后端的区别？
3. Detokenizer 如何实现增量 detokenize？为什么不能简单地对每个新 token 单独 decode？
4. LogitsProcessor（repetition_penalty 等）在采样流水线中的位置在哪？

## 调用链概览

```
GPUModelRunner.execute_model():
  → model(input_ids, ...) → hidden_states
  → lm_head(hidden_states) → logits
  → LogitsProcessor.apply(logits)        # 各种 penalty
  → StructuredOutput.apply(logits)        # 语法约束 mask
  → Sampler.forward(logits, metadata)     # temperature → top-p → sample
  → 返回 token_ids, logprobs

Engine 侧:
  → OutputProcessor: token_ids → RequestOutput
    → Detokenizer: token_ids → 增量文本
```

## 官方文档参考

- `docs/features/structured_outputs.md` — 结构化输出的使用方法
- `docs/design/logits_processors.md` — Logits 处理器设计

## 详细笔记

> （实际阅读后填充）
