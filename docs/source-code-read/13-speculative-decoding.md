# 第13章：投机解码

> 一句话：投机解码用小模型（draft）快速提议多个候选 token，再由大模型（target）一次验证，以更少的前向次数换取更高吞吐，vLLM 支持 N-gram、Eagle、MTP 等多种策略。

## 涉及文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/v1/spec_decode/llm_base_proposer.py` | ~1887 | LLM 基础提议器：使用独立小模型作为 draft model |
| `vllm/v1/sample/rejection_sampler.py` | ~953 | 拒绝采样器：验证 draft tokens 的接受/拒绝逻辑 |
| `vllm/v1/spec_decode/ngram_proposer.py` | ~293 | N-gram 提议器：基于历史 token 统计的无模型提议策略 |
| `vllm/v1/spec_decode/utils.py` | ~601 | 投机解码工具函数：token 对齐、概率校正 |
| `vllm/config/speculative.py` | ~1492 | SpeculativeConfig：draft 模型路径、num_speculative_tokens 等配置 |

## 关键问题（带着这些问题读）

1. 投机解码的核心流程？Draft 提议 → Target 验证 → 接受/拒绝 → 回退的完整路径？
2. Rejection Sampling 的数学原理？acceptance probability = min(1, p_target / p_draft) 如何保证正确性？
3. N-gram Proposer 与 LLM Proposer 的性能/质量权衡？什么场景适合用哪种？
4. Eagle / MTP（Multi-Token Prediction）与普通 draft model 的区别？它们如何复用 target 模型的隐状态？
5. 投机解码如何与 KV Cache 管理交互？被拒绝的 draft tokens 的 KV Cache 如何回收？

## 调用链概览

```
EngineCore.step() (启用 spec decode):
  1. Proposer.propose(request_batch)
     → draft_model.forward() × num_spec_tokens 步
     → 返回 draft_token_ids + draft_probs

  2. Target.verify(input_ids + draft_tokens)
     → target_model.forward() (一次前向处理所有 draft tokens)
     → 返回 target_probs

  3. RejectionSampler.forward(target_probs, draft_probs)
     → 逐位置比较 → 接受或拒绝
     → 第一个拒绝位置重新从 target 分布采样
     → 返回最终 accepted_tokens
```

## 官方文档参考

- `docs/features/speculative_decoding/` — 各种投机解码策略的使用指南
- `docs/features/speculative_decoding/eagle.md` — Eagle 策略详解
- `docs/features/speculative_decoding/mtp.md` — MTP 策略详解

## 详细笔记

> （实际阅读后填充）
