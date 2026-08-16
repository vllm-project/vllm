# 第14章：多模态支持

> 一句话：vLLM 通过 MultiModalRegistry 和 BaseMultiModalProcessor 统一处理图片/视频/音频输入，将多模态数据转换为 placeholder tokens + encoder embeddings 注入推理流水线。

## 涉及文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/multimodal/registry.py` | ~378 | MultiModalRegistry：注册和查找各模态的处理器 |
| `vllm/multimodal/inputs.py` | ~1061 | 多模态输入数据结构：MultiModalKwargs、PlaceholderRange |
| `vllm/multimodal/parse.py` | ~830 | 解析 HF processor 输出，提取 placeholder 与数据的映射 |
| `vllm/multimodal/video.py` | ~2116 | 视频处理：帧提取、resize、归一化 |
| `vllm/v1/core/encoder_cache_manager.py` | ~385 | Encoder Cache：缓存 vision encoder 的输出避免重复计算 |

## 关键问题（带着这些问题读）

1. 一张图片从用户请求到进入模型前向，经历了哪些处理步骤？（HTTP → PIL → processor → embedding）
2. Placeholder tokens 如何与真实的图片 embedding 对应？`PlaceholderRange` 的作用？
3. HF Processor 的输出缓存机制如何工作？为什么 Qwen2-VL 的 processor 很慢需要缓存？
4. Encoder Cache Manager 缓存了什么？它的淘汰策略是什么？
5. 融合设备归一化（`mm_device_do_normalize`）如何将 uint8 → GPU 归一化，减少 PCIe 带宽？

## 调用链概览

```
用户请求 (含 image_url):
  → API Server: 下载图片 → PIL.Image
  → InputProcessor:
    → MultiModalProcessor.apply(prompt, images)
      → HF Processor: tokenize + resize + normalize
      → 检测 prompt 中的 placeholder 位置
      → 返回 token_ids + MultiModalKwargs(pixel_values=...)

  → EngineCore → Worker:
    → VisionEncoder.forward(pixel_values) → image_embeds
    → 将 image_embeds 替换 placeholder 位置的 embedding
    → LLM.forward(combined_embeddings)
```

## 官方文档参考

- `docs/design/mm_processing.md` — 多模态数据处理的设计文档
- `docs/features/multimodal_inputs.md` — 多模态输入使用指南

## 详细笔记

> （实际阅读后填充）
