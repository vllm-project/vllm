# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""注意力操作模块。

本模块包含各种注意力相关的底层操作实现，包括：
- Triton kernel 实现（解码注意力、预填充注意力、统一注意力等）
- Flash Attention 封装
- KV 缓存操作
- 注意力状态合并
- 上下文并行操作
- DCP All-to-All 通信
- FlashMLA 操作
- ViT 注意力封装
- XPU 稀疏 MLA 操作
"""