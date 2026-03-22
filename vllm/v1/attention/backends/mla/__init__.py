# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA（多头潜在注意力）后端模块。

本模块包含各种 MLA 后端的实现，包括：
- FlashAttn MLA：基于 Flash Attention 的 MLA 实现
- FlashInfer MLA：基于 FlashInfer 的 MLA 实现
- FlashMLA：DeepSeek 优化的 MLA 实现
- Cutlass MLA：基于 CUTLASS 的 MLA 实现（SM100）
- Aiter MLA：ROCm Aiter 后端
- Triton MLA：基于 Triton kernel 的 MLA 实现
- 稀疏 MLA：支持 DeepSeek-V3.2 等模型的稀疏注意力
"""