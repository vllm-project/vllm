# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM V1 核心调度模块。

本模块实现 vLLM V1 的核心调度功能，负责：
- 管理请求的生命周期（等待、运行、完成）
- 分配 KV 缓存块和编码器缓存
- 处理推测解码和前缀缓存
- 支持多模态输入和 LoRA 适配器
"""