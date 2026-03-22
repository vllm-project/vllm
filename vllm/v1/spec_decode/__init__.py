# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""推测解码模块。

本模块为 vLLM V1 引擎提供推测解码（Speculative Decoding）功能，负责：
- 使用小型草稿模型或无模型方法生成候选 token
- 通过大型目标模型验证候选 token
- 加速推理过程，降低延迟

主要 proposer 实现：
- EagleProposer: 基于 EAGLE 架构的推测解码
- MedusaProposer: 基于 Medusa 多头的推测解码
- NgramProposer: 基于 n-gram 匹配的无模型推测
- SuffixDecodingProposer: 基于后缀树的推测解码
- DraftModelProposer: 通用草稿模型 proposer
- ExtractHiddenStatesProposer: 提取隐藏状态用于 KV 缓存

工作流程：
1. propose: 草稿模型生成候选 token
2. verify: 目标模型并行验证所有候选 token
3. accept: 根据验证结果接受或拒绝 token
"""
