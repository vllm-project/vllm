# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""池化（Pooling）模块入口。

本模块是 vLLM V1 池化功能的入口文件，提供池化相关的
数据结构和工具函数。

主要功能：
- PoolingCursor: 池化操作游标，定位首批和末尾 token
- PoolingStates: 池化状态容器，缓存隐藏状态
- PoolingMetadata: 池化元数据容器

池化操作用于生成式任务之外的场景，如：
- 文本嵌入（Embedding）
- 交叉注意力（Cross Attention）
- 重排序（Reranking）
"""
