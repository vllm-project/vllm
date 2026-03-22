# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 数据卸载模块。

本模块实现 vLLM V1 中的 KV 缓存数据卸载功能，负责：
- 将 KV 缓存数据从 GPU 卸载到 CPU 或其他存储介质
- 管理卸载数据的生命周期和 LRU 策略
- 提供透明的数据加载/存储接口

主要组件：
- OffloadingManager: 卸载管理器抽象基类
- 具体后端实现（CPU、磁盘等）
"""
