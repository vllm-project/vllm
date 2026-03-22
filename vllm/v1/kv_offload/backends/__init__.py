# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 卸载后端包入口。

本模块导出后端模块的核心类，负责：
- 导出后端抽象基类 Backend
- 导出 CPU 后端实现 CPUBackend

后端负责管理 KV 数据块的存储空间分配和释放，
并提供后端特定的加载/存储规范生成方法。
"""

from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.backend import Backend

__all__ = [
    "Backend",
    "CPUBackend",
]
