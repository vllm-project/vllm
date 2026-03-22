# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 卸载 Worker 包入口。

本模块导出 worker 模块的核心类和类型，负责：
- 导出卸载处理程序抽象基类 OffloadingHandler
- 导出卸载处理程序管理器 OffloadingWorker
- 导出传输结果数据类 TransferResult
- 导出传输类型别名 TransferSpec 和 TransferType

Worker 模块在 vLLM worker 端运行，负责执行实际的
KV 数据异步传输操作。
"""

from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    OffloadingWorker,
    TransferResult,
    TransferSpec,
    TransferType,
)

__all__ = [
    "OffloadingHandler",
    "OffloadingWorker",
    "TransferResult",
    "TransferSpec",
    "TransferType",
]
