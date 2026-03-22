# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ray 分布式执行器模块（向后兼容）。

本模块提供向后兼容的 RayDistributedExecutor 导入。
实际实现位于 ray_executor.py 中。
"""

from vllm.v1.executor.ray_executor import (
    RayDistributedExecutor as _RayDistributedExecutor,
)

# 向后兼容
RayDistributedExecutor = _RayDistributedExecutor
