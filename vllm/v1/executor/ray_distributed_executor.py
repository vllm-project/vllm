# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.executor.ray_executor import (
    RayDistributedExecutor as _RayDistributedExecutor,
)

# For backwards compatibility.
RayDistributedExecutor = _RayDistributedExecutor
