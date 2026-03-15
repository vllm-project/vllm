# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""I64 token-routed expert router."""

from vllm.model_executor.layers.token_routed_i64.router.i64_router import (
    I64Router,
)

__all__ = ["I64Router"]
