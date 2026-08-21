# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ops shared across model implementations."""

from .fused_qk_rmsnorm import _FUSED_Q_KV_RMSNORM_KERNEL

__all__ = [
    "_FUSED_Q_KV_RMSNORM_KERNEL",
]
