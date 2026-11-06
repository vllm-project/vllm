# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for late-interaction (MaxSim) scoring."""

from .flash_maxsim_rerank import flash_maxsim_rerank_direct

__all__ = ["flash_maxsim_rerank_direct"]
