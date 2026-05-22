# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sonic-MoE fused-experts backend (quack grouped-GEMM + SwiGLU)."""

from .sonic_moe_experts import SonicMoEExperts

__all__ = ["SonicMoEExperts"]
