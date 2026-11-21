# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shapes the gfx942 large-step MoE kernels are built for: DeepSeek-V4.1's
routing and hidden width. `largestep_gate.build` checks a layer against them
before it gets the block."""

GROUP = 16  # picks a group: one MFMA's N
HIDDEN = 5120
LIMIT = 10.0  # DeepSeek-V4.1's swiglu_limit
EXPERTS, TOPK = 384, 6  # the target model's routing
