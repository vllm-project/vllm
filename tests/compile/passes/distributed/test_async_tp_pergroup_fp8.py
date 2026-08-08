# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AsyncTP fusion: dynamic per-group (block-wise) FP8 scaled_mm + comms.

WIP. Tracks the `scaled_mm+comms` row, dynamic per-group FP8 variant in the
central fusion tracker (#36066), currently marked not supported across
sm100/sm90/ROCm.

Existing AsyncTP FP8 coverage:
- per-tensor (scalar) FP8 via the FlashInfer bmm_fp8 path (#39505)
- per-token / row-wise FP8 via ScaledMMReduceScatterPattern /
  AllGatherScaledMMPattern (scale_a=[M,1], scale_b=[1,N])

Gap targeted here: block-wise scales (scale_a=[M, K/group_size],
scale_b=[N/group_size, K/group_size], e.g. group_size=128). The fused
collective path is scalar-only today: the symm-mem adapter asserts
A_scale.numel()==1 and B_scale.numel()==1, and no registered pattern matches
block-wise scale shapes.

This module will hold the correctness check (fused block-wise scaled_mm +
reduce_scatter / all_gather vs the unfused quant -> block scaled_mm ->
reduce_scatter reference) once the pattern + fused path land.
"""

import pytest

pytest.skip(
    "WIP: per-group FP8 scaled_mm+comms AsyncTP fusion not implemented yet "
    "(see #36066).",
    allow_module_level=True,
)