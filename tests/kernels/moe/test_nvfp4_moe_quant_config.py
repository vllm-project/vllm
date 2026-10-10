# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for `make_nvfp4_moe_quant_config`.

The function builds the activation/weight scale tensors that NVFP4 MoE
kernels consume at inference time. The two scale tensors `a13_scale` and
`a2_scale` may legitimately contain 0.0 entries when an expert was never
seen during quantization calibration ("dead experts" — common on very
large MoE checkpoints such as Qwen3.5-122B-A10B-NVFP4). Without a clamp
on the input, `1.0 / 0.0 == inf` propagates as NaN through the FP4
quantization kernel and corrupts every subsequent token.

See https://github.com/vllm-project/vllm/pull/42601 for the fix.
"""

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    make_nvfp4_moe_quant_config,
)

# Backends whose default path runs `1.0 / a_scale` over the per-expert
# scale tensor. MARLIN (W4A16) and EMULATION take separate branches and
# are not affected.
_BUGGY_PATH_BACKENDS = [
    NvFp4MoeBackend.VLLM_CUTLASS,
    NvFp4MoeBackend.FLASHINFER_CUTLASS,
    NvFp4MoeBackend.FLASHINFER_TRTLLM,
    NvFp4MoeBackend.FLASHINFER_CUTEDSL,
    NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED,
]


def _placeholder_weight_scales(num_experts: int):
    """Minimal-shape weight scale tensors. The function under test does
    not inspect their contents in the buggy path; only `a13_scale` and
    `a2_scale` go through the inversion."""
    return (
        torch.ones((num_experts, 1, 1), dtype=torch.uint8),  # w13_scale
        torch.ones((num_experts, 1, 1), dtype=torch.uint8),  # w2_scale
        torch.ones((num_experts,), dtype=torch.float32),  # w13_scale_2
        torch.ones((num_experts,), dtype=torch.float32),  # w2_scale_2
    )


@pytest.mark.parametrize("backend", _BUGGY_PATH_BACKENDS)
def test_make_nvfp4_moe_quant_config_clamps_dead_expert_scales(
    backend: NvFp4MoeBackend,
) -> None:
    """Dead-expert scales (== 0.0) must not yield inf/NaN in the returned
    global scales. The clamp uses `torch.finfo(float32).tiny` (~1.18e-38),
    so the largest possible inverse stays finite at ~8.5e+37."""
    num_experts = 8
    # Two dead experts in a13 (idx 0, 3) and two in a2 (idx 2, 7).
    a13_scale = torch.tensor(
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        dtype=torch.float32,
    )
    a2_scale = torch.tensor(
        [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        dtype=torch.float32,
    )
    w13_scale, w2_scale, w13_scale_2, w2_scale_2 = _placeholder_weight_scales(
        num_experts
    )

    config = make_nvfp4_moe_quant_config(
        backend=backend,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_scale_2=w13_scale_2,
        w2_scale_2=w2_scale_2,
        a13_scale=a13_scale,
        a2_scale=a2_scale,
    )

    assert torch.isfinite(config.a1_gscale).all(), (
        f"[{backend.name}] a1_gscale must be finite for dead-expert scale=0.0; "
        f"got {config.a1_gscale}"
    )
    assert torch.isfinite(config.a2_gscale).all(), (
        f"[{backend.name}] a2_gscale must be finite for dead-expert scale=0.0; "
        f"got {config.a2_gscale}"
    )
    max_inv = 1.0 / torch.finfo(torch.float32).tiny
    assert (config.a1_gscale <= max_inv).all()
    assert (config.a2_gscale <= max_inv).all()


def test_make_nvfp4_moe_quant_config_healthy_scales_unchanged() -> None:
    """Sanity check: when all input scales are healthy (no zeros), the
    clamp is a no-op and the returned inverse exactly matches the simple
    1/scale baseline. Prevents a future maintainer from widening the
    clamp to a value that silently shifts realistic activation scales."""
    num_experts = 4
    a13_scale = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32)
    a2_scale = torch.tensor([1.0, 2.0, 0.5, 4.0], dtype=torch.float32)
    w13_scale, w2_scale, w13_scale_2, w2_scale_2 = _placeholder_weight_scales(
        num_experts
    )

    config = make_nvfp4_moe_quant_config(
        backend=NvFp4MoeBackend.VLLM_CUTLASS,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_scale_2=w13_scale_2,
        w2_scale_2=w2_scale_2,
        a13_scale=a13_scale,
        a2_scale=a2_scale,
    )

    torch.testing.assert_close(config.a1_gscale, 1.0 / a13_scale)
    torch.testing.assert_close(config.a2_gscale, 1.0 / a2_scale)
