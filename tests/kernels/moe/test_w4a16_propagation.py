# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for use_a16 propagation in NVFP4 MoE quant configs.

Regression test for the gap where ``use_a16`` in
``ModelOptNvFp4FusedMoE`` / ``CompressedTensorsW4A4Nvfp4MoEMethod``
only gated backend selection but left activation global scales flowing
through to the kernel via ``a1_gscale``.  After the fix, a missing /
``None`` activation scale must produce a quant config with
``a1_gscale=None``, which signals ``activation_precision="bf16"`` in
``FlashInferB12xExperts``.  No SM12x hardware required — this exercises
the config builder, not the kernel.
"""

import torch

from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    make_nvfp4_moe_quant_config,
)


def _dummy_weight_scales(e: int, n: int, k: int):
    w13_scale = torch.zeros((e, 2 * n, k // 16), dtype=torch.uint8)
    w2_scale = torch.zeros((e, k, n // 16), dtype=torch.uint8)
    w13_scale_2 = torch.ones(e, dtype=torch.float32)
    w2_scale_2 = torch.ones(e, dtype=torch.float32)
    return w13_scale, w2_scale, w13_scale_2, w2_scale_2


def test_w4a16_b12x_yields_none_a_gscale():
    """W4A16 (a13_scale=None) on the B12x backend yields a1_gscale=None.

    Pre-fix behavior: ``make_nvfp4_moe_quant_config`` only routed MARLIN
    through the W4A16 builder; B12x with ``a13_scale=None`` would have
    crashed on ``1.0 / None`` in the W4A4 branch.  Post-fix: any backend
    with ``a13_scale=None`` routes to ``nvfp4_w4a16_moe_quant_config``.
    """
    w13_scale, w2_scale, w13_scale_2, w2_scale_2 = _dummy_weight_scales(8, 64, 128)
    config = make_nvfp4_moe_quant_config(
        backend=NvFp4MoeBackend.FLASHINFER_B12X,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_scale_2=w13_scale_2,
        w2_scale_2=w2_scale_2,
        a13_scale=None,
        a2_scale=None,
    )
    assert config.a1_gscale is None
    assert config.a2_gscale is None


def test_w4a4_b12x_populates_a_gscale():
    """W4A4 (a13_scale not None) on B12x should still populate a1_gscale.

    Regression guard: the W4A16 routing must be triggered only by
    ``a13_scale is None`` — not by backend choice — so the W4A4 path
    remains intact.
    """
    w13_scale, w2_scale, w13_scale_2, w2_scale_2 = _dummy_weight_scales(8, 64, 128)
    a13_scale = torch.full((8,), 0.5, dtype=torch.float32)
    a2_scale = torch.full((8,), 0.5, dtype=torch.float32)
    config = make_nvfp4_moe_quant_config(
        backend=NvFp4MoeBackend.FLASHINFER_B12X,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_scale_2=w13_scale_2,
        w2_scale_2=w2_scale_2,
        a13_scale=a13_scale,
        a2_scale=a2_scale,
    )
    assert config.a1_gscale is not None
    assert config.a2_gscale is not None


def test_w4a16_marlin_yields_none_a_gscale():
    """MARLIN W4A16 (already correct pre-fix) — keeps working post-fix."""
    w13_scale, w2_scale, w13_scale_2, w2_scale_2 = _dummy_weight_scales(8, 64, 128)
    config = make_nvfp4_moe_quant_config(
        backend=NvFp4MoeBackend.MARLIN,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_scale_2=w13_scale_2,
        w2_scale_2=w2_scale_2,
        a13_scale=None,
        a2_scale=None,
    )
    assert config.a1_gscale is None
    assert config.a2_gscale is None
