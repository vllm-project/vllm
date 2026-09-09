# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``vllm.model_executor.layers.quantization.utils.nvfp4_utils``.

Run `pytest tests/quantization/test_nvfp4_utils.py`.
"""

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    requantize_nvfp4_moe_w13_scale_2,
)


def _make_mismatched_w13(
    num_experts: int,
    intermediate_size: int,
    num_blocks: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a fused w13 whose gate and up global scales differ per expert.

    Returns ``(w13_weight_scale, w13_weight_scale_2, gate_block_scale,
    up_block_scale)`` where the last two are the *original* (float32,
    pre-quantization) block scales for each half, kept around so the test
    can recompute what each half's true per-element scale (``block_scale *
    global_scale``) was meant to be.
    """
    g = torch.Generator().manual_seed(seed)

    # Global scales (gs_gate, gs_up) per expert, deliberately mismatched --
    # some ties (ratio 1), some far apart, matching the issue's report of
    # ratios from 1.0 up to 10x.
    gs_gate = torch.empty(num_experts).uniform_(0.05, 2.0, generator=g)
    ratio = torch.empty(num_experts).uniform_(0.5, 10.0, generator=g)
    gs_up = gs_gate * ratio
    w13_weight_scale_2 = torch.stack([gs_gate, gs_up], dim=1)

    # Random block scales in the normal e4m3 range for each half.
    gate_block_scale = torch.empty(num_experts, intermediate_size, num_blocks).uniform_(
        0.1, 1.0, generator=g
    )
    up_block_scale = torch.empty(num_experts, intermediate_size, num_blocks).uniform_(
        0.1, 1.0, generator=g
    )

    w13_weight_scale = torch.cat([gate_block_scale, up_block_scale], dim=1).to(
        torch.float8_e4m3fn
    )

    return (
        w13_weight_scale,
        w13_weight_scale_2,
        gate_block_scale,
        up_block_scale,
    )


def test_requantize_shares_the_max_global_scale() -> None:
    """The returned per-expert scale must be exactly max(gs_gate, gs_up)."""
    w13_weight_scale, w13_weight_scale_2, _, _ = _make_mismatched_w13(
        num_experts=4, intermediate_size=8, num_blocks=3, seed=0
    )
    expected_gs = torch.maximum(w13_weight_scale_2[:, 0], w13_weight_scale_2[:, 1])

    gs = requantize_nvfp4_moe_w13_scale_2(w13_weight_scale, w13_weight_scale_2)

    assert gs.shape == (4,)
    torch.testing.assert_close(gs, expected_gs)


def test_dequantized_product_matches_original_within_e4m3_tolerance() -> None:
    """Rescaling must preserve each half's true per-element scale.

    Before the fix, the up half was dequantized as
    ``block_scale * gs_gate`` instead of ``block_scale * gs_up`` -- exactly
    the bug this function fixes. After requantizing onto the shared
    ``gs = max(gs_gate, gs_up)``, ``new_block_scale * gs`` must reproduce
    the *original* ``block_scale * gs_half`` for both halves, up to e4m3
    block-scale rounding.
    """
    num_experts, intermediate_size, num_blocks = 4, 8, 3
    (
        w13_weight_scale,
        w13_weight_scale_2,
        gate_block_scale,
        up_block_scale,
    ) = _make_mismatched_w13(num_experts, intermediate_size, num_blocks, seed=1)

    gs_gate = w13_weight_scale_2[:, 0]
    gs_up = w13_weight_scale_2[:, 1]
    # The true per-element scale each half was quantized against.
    original_gate = gate_block_scale * gs_gate.view(-1, 1, 1)
    original_up = up_block_scale * gs_up.view(-1, 1, 1)

    gs = requantize_nvfp4_moe_w13_scale_2(w13_weight_scale, w13_weight_scale_2)

    new_gate_block_scale = w13_weight_scale[:, :intermediate_size, :].to(torch.float32)
    new_up_block_scale = w13_weight_scale[:, intermediate_size:, :].to(torch.float32)
    new_gate = new_gate_block_scale * gs.view(-1, 1, 1)
    new_up = new_up_block_scale * gs.view(-1, 1, 1)

    # e4m3fn has ~2 bits of mantissa; the block scales pass through it twice
    # (once on disk, once in this rescale), so allow a generous but bounded
    # relative tolerance -- this must catch the old bug (which is off by the
    # full gs_gate/gs_up ratio, up to 10x in the reported checkpoint) while
    # tolerating quantization noise.
    torch.testing.assert_close(new_gate, original_gate, rtol=0.15, atol=1e-4)
    torch.testing.assert_close(new_up, original_up, rtol=0.15, atol=1e-4)


def test_rescale_factor_never_exceeds_one() -> None:
    """The minority half's block scales must only shrink, never overflow.

    ``factor = gs_half / max(gs_gate, gs_up) <= 1`` by construction; the
    e4m3 block scale for the max-scale half is unchanged (factor == 1).
    """
    num_experts, intermediate_size, num_blocks = 6, 4, 2
    w13_weight_scale, w13_weight_scale_2, _, _ = _make_mismatched_w13(
        num_experts, intermediate_size, num_blocks, seed=2
    )
    before = w13_weight_scale.clone().to(torch.float32)

    requantize_nvfp4_moe_w13_scale_2(w13_weight_scale, w13_weight_scale_2)

    after = w13_weight_scale.to(torch.float32)
    # No block scale may grow: every rescale factor is <= 1.
    assert torch.all(after <= before + 1e-6)


def test_tied_scales_are_a_no_op() -> None:
    """When gs_gate == gs_up, requantizing must not change the block scales.

    This is the common case in practice (the issue's two modelopt-0.46
    checkpoints tie exactly on every pair); the function should still be
    safe to call on it.
    """
    num_experts, intermediate_size, num_blocks = 3, 5, 2
    g = torch.Generator().manual_seed(3)
    gs = torch.empty(num_experts).uniform_(0.1, 2.0, generator=g)
    w13_weight_scale_2 = torch.stack([gs, gs], dim=1)
    block_scale = torch.empty(num_experts, 2 * intermediate_size, num_blocks).uniform_(
        0.1, 1.0, generator=g
    )
    w13_weight_scale = block_scale.to(torch.float8_e4m3fn)
    before = w13_weight_scale.clone()

    result_gs = requantize_nvfp4_moe_w13_scale_2(w13_weight_scale, w13_weight_scale_2)

    torch.testing.assert_close(result_gs, gs)
    torch.testing.assert_close(w13_weight_scale, before)
