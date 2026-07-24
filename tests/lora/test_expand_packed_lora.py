# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MergedColumnParallelLinearWithLoRA.expand_packed_lora.

Loading a LoRA adapter that targets only some members of a GatedDeltaNet
packed projection group on Qwen3.5/Qwen3.6 (e.g. ``in_proj_qkv`` but not
``in_proj_z``) used to crash in ``expand_packed_lora`` with
``AttributeError: 'NoneType' object has no attribute 'shape'``. These tests
exercise the packed-group expansion logic directly, without a GPU/punica
backend, since it depends only on ``output_sizes`` and ``n_slices``.
"""

import torch

from vllm.lora.layers.column_parallel_linear import (
    MergedColumnParallelLinearWithLoRA,
)

# GDN in_proj_qkvz layout: q, k, v, z output sizes for a 4-slice fused layer.
OUTPUT_SIZES = [512, 128, 128, 256]
QKV_ROWS = sum(OUTPUT_SIZES[:3])  # in_proj_qkv covers slices 0, 1, 2
Z_ROWS = OUTPUT_SIZES[3]  # in_proj_z covers slice 3
RANK = 8


def _make_layer() -> MergedColumnParallelLinearWithLoRA:
    """Build a layer with only the attributes expand_packed_lora reads."""
    layer = MergedColumnParallelLinearWithLoRA.__new__(
        MergedColumnParallelLinearWithLoRA
    )
    layer.output_sizes = OUTPUT_SIZES
    layer.n_slices = len(OUTPUT_SIZES)
    return layer


def test_expand_packed_lora_partial_group_z_absent():
    """in_proj_qkv present, in_proj_z absent (the reported crash)."""
    layer = _make_layer()
    a_qkv = torch.randn(RANK, 64)
    b_qkv = torch.randn(QKV_ROWS, RANK)
    lora_a = [a_qkv, None]
    lora_b = [b_qkv, None]

    # Regression assertion: this must not raise AttributeError.
    expanded_a, expanded_b = layer.expand_packed_lora(lora_a, lora_b)

    assert len(expanded_a) == layer.n_slices
    assert len(expanded_b) == layer.n_slices
    # qkv slices are populated with the right per-slice shapes...
    start = 0
    for i in range(3):
        assert expanded_a[i] is a_qkv
        assert expanded_b[i] is not None
        assert expanded_b[i].shape[0] == OUTPUT_SIZES[i]
        torch.testing.assert_close(
            expanded_b[i], b_qkv[start : start + OUTPUT_SIZES[i], :]
        )
        start += OUTPUT_SIZES[i]
    # ...and the unadapted z slice is left as None (base weights).
    assert expanded_a[3] is None
    assert expanded_b[3] is None


def test_expand_packed_lora_complete_group_unchanged():
    """Both members present: fix must not change existing behavior."""
    layer = _make_layer()
    a_qkv = torch.randn(RANK, 64)
    b_qkv = torch.randn(QKV_ROWS, RANK)
    a_z = torch.randn(RANK, 64)
    b_z = torch.randn(Z_ROWS, RANK)
    lora_a = [a_qkv, a_z]
    lora_b = [b_qkv, b_z]

    expanded_a, expanded_b = layer.expand_packed_lora(lora_a, lora_b)

    assert len(expanded_a) == layer.n_slices
    assert len(expanded_b) == layer.n_slices
    start = 0
    for i in range(3):
        assert expanded_a[i] is a_qkv
        torch.testing.assert_close(
            expanded_b[i], b_qkv[start : start + OUTPUT_SIZES[i], :]
        )
        start += OUTPUT_SIZES[i]
    assert expanded_a[3] is a_z
    torch.testing.assert_close(expanded_b[3], b_z)
