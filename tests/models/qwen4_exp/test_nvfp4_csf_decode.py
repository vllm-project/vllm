# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.qwen4_exp.nvidia.model import (
    decode_nvfp4_csf,
    maybe_decode_nvfp4_csf_weights,
)


def test_decode_nvfp4_csf_dtype_and_shape():
    """Verify decode_nvfp4_csf returns float8_e4m3fn and correct shapes."""
    num_slabs = 4
    slab_bytes = 48  # (48 - 16) // 16 = 2 packed cols -> 4 unpacked cols
    num_rows = num_slabs * 16
    expected_cols = 4

    fixed = torch.zeros((num_slabs, slab_bytes), dtype=torch.uint8)
    fixed[:, :16] = 0x38  # 1.0 in FP8 E4M3FN

    decoded = decode_nvfp4_csf(fixed)
    assert decoded.dtype == torch.float8_e4m3fn
    assert decoded.shape == (num_rows, expected_cols)


def test_decode_nvfp4_csf_with_exceptions():
    """Verify exceptions overwrite decoded base+nibbles correctly."""
    num_slabs = 2
    slab_bytes = 32
    num_rows = num_slabs * 16

    fixed = torch.zeros((num_slabs, slab_bytes), dtype=torch.uint8)
    # Pack an exception at flat position 5 with byte value 0x40
    target_pos = 5
    exact_byte = 0x40
    packed_val = target_pos | (exact_byte << 24)
    exceptions = torch.tensor([packed_val], dtype=torch.int32)

    decoded = decode_nvfp4_csf(fixed, exceptions)
    assert decoded.dtype == torch.float8_e4m3fn

    raw_bytes = decoded.view(torch.uint8).flatten()
    assert raw_bytes[target_pos].item() == exact_byte


def test_maybe_decode_nvfp4_csf_weights():
    """Verify paired CSF weight keys are fused and decoded."""
    prefix = "model.layers.0.mlp.experts.0.gate_up_proj.weight_scale"
    fixed_name = f"{prefix}.nvfp4_csf_fixed"
    exc_name = f"{prefix}.nvfp4_csf_exceptions"

    fixed_tensor = torch.zeros((2, 32), dtype=torch.uint8)
    exc_tensor = torch.empty((0,), dtype=torch.int32)

    weights = [
        (fixed_name, fixed_tensor),
        (exc_name, exc_tensor),
        ("unrelated_layer.weight", torch.ones((4, 4), dtype=torch.bfloat16)),
    ]

    decoded_weights = dict(maybe_decode_nvfp4_csf_weights(weights))
    assert prefix in decoded_weights
    assert fixed_name not in decoded_weights
    assert exc_name not in decoded_weights
    assert decoded_weights[prefix].dtype == torch.float8_e4m3fn
    assert "unrelated_layer.weight" in decoded_weights
