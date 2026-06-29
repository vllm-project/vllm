# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization.utils.quant_fusion import (
    get_mla_attn_quant_params,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
)


def _mla_attn(supported: bool = True, fuse_attn_quant: bool = True):
    """Create a mock MLA attention object."""
    attn = SimpleNamespace(
        use_fused_attn_quant=fuse_attn_quant,
        impl=SimpleNamespace(
            fused_output_quant_supported=lambda quant_key: supported,
        ),
    )
    return attn


def _output_proj(activation_quant_key=kFp8StaticTensorSym):
    """Create a mock output projection layer."""
    return SimpleNamespace(
        input_scale=torch.tensor([0.5]),
        input_quant_key=activation_quant_key,
        input_block_scale=torch.tensor([0.25])
        if activation_quant_key != kFp8StaticTensorSym
        else None,
        quant_group_size=128 if activation_quant_key != kFp8StaticTensorSym else None,
    )


def test_mla_attn_quant_params_selected_when_supported_fp8_static():
    """Test that FP8 static quantization params are returned when supported."""
    attn = _mla_attn(supported=True, fuse_attn_quant=True)
    output_proj = _output_proj(kFp8StaticTensorSym)

    output_scale, output_block_scale, quant_group_size = get_mla_attn_quant_params(
        attn, output_proj
    )

    assert output_scale is output_proj.input_scale
    assert output_block_scale is None
    assert quant_group_size is None


def test_mla_attn_quant_params_selected_when_supported_fp8_group():
    """Test that per-group FP8 quantization params are returned when supported."""
    attn = _mla_attn(supported=True, fuse_attn_quant=True)
    output_proj = _output_proj(kFp8Dynamic128Sym)

    output_scale, output_block_scale, quant_group_size = get_mla_attn_quant_params(
        attn, output_proj
    )

    assert output_scale is output_proj.input_scale
    assert output_block_scale is output_proj.input_block_scale
    assert quant_group_size == output_proj.quant_group_size


def test_mla_attn_quant_params_skipped_without_input_quant_key():
    """Test that quant params are None when output_proj has no input_quant_key."""
    attn = _mla_attn(supported=True, fuse_attn_quant=True)
    output_proj = SimpleNamespace(input_scale=torch.tensor([0.5]))

    output_scale, output_block_scale, quant_group_size = get_mla_attn_quant_params(
        attn, output_proj
    )

    assert output_scale is None
    assert output_block_scale is None
    assert quant_group_size is None


def test_mla_attn_quant_params_skipped_when_flag_disabled():
    """Test that quant params are None when use_fused_attn_quant is False."""
    attn = _mla_attn(supported=True, fuse_attn_quant=False)
    output_proj = _output_proj(kFp8StaticTensorSym)

    output_scale, output_block_scale, quant_group_size = get_mla_attn_quant_params(
        attn, output_proj
    )

    assert output_scale is None
    assert output_block_scale is None
    assert quant_group_size is None


def test_mla_attn_quant_params_skipped_when_backend_unsupported():
    """Test that quant params are None when backend doesn't support fusion."""
    attn = _mla_attn(supported=False, fuse_attn_quant=True)
    output_proj = _output_proj(kFp8StaticTensorSym)

    output_scale, output_block_scale, quant_group_size = get_mla_attn_quant_params(
        attn, output_proj
    )

    assert output_scale is None
    assert output_block_scale is None
    assert quant_group_size is None


def test_mla_attn_quant_params_skipped_when_backend_has_no_support_probe():
    """Test that quant params are None when backend has no support probe method."""
    attn = _mla_attn(supported=True, fuse_attn_quant=True)
    attn.impl = SimpleNamespace()  # No fused_output_quant_supported method
    output_proj = _output_proj(kFp8StaticTensorSym)

    output_scale, output_block_scale, quant_group_size = get_mla_attn_quant_params(
        attn, output_proj
    )

    assert output_scale is None
    assert output_block_scale is None
    assert quant_group_size is None


def test_mla_attn_quant_params_skipped_for_unsupported_quant_key():
    """Test that quant params are None for unsupported quantization keys."""
    attn = _mla_attn(supported=True, fuse_attn_quant=True)
    # Use a quant key that's not in the supported set
    output_proj = SimpleNamespace(
        input_scale=torch.tensor([0.5]),
        input_quant_key="unsupported_key",
    )

    output_scale, output_block_scale, quant_group_size = get_mla_attn_quant_params(
        attn, output_proj
    )

    assert output_scale is None
    assert output_block_scale is None
    assert quant_group_size is None
