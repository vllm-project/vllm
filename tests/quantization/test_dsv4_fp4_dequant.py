# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 -> block-FP8 expert re-encode must fail closed on inexact input.

The DeepSeek-V4 fp4->fp8 dequant path (``VLLM_DSV4_FP4_DEQUANT=1``) accepts
only inputs whose decoded MXFP4 weight values are exactly representable by the
block-FP8 result.
"""

import pytest
import torch

from vllm.models.deepseek_v4.fp4_dequant import (
    _FP4_LUT,
    DsV4Fp4DequantMoEMethod,
    cast_mxfp4_to_fp8_block,
)


def _dequant_mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    low = (weight & 0x0F).long()
    high = ((weight >> 4) & 0x0F).long()
    vals = torch.stack([_FP4_LUT[low], _FP4_LUT[high]], dim=-1).flatten(1)
    scl = scale.view(torch.float8_e8m0fnu).float().repeat_interleave(32, dim=1)
    return vals * scl


def _dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_dim, in_dim = weight.shape
    bs = 128
    w = weight.float().view(out_dim // bs, bs, in_dim // bs, bs).transpose(1, 2)
    s = scale.float().view(out_dim // bs, in_dim // bs, 1, 1)
    return (w * s).transpose(1, 2).reshape(out_dim, in_dim)


def _make_expert(
    packed_weight: int = 0x11, scale_byte: int = 127
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.full((128, 64), packed_weight, dtype=torch.uint8)
    scale = torch.full((128, 4), scale_byte, dtype=torch.uint8)
    return weight, scale


def _assert_exact_conversion(weight: torch.Tensor, scale: torch.Tensor) -> None:
    ref = _dequant_mxfp4(weight, scale)
    fp8_w, block_scale = cast_mxfp4_to_fp8_block(weight, scale)
    got = _dequant_fp8_block(fp8_w, block_scale)
    assert fp8_w.dtype == torch.float8_e4m3fn
    assert torch.equal(ref, got)


@pytest.mark.parametrize(
    "out_dim,in_dim", [(256, 256), (256, 512), (512, 256), (128, 1024)]
)
def test_cast_mxfp4_to_fp8_block_is_lossless(out_dim: int, in_dim: int):
    torch.manual_seed(0)
    weight = torch.randint(0, 256, (out_dim, in_dim // 2), dtype=torch.uint8)
    # e8m0 exponents around a realistic range (bias 127): 2**-8 .. 2**0.
    scale = torch.randint(127 - 8, 127 + 1, (out_dim, in_dim // 32), dtype=torch.uint8)

    _assert_exact_conversion(weight, scale)


def test_cast_rejects_byte_110_underflow():
    weight, scale = _make_expert()
    scale[0, 0] = 110

    with pytest.raises(ValueError, match="not exactly representable"):
        cast_mxfp4_to_fp8_block(weight, scale)


def test_cast_rejects_non_finite_e8m0_scale():
    weight, scale = _make_expert(packed_weight=0)
    scale[0, 0] = 255

    with pytest.raises(ValueError, match="1 invalid E8M0 scale"):
        cast_mxfp4_to_fp8_block(weight, scale)


@pytest.mark.parametrize(
    "small_scale,packed_weight,exact",
    [
        (113, 0x11, True),  # 0.5 * 2**(6 - 14) is the minimum e4m3 subnormal.
        (112, 0x11, False),  # The same value at spread 15 is not representable.
        (110, 0x66, True),  # 4.0 remains representable at spread 17.
    ],
)
def test_cast_checks_exactness_per_value(
    small_scale: int, packed_weight: int, exact: bool
):
    weight, scale = _make_expert(packed_weight=packed_weight)
    scale[0, 0] = small_scale

    if exact:
        _assert_exact_conversion(weight, scale)
    else:
        with pytest.raises(ValueError, match="not exactly representable"):
            cast_mxfp4_to_fp8_block(weight, scale)


def test_cast_rejects_extreme_scale_spread_without_float32_false_pass():
    weight, scale = _make_expert()
    scale[0, 0] = 0
    scale[0, 1] = 254

    with pytest.raises(ValueError, match="not exactly representable"):
        cast_mxfp4_to_fp8_block(weight, scale)


def test_process_failure_does_not_replace_layer_parameters():
    weight, invalid_scale = _make_expert(packed_weight=0)
    invalid_scale[0, 0] = 255
    _, valid_scale = _make_expert(packed_weight=0)

    layer = torch.nn.Module()
    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(weight.unsqueeze(0), requires_grad=False),
    )
    layer.register_parameter(
        "w13_weight_scale",
        torch.nn.Parameter(invalid_scale.unsqueeze(0), requires_grad=False),
    )
    layer.register_parameter(
        "w2_weight",
        torch.nn.Parameter(weight.unsqueeze(0).clone(), requires_grad=False),
    )
    layer.register_parameter(
        "w2_weight_scale",
        torch.nn.Parameter(valid_scale.unsqueeze(0), requires_grad=False),
    )
    original_parameters = dict(layer._parameters)
    method = object.__new__(DsV4Fp4DequantMoEMethod)

    with pytest.raises(ValueError, match="invalid E8M0 scale"):
        method.process_weights_after_loading(layer)

    assert layer._parameters.keys() == original_parameters.keys()
    for name, parameter in original_parameters.items():
        assert layer._parameters[name] is parameter


def test_cast_rejects_unaligned_dims():
    weight = torch.zeros((64, 64), dtype=torch.uint8)  # out=64 not %128
    scale = torch.full((64, 4), 127, dtype=torch.uint8)
    with pytest.raises(ValueError):
        cast_mxfp4_to_fp8_block(weight, scale)
