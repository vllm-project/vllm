# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validated MXFP4 -> block-FP8 expert dequant for DeepSeek-V4 on Hopper.

DeepSeek-V4 Instruct checkpoints ship MXFP4 (e2m1 + e8m0 microscale) routed
experts. On Hopper (SM90) the default MXFP4 path runs the Marlin W4A16 kernel.
When ``VLLM_DSV4_FP4_DEQUANT=1`` we instead re-encode exactly representable
experts to block-FP8 (e4m3, 128x128 scales) at load time and run them through
the existing block-FP8 MoE kernel, which uses FP8 tensor cores for materially
higher prefill throughput. Loading fails before replacing layer parameters if
an invalid e8m0 scale or an inexact weight conversion is found. The trade-off
is larger expert weight memory (FP8 is 2x MXFP4).

Exactness here applies to decoded MXFP4 weight values accepted by the load-time
check. The FP8 execution path still uses dynamic activation quantization.
"""

import torch

from vllm import envs
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

# e2m1 (FP4) code -> value lookup table (sign bit is the high code bit).
_FP4_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

# Largest exponent spread for which each e2m1 code remains exactly
# representable after normalization to an e4m3fn block. Zero is unrestricted.
_FP4_MAX_EXACT_SCALE_SPREAD = torch.tensor(
    [255, 14, 15, 14, 16, 15, 17, 16] * 2,
    dtype=torch.uint8,
)

_FP8_BLOCK = 128
_FP4_GROUP = 32
# max_fp4 (6.0) * 2**MAX_OFFSET_BITS must fit e4m3 (max 448): 6*2**6=384 < 448.
_MAX_OFFSET_BITS = 6
_INVALID_E8M0_SCALE = 0xFF


def dsv4_fp4_dequant_enabled() -> bool:
    """Whether the opt-in MXFP4->FP8 expert dequant path is enabled."""
    return envs.VLLM_DSV4_FP4_DEQUANT


def _raise_for_conversion_issues(issue_counts: torch.Tensor) -> None:
    invalid_scales, inexact_values = issue_counts.detach().cpu().tolist()
    if invalid_scales or inexact_values:
        raise ValueError(
            "MXFP4->FP8 conversion rejected: "
            f"{invalid_scales} invalid E8M0 scale value(s), "
            f"{inexact_values} weight value(s) not exactly representable "
            "as block E4M3FN"
        )


def _cast_mxfp4_to_fp8_block(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            "MXFP4->FP8 requires 2D weight and scale tensors, "
            f"got weight.ndim={weight.ndim} scale.ndim={scale.ndim}"
        )
    if weight.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise ValueError(
            "MXFP4->FP8 requires uint8 weight and scale tensors, "
            f"got weight.dtype={weight.dtype} scale.dtype={scale.dtype}"
        )
    if weight.device != scale.device:
        raise ValueError(
            "MXFP4->FP8 requires weight and scale on the same device, "
            f"got weight.device={weight.device} scale.device={scale.device}"
        )

    out_dim, in_half = weight.shape
    in_dim = in_half * 2
    if in_dim % _FP8_BLOCK or out_dim % _FP8_BLOCK:
        raise ValueError(
            f"MXFP4->FP8 requires dims divisible by {_FP8_BLOCK}, "
            f"got out={out_dim} in={in_dim}"
        )
    expected_scale_shape = (out_dim, in_dim // _FP4_GROUP)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            "MXFP4->FP8 scale shape mismatch: "
            f"expected {expected_scale_shape}, got {tuple(scale.shape)}"
        )

    invalid_scale = scale == _INVALID_E8M0_SCALE
    safe_scale = scale.masked_fill(invalid_scale, 127)

    low = weight & 0x0F
    high = (weight >> 4) & 0x0F
    codes = torch.stack([low, high], dim=-1).flatten(1)
    lut = _FP4_LUT.to(weight.device)
    x = lut[codes.long()]

    b_out, b_in = out_dim // _FP8_BLOCK, in_dim // _FP8_BLOCK
    code_blocks = codes.view(b_out, _FP8_BLOCK, b_in, _FP8_BLOCK).transpose(1, 2)
    scale_blocks = safe_scale.view(
        b_out, _FP8_BLOCK, b_in, _FP8_BLOCK // _FP4_GROUP
    ).transpose(1, 2)
    max_scale = scale_blocks.amax(dim=(2, 3), keepdim=True)
    scale_spread = (max_scale - scale_blocks).repeat_interleave(_FP4_GROUP, dim=-1)
    max_exact_spread = _FP4_MAX_EXACT_SCALE_SPREAD.to(weight.device)[code_blocks.long()]
    inexact_value = scale_spread > max_exact_spread
    issue_counts = torch.stack(
        [invalid_scale.count_nonzero(), inexact_value.count_nonzero()]
    )

    scl = safe_scale.view(torch.float8_e8m0fnu).float()
    scl = scl.view(b_out, _FP8_BLOCK, b_in, -1).transpose(1, 2).flatten(2)
    block_scale = scl.amax(dim=-1, keepdim=True) / (2**_MAX_OFFSET_BITS)
    offset = scl / block_scale
    offset = offset.unflatten(-1, (_FP8_BLOCK, -1)).repeat_interleave(
        _FP4_GROUP, dim=-1
    )
    x = x.view(b_out, _FP8_BLOCK, b_in, _FP8_BLOCK).transpose(1, 2)
    fp8_weight = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return (
        fp8_weight.to(torch.float8_e4m3fn),
        block_scale.squeeze(-1).float(),
        issue_counts,
    )


def cast_mxfp4_to_fp8_block(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-encode one MXFP4 expert weight to block-FP8 exactly or fail.

    Args:
        weight: uint8 ``[out, in // 2]`` MXFP4 weights (two e2m1 nibbles/byte).
        scale: uint8 ``[out, in // 32]`` e8m0 microscale, one per 32-elem group.

    Returns:
        A tuple ``(fp8_weight, block_scale)`` where ``fp8_weight`` is
        float8_e4m3fn ``[out, in]`` and ``block_scale`` is float32
        ``[out // 128, in // 128]``.

    Raises:
        ValueError: If the tensor layout is unsupported, an e8m0 scale is
            invalid, or any decoded MXFP4 value is not exactly representable.

    """
    fp8_weight, block_scale, issue_counts = _cast_mxfp4_to_fp8_block(weight, scale)
    _raise_for_conversion_issues(issue_counts)
    return fp8_weight, block_scale


def _convert_experts(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights, scales, issue_counts = [], [], []
    for e in range(weight.shape[0]):
        w, s, issues = _cast_mxfp4_to_fp8_block(weight[e], scale[e])
        weights.append(w)
        scales.append(s)
        issue_counts.append(issues)
    return (
        torch.stack(weights),
        torch.stack(scales),
        torch.stack(issue_counts).sum(dim=0),
    )


class DsV4Fp4DequantMoEMethod(Fp8MoEMethod):
    """Load MXFP4 experts, validate exactness, then run the FP8 MoE kernel."""

    supports_pre_processed_weights = False

    def __init__(self, layer):
        super().__init__(
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=[_FP8_BLOCK, _FP8_BLOCK],
            ),
            layer,
        )
        self._mxfp4 = Mxfp4MoEMethod(layer.moe_config)

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        self._mxfp4.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer) -> None:
        w13, w13_scale, w13_issues = _convert_experts(
            layer.w13_weight.data, layer.w13_weight_scale.data
        )
        w2, w2_scale, w2_issues = _convert_experts(
            layer.w2_weight.data, layer.w2_weight_scale.data
        )
        _raise_for_conversion_issues(w13_issues + w2_issues)

        for name in ("w13_weight_scale", "w2_weight_scale"):
            layer._parameters.pop(name, None)

        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.register_parameter(
            "w13_weight_scale_inv",
            torch.nn.Parameter(w13_scale, requires_grad=False),
        )
        layer.register_parameter(
            "w2_weight_scale_inv",
            torch.nn.Parameter(w2_scale, requires_grad=False),
        )
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        layer.weight_block_size = [_FP8_BLOCK, _FP8_BLOCK]
        super().process_weights_after_loading(layer)
