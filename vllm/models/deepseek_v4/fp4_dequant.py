# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless MXFP4 -> block-FP8 expert dequant for DeepSeek-V4 on Hopper.

DeepSeek-V4 Instruct checkpoints ship MXFP4 (e2m1 + e8m0 microscale) routed
experts. On Hopper (SM90) the default MXFP4 path runs the Marlin W4A16 kernel.
When ``VLLM_DSV4_FP4_DEQUANT=1`` we instead losslessly re-encode the experts to
block-FP8 (e4m3, 128x128 scales) at load time and run them through the existing
block-FP8 MoE kernel, which uses FP8 tensor cores for materially higher prefill
throughput. The trade-off is larger expert weight memory (FP8 is 2x MXFP4).

The re-encode is bit-exact: MX scales are pure powers of two (e8m0) and e4m3 has
enough mantissa/range to absorb the per-group scale ratio into the stored value,
so no precision is lost relative to the MXFP4 weights.
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

_FP8_BLOCK = 128
_FP4_GROUP = 32
# max_fp4 (6.0) * 2**MAX_OFFSET_BITS must fit e4m3 (max 448): 6*2**6=384 < 448.
_MAX_OFFSET_BITS = 6


def dsv4_fp4_dequant_enabled() -> bool:
    """Whether the opt-in MXFP4->FP8 expert dequant path is enabled."""
    return envs.VLLM_DSV4_FP4_DEQUANT


def cast_mxfp4_to_fp8_block(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Losslessly re-encode one MXFP4 expert weight to block-FP8.

    Args:
        weight: uint8 ``[out, in // 2]`` MXFP4 weights (two e2m1 nibbles/byte).
        scale: uint8 ``[out, in // 32]`` e8m0 microscale, one per 32-elem group.

    Returns:
        A tuple ``(fp8_weight, block_scale)`` where ``fp8_weight`` is
        float8_e4m3fn ``[out, in]`` and ``block_scale`` is float32
        ``[out // 128, in // 128]``.
    """
    lut = _FP4_LUT.to(weight.device)
    out_dim, in_half = weight.shape
    in_dim = in_half * 2
    if in_dim % _FP8_BLOCK or out_dim % _FP8_BLOCK:
        raise ValueError(
            f"MXFP4->FP8 requires dims divisible by {_FP8_BLOCK}, "
            f"got out={out_dim} in={in_dim}"
        )
    low = weight & 0x0F
    high = (weight >> 4) & 0x0F
    x = torch.stack([lut[low.long()], lut[high.long()]], dim=-1).flatten(1)
    scl = scale.view(torch.float8_e8m0fnu).float()

    b_out, b_in = out_dim // _FP8_BLOCK, in_dim // _FP8_BLOCK
    x = x.view(b_out, _FP8_BLOCK, b_in, _FP8_BLOCK).transpose(1, 2)
    scl = scl.view(b_out, _FP8_BLOCK, b_in, -1).transpose(1, 2).flatten(2)
    block_scale = scl.amax(dim=-1, keepdim=True) / (2**_MAX_OFFSET_BITS)
    offset = scl / block_scale
    offset = offset.unflatten(-1, (_FP8_BLOCK, -1)).repeat_interleave(
        _FP4_GROUP, dim=-1
    )
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), block_scale.squeeze(-1).float()


def _convert_experts(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weights, scales = [], []
    for e in range(weight.shape[0]):
        w, s = cast_mxfp4_to_fp8_block(weight[e], scale[e])
        weights.append(w)
        scales.append(s)
    return torch.stack(weights), torch.stack(scales)


class DsV4Fp4DequantMoEMethod(Fp8MoEMethod):
    """Load MXFP4 experts, re-encode to block-FP8, run the FP8 MoE kernel."""

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
        w13, w13_scale = _convert_experts(
            layer.w13_weight.data, layer.w13_weight_scale.data
        )
        w2, w2_scale = _convert_experts(
            layer.w2_weight.data, layer.w2_weight_scale.data
        )
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
