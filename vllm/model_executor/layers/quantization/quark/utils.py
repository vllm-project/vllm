# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

import regex as re
import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from torch import nn


def deep_compare(dict1: Any, dict2: Any) -> bool:
    if type(dict1) is not type(dict2):
        return False
    if isinstance(dict1, dict):
        if dict1.keys() != dict2.keys():
            return False
        return all(deep_compare(dict1[k], dict2[k]) for k in dict1)
    elif isinstance(dict1, list):
        return set(dict1) == set(dict2)
    else:
        return dict1 == dict2


def should_ignore_layer(
    layer_name: str | None,
    ignore: Iterable[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> bool:
    if layer_name is None:
        return False

    # layer_name = model.layers.0.self_attn.qkv_proj
    # proj_name = qkv_proj
    proj_name = layer_name.split(".")[-1]

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    if proj_name in fused_mapping:
        shard_proj_names = fused_mapping[proj_name]

        # Convert fused_name --> [shard_names]
        shard_names = [
            layer_name.replace(proj_name, shard_proj_name)
            for shard_proj_name in shard_proj_names
        ]

        # Layer should be ignored if shards are ignored.
        should_ignore_layer = None
        for shard_name in shard_names:
            should_ignore_shard = check_equal_or_regex_match(
                layer_name=shard_name, targets=ignore
            )

            # If shard_idx=0, set layer ignore to match shard.
            if should_ignore_layer is None:
                should_ignore_layer = should_ignore_shard

            # If shard_idx=1+ confirm scheme matches prior shards.
            elif should_ignore_shard != should_ignore_layer:
                raise ValueError(
                    f"Found a different quantization schemes for "
                    f"{shard_proj_names} in {layer_name}. vLLM "
                    "requires all to use the same scheme."
                )

    # Unfused layers like down_proj and o_proj will match
    # the safetensors checkpoint already.
    else:
        should_ignore_layer = check_equal_or_regex_match(
            layer_name=layer_name, targets=ignore
        )

    assert should_ignore_layer is not None
    return should_ignore_layer


def check_equal_or_regex_match(layer_name: str, targets: Iterable[str]) -> bool:
    """
    Checks whether a layer_name is exactly equal or a regex match for
    if target starts with 're:' to any target in list.
    """
    return any(_is_equal_or_regex_match(layer_name, target) for target in targets)


def _is_equal_or_regex_match(
    value: str, target: str, check_contains: bool = False
) -> bool:
    """
    Checks whether a value is exactly equal or a regex match for target
    if target starts with 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.
    """

    if target.startswith("re:"):
        pattern = target[3:]
        if re.match(pattern, value):
            return True
    elif check_contains:
        if target.lower() in value.lower():
            return True
    elif target == value:
        return True
    return False


# utility for tensor dims > 2 cases
def b_dynamic_mxfp4_quant(x):
    h, b, d = x.shape
    x, x_scales = dynamic_mxfp4_quant(x.reshape(-1, d))
    return x.view(h, b, d // 2), x_scales.view(h, b, d // 32)


def mxfp4_to_f32(x, is_threed):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=-1)
    if is_threed:
        x[..., ::2] = x[..., ::2] & 0xF
        x[..., 1::2] = x[..., 1::2] >> 4
    else:
        x[:, ::2] = x[:, ::2] & 0xF
        x[:, 1::2] = x[:, 1::2] >> 4

    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    # Convert the input tensor `x` (assumed to be in e8m0 format) to float32.
    # e8m0 is a custom 8-bit floating point format with 8 bits for exponent, 0 for
    # mantissa. This means the value is essentially 2^(exponent - 127), similar to how
    # IEEE-754 stores floats.

    # Convert x to float32 for computation, and compute the power of 2 by subtracting
    # the bias (127).
    x_f32 = 2 ** ((x.to(torch.float32)) - 127)

    # If the exponent value was 255 (i.e., 2^(128)), this is a special case usually used
    # to represent NaN or Inf.
    # Since this custom format has no mantissa, treat 2^128 as NaN.
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def quark_post_load_weights(self_attn: nn.Module, w: torch.Tensor, quant_format: str):
    if "mxfp4" in quant_format:
        # when dtype is bf16, the processing flow is to dynamic quantize bf16 tensor to
        # uint8 tensor do w_kc (bf16) first to get the w_kc(uint8) w_s_kc(uint8) and
        # w_vc repeating the same procedure of w_kc to get  w_vc(uint8) w_s_vc(uint8)
        if w.dtype == torch.bfloat16:
            # w_kc, w_vc = w.split(
            # [self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            w_kc, w_vc = w.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            w_kc, w_s_kc = b_dynamic_mxfp4_quant(w_kc.transpose(-2, -1))
            w_kc = w_kc.transpose(-2, -1)
            w_s_kc = w_s_kc.transpose(-2, -1)
            w_vc, w_s_vc = b_dynamic_mxfp4_quant(w_vc)
            w_s_kc = w_s_kc.transpose(1, 2).contiguous().transpose(1, 2)
            w_s_vc = w_s_vc.contiguous().transpose(1, 2)
        elif w.dtype == torch.uint8:  # static quant for mxfp4
            # when dtype is uint8, it means the w has been quantized to mxfp4 format
            # but we must separate it to w_kc and w_vc.
            # The quantized tensor size is only half of original tensor size
            # and the scaling factor is 1/32, the transpose behavior will be not correct
            # need to upcast it to fp32 to separate w to w_kc and w_vc
            # to ensure the following transpose behavior is correct
            # and then do mxfp4 quant again
            w = mxfp4_to_f32(w, True).to(torch.bfloat16)
            w_scales = self_attn.kv_b_proj.weight_scale.repeat_interleave(32, dim=-1)
            w_scales = e8m0_to_f32(w_scales).to(torch.bfloat16)
            w = w * w_scales
            w_kc, w_vc = w.unflatten(
                0, (-1, (self_attn.qk_nope_head_dim + self_attn.v_head_dim))
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            w_kc, w_s_kc = b_dynamic_mxfp4_quant(w_kc.transpose(-2, -1))
            w_kc = w_kc.transpose(-2, -1)
            w_s_kc = w_s_kc.transpose(-2, -1)
            w_vc, w_s_vc = b_dynamic_mxfp4_quant(w_vc)
            w_s_kc = w_s_kc.transpose(1, 2).contiguous().transpose(1, 2)
            w_s_vc = w_s_vc.contiguous().transpose(1, 2)

        return w_kc, w_s_kc, w_vc, w_s_vc
