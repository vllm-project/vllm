# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

import regex as re
import torch


def deep_compare(dict1: Any, dict2: Any) -> bool:
    if type(dict1) is not type(dict2):
        return False
    if isinstance(dict1, dict):
        if dict1.keys() != dict2.keys():
            return False
        return all(deep_compare(dict1[k], dict2[k]) for k in dict1)
    elif isinstance(dict1, list):
        # `dict1` may be a list of dict.
        return all(deep_compare(dict1[i], dict2[i]) for i in range(len(dict1)))
    else:
        return dict1 == dict2


def should_ignore_layer(
    layer_name: str | None,
    ignore: Iterable[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
    *,
    check_children: bool = False,
) -> bool:
    if layer_name is None:
        return False

    # MoE layers are currently all-or-nothing: if any child is ignored,
    # the parent layer must be ignored as well. For example, the
    # amd/GLM-5.2-MXFP4 config ignores children like
    # model.layers.78.mlp.experts.*.down_proj, while the layer checked
    # here is the parent model.layers.N.mlp.experts.
    # See:
    # https://huggingface.co/amd/GLM-5.2-MXFP4/blob/main/config.json#L793-L795
    if check_children and any(
        target == layer_name or target.startswith(layer_name + ".")
        for target in ignore
        if not target.startswith("re:")
    ):
        return True

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


def parse_w4a16_int4_weight_config(
    weight_config: Mapping[str, Any],
) -> tuple[int, bool]:
    """Parse required W4A16 INT4/UINT4 weight fields from Quark config."""
    if "group_size" not in weight_config:
        raise ValueError(
            "Quark W4A16 INT4/UINT4 configs must specify weight.group_size"
        )
    if "symmetric" not in weight_config:
        raise ValueError(
            "Quark W4A16 INT4/UINT4 configs must specify weight.symmetric"
        )

    group_size = weight_config["group_size"]
    is_symmetric = weight_config["symmetric"]
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError(
            f"Quark W4A16 weight.group_size must be a positive int, got {group_size!r}"
        )
    if not isinstance(is_symmetric, bool):
        raise ValueError(
            "Quark W4A16 weight.symmetric must be a bool, "
            f"got {is_symmetric!r}"
        )
    return group_size, is_symmetric


_AWQ_PACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def canonicalize_quark_packed_int4(
    packed_weight: torch.Tensor,
    *,
    pack_reorder: bool,
    is_symmetric: bool,
    pack_factor: int = 8,
) -> torch.Tensor:
    """Convert Quark export nibble layout to AWQ checkpoint layout."""
    if pack_reorder:
        source_order = torch.tensor(
            _AWQ_PACK_ORDER, device=packed_weight.device, dtype=torch.int32
        )
    else:
        source_order = torch.arange(
            pack_factor, device=packed_weight.device, dtype=torch.int32
        )
    target_order = torch.tensor(
        _AWQ_PACK_ORDER, device=packed_weight.device, dtype=torch.int32
    )
    source_shifts = source_order * 4
    target_shifts = target_order * 4

    values = (packed_weight.to(torch.int32)[..., None] >> source_shifts) & 0xF
    if is_symmetric:
        values = values ^ 0x8
    packed = (values.to(torch.int64) << target_shifts.to(torch.int64)).sum(dim=-1)
    return packed.to(torch.int32)


# utility for tensor dims > 2 cases
def quark_quantize_weight_to_mxfp4(w: torch.Tensor):
    assert w.dtype == torch.bfloat16, (
        "Quark dynamic quantization is supported only for fp16 weights and only to MXF4"
    )

    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    *dims, d = w.shape
    w, w_scales = dynamic_mxfp4_quant(w.reshape(-1, d))
    return w.view(*dims, d // 2), w_scales.view(*dims, d // 32)
