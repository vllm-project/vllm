# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

import torch

from vllm.model_executor.layers.quantization.utils.config_utils import (
    find_matching_patterns,
)

QuarkQTensorHint = dict[str, Any] | list[dict[str, Any]] | None


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

    # A direct fused-layer pattern takes precedence over expansion. For
    # model.layers.0.self_attn.qkv_proj,
    # ignore=["re:.*qkv_proj.*"] yields [{"re:.*qkv_proj.*"}]. In contrast,
    # ignore=["re:.*[qkv]_proj"] yields one matching set per expanded shard.
    per_shard_matches = find_matching_patterns(layer_name, ignore, fused_mapping)
    shards_ignored = [len(matches) > 0 for matches in per_shard_matches]
    if any(shards_ignored) and not all(shards_ignored):
        raise ValueError(
            f"Found different quantization schemes for the shards of "
            f"{layer_name}. vLLM requires all to use the same scheme."
        )
    return all(shards_ignored)


def parse_w4a16_int4_weight_config(
    weight_config: Mapping[str, Any],
) -> tuple[int, bool]:
    """Parse required W4A16 INT4/UINT4 weight fields from Quark config."""
    if "group_size" not in weight_config:
        raise ValueError(
            "Quark W4A16 INT4/UINT4 configs must specify weight.group_size"
        )
    if "symmetric" not in weight_config:
        raise ValueError("Quark W4A16 INT4/UINT4 configs must specify weight.symmetric")

    group_size = weight_config["group_size"]
    is_symmetric = weight_config["symmetric"]
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError(
            f"Quark W4A16 weight.group_size must be a positive int, got {group_size!r}"
        )
    if not isinstance(is_symmetric, bool):
        raise ValueError(
            f"Quark W4A16 weight.symmetric must be a bool, got {is_symmetric!r}"
        )
    return group_size, is_symmetric


def canonicalize_quark_packed_int4(
    packed_weight: torch.Tensor,
    *,
    pack_reorder: bool,
    is_symmetric: bool,
    pack_factor: int = 8,
) -> torch.Tensor:
    """Convert Quark export nibble layout to AWQ checkpoint layout."""
    from vllm.model_executor.layers.quantization.auto_awq import (
        _REVERSE_AWQ_PACK_ORDER,
    )

    if pack_reorder:
        source_order = torch.tensor(
            _REVERSE_AWQ_PACK_ORDER, device=packed_weight.device, dtype=torch.int32
        )
    else:
        source_order = torch.arange(
            pack_factor, device=packed_weight.device, dtype=torch.int32
        )
    target_order = torch.tensor(
        _REVERSE_AWQ_PACK_ORDER, device=packed_weight.device, dtype=torch.int32
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
