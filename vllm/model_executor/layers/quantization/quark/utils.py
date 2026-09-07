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


# utility for tensor dims > 2 cases
def quark_quantize_weight_to_mxfp4(w: torch.Tensor):
    assert w.dtype == torch.bfloat16, (
        "Quark dynamic quantization is supported only for fp16 weights and only to MXF4"
    )

    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    *dims, d = w.shape
    w, w_scales = dynamic_mxfp4_quant(w.reshape(-1, d))
    return w.view(*dims, d // 2), w_scales.view(*dims, d // 32)
