# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

import torch

from vllm.model_executor.layers.quantization.utils.config_utils import (
    check_equal_or_regex_match,
)


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


# utility for tensor dims > 2 cases
def quark_quantize_weight_to_mxfp4(w: torch.Tensor):
    assert w.dtype == torch.bfloat16, (
        "Quark dynamic quantization is supported only for fp16 weights and only to MXF4"
    )

    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    *dims, d = w.shape
    w, w_scales = dynamic_mxfp4_quant(w.reshape(-1, d))
    return w.view(*dims, d // 2), w_scales.view(*dims, d // 32)
