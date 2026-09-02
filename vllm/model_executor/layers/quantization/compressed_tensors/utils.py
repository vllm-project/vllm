# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Module

from vllm.model_executor.layers.quantization.utils.config_utils import (
    find_matching_patterns,
    is_equal_or_regex_match,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    PerTensorScaleParameter,
)

# Maps quantization strategy to the corresponding scale parameter type.
# Shared across compressed-tensor scheme classes (w8a16_fp8, w8a8_fp8, …).
STRATEGY_TO_PARAMETER_TYPE = {
    QuantizationStrategy.BLOCK: BlockQuantScaleParameter,
    QuantizationStrategy.CHANNEL: ChannelQuantScaleParameter,
    QuantizationStrategy.TENSOR: PerTensorScaleParameter,
}

# Maps quantization strategy to the vLLM weight-quant key used for
# kernel selection.  Shared across compressed-tensor scheme classes.
STRATEGY_TO_WEIGHT_QUANT_KEY = {
    QuantizationStrategy.BLOCK: kFp8Static128BlockSym,
    QuantizationStrategy.CHANNEL: kFp8StaticChannelSym,
    QuantizationStrategy.TENSOR: kFp8StaticTensorSym,
}


def is_activation_quantization_format(format: str) -> bool:
    _ACTIVATION_QUANTIZATION_FORMATS = [
        CompressionFormat.naive_quantized.value,
        CompressionFormat.int_quantized.value,
        CompressionFormat.float_quantized.value,
        CompressionFormat.nvfp4_pack_quantized.value,
    ]
    return format in _ACTIVATION_QUANTIZATION_FORMATS


def should_ignore_layer(
    layer_name: str | None,
    ignore: Iterable[str] = tuple(),
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
    use_fnmatch: bool = False,
) -> bool:
    if layer_name is None:
        return False
    per_shard_matches = find_matching_patterns(
        layer_name, ignore, fused_mapping, use_fnmatch=use_fnmatch
    )
    shards_ignored = [len(matches) > 0 for matches in per_shard_matches]
    if any(shards_ignored) and not all(shards_ignored):
        raise ValueError(
            f"Found different quantization schemes for the shards of "
            f"{layer_name}. vLLM requires all to use the same scheme."
        )
    return all(shards_ignored)


def find_matched_target(
    layer_name: str | None,
    module: Module,
    targets: Iterable[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> str | None:
    """
    Helper function to look up which "target" in the compressed-tensors
    config that a layer corresponds to.

    Recall that a compressed-tensors configs has a concept of
    config_groups, where each layer can be quantized with a different
    scheme.

    targets in each config_group will be a list of either layer names
    (or regexes corresponding to layer names) or names of torch Modules.

    First, we try to match the layer_name with a target
    Second, we try to match the module's name with a target
    Third, we try to map the layer_name to a list of fused module names.
        *All* component module names must match in order for a match to be
        successful. A successful match returns the first component target

    Args:
        layer_name: layer name
        module: torch.nn.Module
        targets: list of targets to match the layer against
        fused_mapping: map from fused layer names to its components
    """

    if layer_name is None:
        layer_name = ""

    matched_target = (
        _find_first_match(layer_name, targets)
        or _match_fused_layer(layer_name, targets, fused_mapping)
        or _find_first_match(module.__class__.__name__, targets, True)
    )

    return matched_target


def _find_first_match(
    value: str, targets: Iterable[str], check_contains: bool = False
) -> str | None:
    """
    Returns first element of target that matches value either
    exactly or as a regex after 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.

    Args:
        value: string to compare the list of targets against
        targets: list of targets to match the layer against
        check_contains: whether or not to do a substring match
    """

    for target in targets:
        if is_equal_or_regex_match(value, target, check_contains=check_contains):
            return target
    return None


def _match_fused_layer(
    layer_name: str,
    target_layers: Iterable[str],
    fused_mapping: Mapping[str, list[str]],
) -> str | None:
    """
    Match a fused layer name to its corresponding individual layer in
    target_layers. Returns first value in fused_mapping which matches targets

    Implements an "all" matching strategy where a fused layer matches iff
    "all" of its components match

    Args:
        layer_name: layer name
        target_layers: list of targets to match the layer against
        fused_mapping: map from fused layer names to its components

    Examples:
        layer_name = "model.layers.0.self_attn.qkv_proj"
        target_layers = ["model.layers.0.self_attn.q_proj",
                        "model.layers.0.self_attn.k_proj",
                        "model.layers.0.self_attn.v_proj"]
    """
    # find layer_name in mapping
    fused = next((key for key in fused_mapping if layer_name.endswith(key)), None)
    if fused is None:
        return None

    # expand path of unfused components
    unfused_paths = [
        layer_name.replace(fused, unfused) for unfused in fused_mapping[fused]
    ]

    # for each unfused component, find a match in targets
    unfused_matches: list[str | None] = []
    for unfused in unfused_paths:
        for target in target_layers:
            if is_equal_or_regex_match(unfused, target):
                unfused_matches.append(target)
                break
        else:
            unfused_matches.append(None)

    return unfused_matches[0] if all(unfused_matches) else None
