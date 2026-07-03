# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from types import MappingProxyType

import regex as re
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Module

from vllm.logger import init_logger
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

logger = init_logger(__name__)

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
    if proj_name in fused_mapping and layer_name not in ignore:
        shard_proj_names = fused_mapping[proj_name]

        # Convert fused_name --> [shard_names]
        shard_names = [
            layer_name.replace(proj_name, shard_proj_name)
            for shard_proj_name in shard_proj_names
        ]

        # Layer should be ignored if shards are ignored.
        shard_ignore_flags = [
            check_equal_or_regex_match(layer_name=shard_name, targets=ignore)
            for shard_name in shard_names
        ]

        if any(shard_ignore_flags) and not all(shard_ignore_flags):
            # Fused layer can't mix schemes across shards; treat as ignored.
            missing = tuple(
                name
                for name, flag in zip(shard_proj_names, shard_ignore_flags)
                if not flag
            )
            logger.warning_once(
                "Fused layer %s has mixed ignore state across shards %s "
                "(%s not in ignore); treating as ignored.",
                layer_name,
                tuple(shard_proj_names),
                missing,
            )
            should_ignore_layer = True
        else:
            should_ignore_layer = shard_ignore_flags[0] if shard_ignore_flags else False

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
        or _find_first_match(module.__class__.__name__, targets, True)
        or _match_fused_layer(layer_name, targets, fused_mapping)
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
        if _is_equal_or_regex_match(value, target, check_contains=check_contains):
            return target
    return None


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
            if _is_equal_or_regex_match(unfused, target):
                unfused_matches.append(target)
                break
        else:
            unfused_matches.append(None)

    return unfused_matches[0] if all(unfused_matches) else None
