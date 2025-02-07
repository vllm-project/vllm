# SPDX-License-Identifier: Apache-2.0

import re
from typing import Iterable, Optional

from compressed_tensors import CompressionFormat
from torch.nn import Module

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    FUSED_LAYER_NAME_MAPPING)


def is_activation_quantization_format(format: str) -> bool:
    _ACTIVATION_QUANTIZATION_FORMATS = [
        CompressionFormat.naive_quantized.value,
        CompressionFormat.int_quantized.value,
        CompressionFormat.float_quantized.value,
    ]
    return format in _ACTIVATION_QUANTIZATION_FORMATS


def should_ignore_layer(layer_name: Optional[str],
                        ignore: Iterable[str]) -> bool:
    if layer_name is None:
        return False

    # layer_name = model.layers.0.self_attn.qkv_proj
    # proj_name = qkv_proj
    proj_name = layer_name.split(".")[-1]

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    if proj_name in FUSED_LAYER_NAME_MAPPING and layer_name not in ignore:
        shard_proj_names = FUSED_LAYER_NAME_MAPPING[proj_name]

        # Convert fused_name --> [shard_names]
        shard_names = [
            layer_name.replace(proj_name, shard_proj_name)
            for shard_proj_name in shard_proj_names
        ]

        # Layer should be ignored if shards are ignored.
        should_ignore_layer = None
        for shard_name in shard_names:
            should_ignore_shard = check_equal_or_regex_match(
                layer_name=shard_name, targets=ignore)

            # If shard_idx=0, set layer ignore to match shard.
            if should_ignore_layer is None:
                should_ignore_layer = should_ignore_shard

            # If shard_idx=1+ confirm scheme matches prior shards.
            elif should_ignore_shard != should_ignore_layer:
                raise ValueError(f"Found a different quantization schemes for "
                                 f"{shard_proj_names} in {layer_name}. vLLM "
                                 "requires all to use the same scheme.")

    # Unfused layers like down_proj and o_proj will match
    # the safetensors checkpoint already.
    else:
        should_ignore_layer = check_equal_or_regex_match(layer_name=layer_name,
                                                         targets=ignore)

    assert should_ignore_layer is not None
    return should_ignore_layer


def check_equal_or_regex_match(layer_name: str,
                               targets: Iterable[str]) -> bool:
    """
    Checks whether a layer_name is exactly equal or a regex match for
    if target starts with 're:' to any target in list.
    """
    for target in targets:
        if _is_equal_or_regex_match(layer_name, target):
            return True
    return False


def _handle_fused_layers(func):
    """
    Decorator to handle fused layers by mapping vllm fused layer names
    to their corresponding unfused layer names for quantization/pruning schemes.
    """
    # fused_layer_name -> unfused_layer_name
    fused_layer_map = {
        "qkv_proj": "q_proj",
        "gate_up_proj": "up_proj",
    }

    def fused_layer_handler(layer_name: Optional[str], module: Module,
                            targets: Iterable[str]) -> Optional[str]:
        """
        Wrapper function specifically designed to support the
        find_matched_target function.

        It handles cases where the provided layer name corresponds to a
        fused layer in vllm, mapping it to its equivalent unfused layer name
        based on the predefined fused_layer_map. If the original layer name
        raises a ValueError in the wrapped function, this handler
        will attempt to resolve the issue by substituting with unfused
        layer name.

        :param layer_name: Name of the layer, which may be fused.
        :param module: An instance of torch.nn.Module.
        :param targets: A list of target names or patterns to match.
        :return: The result of the wrapped find_matched_target function with
            the resolved layer name.
        :raises ValueError: If the layer name cannot be resolved to a 
            valid target.
        """
        try:
            return func(layer_name, module, targets)
        except ValueError:
            if layer_name is None:
                layer_name = ""
            parent_name, fused_proj_name = layer_name.rsplit(".", 1)
            unfused_proj_name = fused_layer_map.get(fused_proj_name,
                                                    fused_proj_name)
            new_layer_name = f"{parent_name}.{unfused_proj_name}"
            return func(new_layer_name, module, targets)

    return fused_layer_handler


@_handle_fused_layers
def find_matched_target(layer_name: Optional[str], module: Module,
                        targets: Iterable[str]) -> str:
    """
    Helper function to look up which "target" in the compressed-tensors
    config that a layer corresponds to.

    Recall that a compressed-tensors configs has a concept of
    config_groups, where each layer can be quantized with with a different
    scheme.

    targets in each config_group will be a list of either layer names
    (or regexes corresponding to layer names) or names of torch Modules.

    First, we try to match the layer_name with a target
    Second, we try to match the module's name with a target

    :param layer_name: layer name
    :param module: torch.nn.Module
    :param targets: list of targets to match the layer against
    """

    if layer_name is None:
        layer_name = ""

    matched_target = (_find_first_match(layer_name, targets)
                      or _find_first_match(module.__class__.__name__, targets,
                                           True)
                      or _match_fused_layer(layer_name, targets))

    if matched_target is None:
        raise ValueError(
            f"Unable to find matching target for {layer_name} in the "
            "compressed-tensors config.")

    return matched_target


def _find_first_match(value: str,
                      targets: Iterable[str],
                      check_contains: bool = False) -> Optional[str]:
    """
    Returns first element of target that matches value either
    exactly or as a regex after 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.

    :param value: string to compare the list of targets against
    :param targets: list of targets to match the layer against
    :param check_contains: whether or not to do a substring match
    """

    for target in targets:
        if _is_equal_or_regex_match(value,
                                    target,
                                    check_contains=check_contains):
            return target
    return None


def _is_equal_or_regex_match(value: str,
                             target: str,
                             check_contains: bool = False) -> bool:
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


def _match_fused_layer(layer_name: str,
                       target_layers: Iterable[str]) -> Optional[str]:
    """
    Match a fused layer name to its corresponding individual layer in 
    target_layers.

    Examples:
        layer_name = "model.layers.0.self_attn.qkv_proj"
        target_layers = ["model.layers.0.self_attn.q_proj",
                        "model.layers.0.self_attn.k_proj",
                        "model.layers.0.self_attn.v_proj"]
    """
    # Split into parent path and layer type
    # e.g., "model.layers.0.self_attn" and "qkv_proj"
    parent_path = ".".join(layer_name.split(".")[:-1])
    layer_type = layer_name.split(".")[-1]

    if layer_type not in FUSED_LAYER_NAME_MAPPING:
        return None

    possible_layer_types = FUSED_LAYER_NAME_MAPPING[layer_type]

    # Look for a target layer that:
    # 1. Has the same parent path
    # 2. Ends with one of the possible individual layer types
    for target in target_layers:
        is_same_parent = parent_path in target
        is_matching_type = any(type_suffix in target
                               for type_suffix in possible_layer_types)

        if is_same_parent and is_matching_type and all(
            (f"{parent_path}.{type_suffix}" in target_layers)
                for type_suffix in possible_layer_types):
            return target

    return None
