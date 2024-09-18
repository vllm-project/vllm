import re
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Union

from pydantic import BaseModel, Field, field_validator
from torch.nn import Module

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    FUSED_LAYER_NAME_MAPPING)


class CompressionFormat(Enum):
    dense = "dense"
    sparse_bitmask = "sparse-bitmask"
    naive_quantized = "naive-quantized"
    float_quantized = "float-quantized"
    int_quantized = "int-quantized"
    pack_quantized = "pack-quantized"
    marlin_24 = "marlin-24"


class QuantizationType(str, Enum):
    """
    Enum storing quantization type options
    """

    INT = "int"
    FLOAT = "float"


class QuantizationStrategy(str, Enum):
    """
    Enum storing quantization strategy options
    """

    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"
    TOKEN = "token"


class ActivationOrdering(str, Enum):
    """
    Enum storing strategies for activation ordering

    Group: reorder groups and weight\n
    Weight: only reorder weight, not groups. Slightly lower latency and
    accuracy compared to group actorder\n
    """

    GROUP = "group"
    WEIGHT = "weight"


class QuantizationArgs(BaseModel):
    """
    User facing arguments used to define a quantization config 
    for weights or activations

    :param num_bits: quantization bit depth
    :param type: dtype to quantized to, either int or float
    :param symmetric: whether or not quantization scale is symmetric
    :param strategy: string determining the scope of scale/zero-point to apply
    :param group_size: group length to use for the group strategy
    :param block_structure: 2d block structure to use for the block 
    strategy, must be of the format "2x4", "8x16", etc.
    :param dynamic: set True to perform dynamic quantization -
        values will not be calibrated during calibration phase, 
        instead during inference new quantization ranges will be 
        observed with every sample. Defaults to False for static
        quantization. Note that enabling dynamic quantization 
        will change the default observer to a memoryless one
    :param actorder: whether to apply group quantization in decreasing order of
        activation. Defaults to None for arbitrary ordering
    """

    num_bits: int = 8
    type: QuantizationType = QuantizationType.INT
    symmetric: bool = True
    group_size: Optional[int] = None
    strategy: Optional[QuantizationStrategy] = None
    block_structure: Optional[str] = None
    dynamic: bool = False
    actorder: Union[ActivationOrdering, bool, None] = None
    observer: str = Field(
        default="minmax",
        description=("The class to use to compute the quantization param - "
                     "scale and zero-point'"),
    )
    observer_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=
        ("optional dict of kwargs to be passed directly to torch quantization "
         "Observers constructor excluding quantization range or symmetry"),
    )

    @field_validator("actorder", mode="before")
    def validate_actorder(cls, value) -> Optional[ActivationOrdering]:
        if isinstance(value, bool):
            return ActivationOrdering.GROUP if value else None

        if isinstance(value, str):
            return ActivationOrdering(value.lower())

        return value


def is_activation_quantization_format(format: str) -> bool:
    _ACTIVATION_QUANTIZATION_FORMATS = [
        CompressionFormat.naive_quantized.value,
        CompressionFormat.int_quantized.value,
        CompressionFormat.float_quantized.value
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
    if proj_name in FUSED_LAYER_NAME_MAPPING:
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
                                           True))

    if matched_target is None:
        raise ValueError(f"Unable to find matching target for {module} in the "
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


def get_compressed_tensors_cache_scale(name: str) -> Optional[str]:
    """
    Check whether the param name matches the format for k/v cache scales
    in compressed-tensors. If this is the case, return its equivalent
    param name expected by vLLM

    :param name: param name
    :return: matching param name for KV cache scale in vLLM
    """
    if name.endswith(".output_scale") and ".k_proj" in name:
        return name.replace(".k_proj.output_scale", ".attn.k_scale")
    if name.endswith(".output_scale") and ".v_proj" in name:
        return name.replace(".v_proj.output_scale", ".attn.v_scale")
    # If no matches, return None
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
