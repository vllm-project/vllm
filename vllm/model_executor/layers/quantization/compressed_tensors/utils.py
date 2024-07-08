import re
from enum import Enum
from typing import Any, Dict, Iterable, Optional

from pydantic import BaseModel, Field
from torch.nn import Module


class CompressionFormat(Enum):
    dense = "dense"
    sparse_bitmask = "sparse-bitmask"
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
    """

    num_bits: int = 8
    type: QuantizationType = QuantizationType.INT
    symmetric: bool = True
    group_size: Optional[int] = None
    strategy: Optional[QuantizationStrategy] = None
    block_structure: Optional[str] = None
    dynamic: bool = False
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


def find_first_name_or_class_match(
        name: str,
        module: Module,
        targets: Iterable[str],
        check_contains: bool = False) -> Optional[str]:
    """
    Helper function to map the quantization details listed in the config 
    for a given list of targets against each model layer. First uses the
    layer name to try and find a match. If no name match is found, uses
    the layer class name. Returns None otherwise.

    :param name: layer name
    :param module: torch.nn.Module
    :param targets: list of targets to match the layer against
    :param check_contains: whether or not to do a substring match
    """

    return _find_first_match(name, targets) or _find_first_match(
        module.__class__.__name__, targets, check_contains)


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
        if target.startswith("re:"):
            pattern = target[3:]
            if re.match(pattern, value):
                return target
        elif check_contains:
            if target.lower() in value.lower():
                return target
        elif target == value:
            return target
    return None
