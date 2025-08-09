# flake8: noqa
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from typing import Optional

import regex as re
import torch
from compressed_tensors import CompressionFormat

__all__ = ["is_activation_quantization_format", "is_match"]


def is_activation_quantization_format(format: str) -> bool:
    _ACTIVATION_QUANTIZATION_FORMATS = [
        CompressionFormat.naive_quantized.value,
        CompressionFormat.int_quantized.value,
        CompressionFormat.float_quantized.value,
        CompressionFormat.nvfp4_pack_quantized.value
    ]
    return format in _ACTIVATION_QUANTIZATION_FORMATS


# The code below is copied from `compressed_tensors/utils/match.py`
# and will be substituted after the release of `compressed-tensors>=0.11`

FusedMappping = Mapping[str, Iterable[str]]


def is_match(
    name: str,
    module: torch.nn.Module,
    target: str,
    fused: Optional[FusedMappping] = None,
) -> bool:
    """
    Returns true if either module name or module parent classes match against target
    and the module is not an internal module. The name and module may refer to a fused
    module defined by vLLM. In these cases, a `fused` mapping must be provided.

    For example, in `vllm/model_executor/models/llama.py`:
    ```python
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }
    ```

    :param name: name of module
    :param module: module to match
    :param target: target which matches name or module, potentially contains regex
    :fused: optional mapping from suffixes of fused modules to the suffixes of their
        corresponding shards
    """
    return (_match_name(name, target, fused) or _match_class(module, target))


def _match_name(name: str,
                target: str,
                fused: Optional[FusedMappping] = None) -> bool:
    """
    Returns true if target string begins with "re:" and regex matches or if target
    string exactly matches name. If the name refers to a fused module defined by vLLM,
    a `fused` mapping must be provided.

    :param name: name of module
    :param target: target name, potentially contains regex
    :fused: optional mapping from suffixes of fused modules to the suffixes of their
        corresponding shards
    """
    if fused is not None:
        for fused_suffix in fused:
            if name.endswith(fused_suffix):
                name_stripped = name.removesuffix(fused_suffix)
                return any(
                    _match_name(name_stripped + shard_suffix, target)
                    for shard_suffix in fused[fused_suffix])

    if target.startswith("re:"):
        return re.match(target.removeprefix("re:"), name) is not None
    else:
        return target == name


def _match_class(module: torch.nn.Module, target: str) -> bool:
    """
    Returns true if any torch parent class names match the target string exactly.
    A special exception is made for vllm's `LinearBase` class which matches `Linear`

    :param module: module to match
    :param target: target which matches name or module
    """
    # will never match against a regex pattern since `:` is not allowed in class names
    return any(
        (issubclass(cls, torch.nn.Module) and (cls.__name__ == target or (
            cls.__name__ == "LinearBase" and target == "Linear")))
        for cls in module.__class__.__mro__)
