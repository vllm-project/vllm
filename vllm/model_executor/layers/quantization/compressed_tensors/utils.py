# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Optional

import torch
from compressed_tensors import CompressionFormat, InternalModule
from compressed_tensors.utils.match import (FusedMappping, _match_class,
                                            _match_name)


def is_activation_quantization_format(format: str) -> bool:
    _ACTIVATION_QUANTIZATION_FORMATS = [
        CompressionFormat.naive_quantized.value,
        CompressionFormat.int_quantized.value,
        CompressionFormat.float_quantized.value,
        CompressionFormat.nvfp4_pack_quantized.value
    ]
    return format in _ACTIVATION_QUANTIZATION_FORMATS


# TODO (@kylesayrs): Replace with CT import on next CT release
def match_targets(
    name: str,
    module: torch.nn.Module,
    targets: Optional[Iterable[str]],
    fused: Optional[FusedMappping] = None,
) -> list[str]:
    """
    Returns the targets that match the given name and module. Outputs are
    ordered by type: exact name match, regex name match, class name match

    :param name: the name of the module
    :param module: the module to match
    :param targets: the target strings, potentially containing "re:" prefixes
    :fused: optional mapping from suffixes of fused modules to the suffixes of
        their corresponding shards. See
        `compressed_tensors.utils.match.is_match`
    :return: the targets that match the given name and module
    """
    targets = targets or []

    if isinstance(module, InternalModule):
        return []

    # The order of the output `matches` list matters, the are arranged from most
    # specific to least specific, and this order will be used when merging
    # configs.
    # The entries are sorted in the following order:
    #     1. matches on exact strings
    #     2. matches on regex patterns
    #     3. matches on module names

    targets = sorted(targets, key=lambda x: ("re:" in x, x))
    matched_targets = []
    for target in targets:
        if _match_name(name, target, fused=fused):
            matched_targets.append(target)

    for target in targets:
        if _match_class(module, target) and target not in matched_targets:
            matched_targets.append(target)

    return matched_targets