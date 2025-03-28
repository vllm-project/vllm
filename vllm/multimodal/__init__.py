# SPDX-License-Identifier: Apache-2.0
from .base import MultiModalPlaceholderMap
from .hasher import MultiModalHashDict, MultiModalHasher
from .inputs import (BatchedTensorInputs, ModalityData, MultiModalDataBuiltins,
                     MultiModalDataDict, MultiModalKwargs,
                     MultiModalPlaceholderDict, NestedTensors)
from .registry import MultiModalRegistry

MULTIMODAL_REGISTRY = MultiModalRegistry()
"""
The global :class:`~MultiModalRegistry` is used by model runners to
dispatch data processing according to the target model.

See also:
    :ref:`mm-processing`
"""

__all__ = [
    "BatchedTensorInputs",
    "ModalityData",
    "MultiModalDataBuiltins",
    "MultiModalDataDict",
    "MultiModalHashDict",
    "MultiModalHasher",
    "MultiModalKwargs",
    "MultiModalPlaceholderDict",
    "MultiModalPlaceholderMap",
    "NestedTensors",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
]
