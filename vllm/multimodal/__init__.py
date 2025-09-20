# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from .base import MultiModalPlaceholderMap
from .hasher import MultiModalHasher
from .inputs import (BatchedTensorInputs, ModalityData, MultiModalDataBuiltins,
                     MultiModalDataDict, MultiModalKwargs,
                     MultiModalKwargsItems, MultiModalPlaceholderDict,
                     MultiModalUUIDDict, NestedTensors)
from .registry import MultiModalRegistry

MULTIMODAL_REGISTRY = MultiModalRegistry()
"""
The global [`MultiModalRegistry`][vllm.multimodal.registry.MultiModalRegistry]
is used by model runners to dispatch data processing according to the target
model.

Info:
    [mm_processing](../../../design/mm_processing.md)
"""

__all__ = [
    "BatchedTensorInputs",
    "ModalityData",
    "MultiModalDataBuiltins",
    "MultiModalDataDict",
    "MultiModalHasher",
    "MultiModalKwargs",
    "MultiModalKwargsItems",
    "MultiModalPlaceholderDict",
    "MultiModalPlaceholderMap",
    "MultiModalUUIDDict",
    "NestedTensors",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
]
