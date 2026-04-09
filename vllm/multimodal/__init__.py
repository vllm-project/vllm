# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from .hasher import MultiModalHasher
from .inputs import BatchedTensorInputs, MultiModalKwargsItems, NestedTensors
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
    "MultiModalHasher",
    "MultiModalKwargsItems",
    "NestedTensors",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
]
