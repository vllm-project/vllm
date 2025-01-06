from .base import MultiModalPlaceholderMap, MultiModalPlugin
from .hasher import MultiModalHashDict, MultiModalHasher
from .inputs import (BatchedTensorInputs, ModalityData, MultiModalDataBuiltins,
                     MultiModalDataDict, MultiModalKwargs,
                     MultiModalPlaceholderDict, NestedTensors)
from .registry import MultiModalRegistry

MULTIMODAL_REGISTRY = MultiModalRegistry()
"""
The global :class:`~MultiModalRegistry` is used by model runners to
dispatch data processing according to its modality and the target model.

See also:
    :ref:`input-processing-pipeline`
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
    "MultiModalPlugin",
    "NestedTensors",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
]
