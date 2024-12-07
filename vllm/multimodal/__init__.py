from .base import MultiModalPlaceholderMap, MultiModalPlugin
from .inputs import (BatchedTensorInputs, MultiModalData,
                     MultiModalDataBuiltins, MultiModalDataDict,
                     MultiModalKwargs, MultiModalPlaceholderDict,
                     NestedTensors)
from .registry import MultiModalRegistry

MULTIMODAL_REGISTRY = MultiModalRegistry()
"""
The global :class:`~MultiModalRegistry` is used by model runners to
dispatch data processing according to its modality and the target model.

See also:
    :ref:`input_processing_pipeline`
"""

__all__ = [
    "BatchedTensorInputs",
    "MultiModalData",
    "MultiModalDataBuiltins",
    "MultiModalDataDict",
    "MultiModalKwargs",
    "MultiModalPlaceholderDict",
    "MultiModalPlaceholderMap",
    "MultiModalPlugin",
    "NestedTensors",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
]
