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


def __getattr__(name: str):
    import warnings

    if name == "MultiModalInputs":
        msg = ("MultiModalInputs has been renamed to MultiModalKwargs. "
               "The original name will take another meaning in an upcoming "
               "version.")

        warnings.warn(DeprecationWarning(msg), stacklevel=2)

        return MultiModalKwargs

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
