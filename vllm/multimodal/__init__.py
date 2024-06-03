from .base import MultiModalData, MultiModalPlugin
from .registry import MultiModalRegistry

MULTIMODAL_REGISTRY = MultiModalRegistry()
"""The global :class:`~MultiModalRegistry` which is used by model runners."""

__all__ = [
    "MultiModalData", "MultiModalPlugin", "MULTIMODAL_REGISTRY",
    "MultiModalRegistry"
]
