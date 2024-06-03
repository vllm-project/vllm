from typing import TYPE_CHECKING, Type, TypeVar

from vllm.logger import init_logger
from vllm.multimodal.base import MultiModalData
from vllm.multimodal.registry import MULTIMODAL_REGISTRY, MultiModalRegistry

if TYPE_CHECKING:
    from torch import nn

logger = init_logger(__name__)

D = TypeVar("D", bound=MultiModalData)
N = TypeVar("N", bound=Type["nn.Module"])


class InputRegistry:
    """
    This registry is used by model runners to dispatch data processing
    according to its modality and the target model.
    """

    def __init__(self,
                 *,
                 multimodal_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
                 ) -> None:
        self._multimodal_registry = multimodal_registry
    
    @property
    def MULTIMODAL(self) -> MultiModalRegistry:
        """Access the registry for processing multimodal inputs."""
        return self._multimodal_registry



INPUT_REGISTRY = InputRegistry()
"""The global :class:`~InputRegistry` which is used by model runners."""
