from abc import ABC, abstractmethod
from typing import (TYPE_CHECKING, Callable, Dict, Generic, Optional, Type,
                    TypeVar)

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch
    from torch import nn

logger = init_logger(__name__)


class MultiModalData:
    """To add a new data type, add a new file under `multimodal` directory.

    In this new file, create a subclass of
    :class:`~vllm.multimodal.base.MultiModalData`
    and :class:`~vllm.multimodal.base.MultiModalPlugin`.

    Finally, update `~vllm.multimodal.registry.MultiModalRegistry`
    with new methods to interact with the newly defined registry.
    """
    pass


D = TypeVar("D", bound=MultiModalData)
N = TypeVar("N", bound=Type["nn.Module"])

MultiModalInputProcessor = Callable[[D, ModelConfig, VisionLanguageConfig],
                                    Dict[str, "torch.Tensor"]]
"""Returns a dictionary which are passed as keyword arguments to
:meth:`torch.nn.Module.forward`.
"""


class MultiModalPlugin(ABC, Generic[D]):

    @classmethod
    def get_model_cls(cls, model_config: ModelConfig) -> Type["nn.Module"]:
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        return get_model_architecture(model_config)[0]

    def __init__(self) -> None:
        self._input_processors: Dict[Type["nn.Module"],
                                     MultiModalInputProcessor[D]] = {}

    @abstractmethod
    def get_data_type(self) -> Type[D]:
        raise NotImplementedError

    @abstractmethod
    def _default_input_processor(
            self, data: D, model_config: ModelConfig,
            vlm_config: VisionLanguageConfig) -> Dict[str, "torch.Tensor"]:
        """Returns a dictionary which are passed as keyword arguments to
        :meth:`torch.nn.Module.forward`.
        """
        raise NotImplementedError

    def register_input_processor(self,
                                 processor: Optional[
                                     MultiModalInputProcessor[D]] = None):

        def wrapper(model_cls: N) -> N:
            if model_cls in self._input_processors:
                logger.warning(
                    f"Model class {model_cls} already has an input processor "
                    f"registered to {self}. It is overwritten by the new one.")

            self._input_processors[model_cls] = processor \
                or self._default_input_processor

            return model_cls

        return wrapper

    def process_input(
            self, data: D, model_config: ModelConfig,
            vlm_config: VisionLanguageConfig) -> Dict[str, "torch.Tensor"]:
        model_cls = self.get_model_cls(model_config)

        processor = self._input_processors.get(model_cls)
        if processor is None:
            raise KeyError(f"No input processor in {self} is registered for "
                           f"model class {model_cls.__name__}.")

        return processor(data, model_config, vlm_config)
