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
    """
    Base class that contains multi-modal data.

    To add a new modality, add a new file under ``multimodal`` directory.

    In this new file, subclass :class:`~MultiModalData` and
    :class:`~MultiModalPlugin`.

    Finally, register the new plugin to
    :const:`vllm.multimodal.MULTIMODAL_REGISTRY`.
    This enables models to call :meth:`MultiModalRegistry.register_input` for
    the new modality.
    """
    pass


D = TypeVar("D", bound=MultiModalData)
N = TypeVar("N", bound=Type["nn.Module"])

MultiModalInputProcessor = Callable[[D, ModelConfig, VisionLanguageConfig],
                                    Dict[str, "torch.Tensor"]]
"""Return a dictionary to be passed as keyword arguments to
:meth:`torch.nn.Module.forward`. This is similar in concept to tokenizers
and processors in HuggingFace Transformers."""


class MultiModalPlugin(ABC, Generic[D]):
    """
    Base class that defines data processing logic for a specific modality.

    In particular, we adopt a registry pattern to dispatch data processing
    according to the model being used (considering that different models may
    process the same data differently). This registry is in turn used by
    :class:`~MultiModalRegistry` which acts at a higher level
    (i.e., the modality of the data).
    """

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
        """
        Get the modality (subclass of :class:`~MultiModalData`) served by
        this plugin.
        """
        raise NotImplementedError

    @abstractmethod
    def _default_input_processor(
            self, data: D, model_config: ModelConfig,
            vlm_config: VisionLanguageConfig) -> Dict[str, "torch.Tensor"]:
        """Return a dictionary to be passed as keyword arguments to
        :meth:`torch.nn.Module.forward`. This is similar in concept to
        tokenizers and processors in HuggingFace Transformers.
        """
        raise NotImplementedError

    def register_input_processor(self,
                                 processor: Optional[
                                     MultiModalInputProcessor[D]] = None):
        """
        Register an input processor to a model class.
        
        When the model receives input data that matches the modality served by
        this plugin (see :meth:`get_data_type`), the provided input processor is
        applied to preprocess the data. If `None` is provided, then the default
        input processor is applied instead.
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._input_processors:
                logger.warning(
                    "Model class %s already has an input processor "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._input_processors[model_cls] = processor \
                or self._default_input_processor

            return model_cls

        return wrapper

    def process_input(
            self, data: D, model_config: ModelConfig,
            vlm_config: VisionLanguageConfig) -> Dict[str, "torch.Tensor"]:
        """
        Apply an input processor to a :class:`~MultiModalData` instance passed
        to the model.
        
        The model is identified by ``model_config``. ``vlm_config`` is
        for compatibility purposes and may be merged into ``model_config``
        in the near future.
        """
        model_cls = self.get_model_cls(model_config)

        processor = self._input_processors.get(model_cls)
        if processor is None:
            raise KeyError(f"No input processor in {self} is registered for "
                           f"model class {model_cls.__name__}.")

        return processor(data, model_config, vlm_config)
