from abc import ABC, abstractmethod
from typing import (TYPE_CHECKING, Callable, Dict, Generic, Optional, Type,
                    TypeVar)

from vllm.config import ModelConfig
from vllm.inputs import InputContext
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
    This enables models to call :meth:`MultiModalRegistry.map_input` for
    the new modality.
    """
    pass


D = TypeVar("D", bound=MultiModalData)
N = TypeVar("N", bound=Type["nn.Module"])

MultiModalInputMapper = Callable[[InputContext, D], Dict[str, "torch.Tensor"]]
"""Return a dictionary to be passed as keyword arguments to
:meth:`~torch.nn.Module.forward`. This is similar in concept to tokenizers
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

    def __init__(self) -> None:
        self._input_mappers: Dict[Type["nn.Module"],
                                  MultiModalInputMapper[D]] = {}

    @abstractmethod
    def get_data_type(self) -> Type[D]:
        """
        Get the modality (subclass of :class:`~MultiModalData`) served by
        this plugin.
        """
        raise NotImplementedError

    @abstractmethod
    def _default_input_mapper(self, ctx: InputContext,
                              data: D) -> Dict[str, "torch.Tensor"]:
        """Return a dictionary to be passed as keyword arguments to
        :meth:`~torch.nn.Module.forward`. This is similar in concept to
        tokenizers and processors in HuggingFace Transformers.
        """
        raise NotImplementedError

    def register_input_mapper(
        self,
        mapper: Optional[MultiModalInputMapper[D]] = None,
    ):
        """
        Register an input mapper to a model class.
        
        When the model receives input data that matches the modality served by
        this plugin (see :meth:`get_data_type`), the provided function is
        invoked to transform the data into a dictionary of model inputs.
        If `None` is provided, then the default input mapper is used instead.

        See also:
            :ref:`input_processing_pipeline`
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._input_mappers:
                logger.warning(
                    "Model class %s already has an input mapper "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._input_mappers[model_cls] = mapper \
                or self._default_input_mapper

            return model_cls

        return wrapper

    def map_input(self, model_config: ModelConfig,
                  data: D) -> Dict[str, "torch.Tensor"]:
        """
        Apply an input mapper to a :class:`~MultiModalData` instance passed
        to the model, transforming the data into a dictionary of model inputs.

        The model is identified by ``model_config``.

        TODO: Add guide [ref: PR #5276]
        """
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)

        mapper = self._input_mappers.get(model_cls)
        if mapper is None:
            raise KeyError(f"No input mapper in {self} is registered for "
                           f"model class {model_cls.__name__}.")

        return mapper(InputContext(model_config), data)
