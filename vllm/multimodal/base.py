from abc import ABC, abstractmethod
from typing import (TYPE_CHECKING, Any, Callable, Dict, Optional, Type,
                    TypedDict, TypeVar, Union)

from vllm.config import ModelConfig
from vllm.inputs import InputContext
from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch
    from PIL import Image
    from torch import nn

logger = init_logger(__name__)

N = TypeVar("N", bound=Type["nn.Module"])


class MultiModalDataBuiltins(TypedDict, total=False):
    image: "Image.Image"


MultiModalDataDict = Union[MultiModalDataBuiltins, Dict[str, Any]]
"""
A dictionary containing an item for each modality type to input.

The data belonging to each modality is converted into keyword arguments 
to the model by the corresponding mapper. By default, the mapper of 
the corresponding plugin with the same modality key is applied.
"""

MultiModalInputMapper = Callable[[InputContext, object], Dict[str,
                                                              "torch.Tensor"]]
"""Return a dictionary to be passed as keyword arguments to
:meth:`~torch.nn.Module.forward`. This is similar in concept to tokenizers
and processors in HuggingFace Transformers."""


class MultiModalPlugin(ABC):
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
                                  MultiModalInputMapper] = {}

    @abstractmethod
    def get_data_key(self) -> str:
        """
        Get the data key corresponding to the modality.
        """
        raise NotImplementedError

    @abstractmethod
    def _default_input_mapper(self, ctx: InputContext,
                              data: object) -> Dict[str, "torch.Tensor"]:
        """Return a dictionary to be passed as keyword arguments to
        :meth:`~torch.nn.Module.forward`. This is similar in concept to
        tokenizers and processors in HuggingFace Transformers.
        """
        raise NotImplementedError

    def register_input_mapper(
        self,
        mapper: Optional[MultiModalInputMapper] = None,
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
                  data: object) -> Dict[str, "torch.Tensor"]:
        """
        Apply an input mapper to a data passed
        to the model, transforming the data into a dictionary of model inputs.

        If the data is not something that the mapper expects, throws TypeError.

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
