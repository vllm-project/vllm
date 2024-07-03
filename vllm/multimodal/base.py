import sys
from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from typing import (Any, Callable, Dict, List, Optional, Type, TypedDict,
                    TypeVar, Union)

import torch
import torch.types
from PIL import Image
from torch import nn

from vllm.config import ModelConfig
from vllm.inputs import InputContext
from vllm.logger import init_logger

logger = init_logger(__name__)

BatchedTensors = Union[torch.Tensor, List[torch.Tensor]]
"""
If each input tensor in the batch has the same size, this is a single batched
tensor; otherwise, this is a list of tensors with one element per batch.
"""

if sys.version_info < (3, 9):
    # UserDict cannot be subscripted
    class _MultiModalInputsBase(UserDict):
        pass
else:

    class _MultiModalInputsBase(UserDict[str, torch.Tensor]):
        pass


class MultiModalInputs(_MultiModalInputsBase):
    """
    A dictionary that represents the keyword arguments to
    :meth:`~torch.nn.Module.forward`.
    """

    @staticmethod
    def try_concat(
        tensors: List[torch.Tensor],
        *,
        device: torch.types.Device,
    ) -> BatchedTensors:
        # Avoid initializing CUDA too early
        import torch

        unbatched_shape = tensors[0].shape[1:]

        for tensor in tensors:
            if tensor.shape[1:] != unbatched_shape:
                return [
                    tensor.squeeze(0).to(device=device) for tensor in tensors
                ]

        return torch.cat(tensors, dim=0).to(device=device)

    @staticmethod
    def batch(
        inputs_list: List["MultiModalInputs"],
        device: torch.types.Device,
    ) -> Dict[str, BatchedTensors]:
        """Batch multiple inputs together into a dictionary."""
        if len(inputs_list) == 0:
            return {}

        keys = inputs_list[0].keys()

        item_lists: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for inputs in inputs_list:
            if inputs.keys() != keys:
                msg = f"Inputs do not share the same keys ({keys})"
                raise ValueError(msg)

            for k, v in inputs.items():
                item_lists[k].append(v)

        return {
            k: MultiModalInputs.try_concat(item_list, device=device)
            for k, item_list in item_lists.items()
        }


class MultiModalDataBuiltins(TypedDict, total=False):
    image: Image.Image


MultiModalDataDict = Union[MultiModalDataBuiltins, Dict[str, Any]]
"""
A dictionary containing an item for each modality type to input.

The data belonging to each modality is converted into keyword arguments 
to the model by the corresponding mapper. By default, the mapper of 
the corresponding plugin with the same modality key is applied.
"""

MultiModalInputMapper = Callable[[InputContext, object], MultiModalInputs]
"""
Return a dictionary to be passed as keyword arguments to
:meth:`~torch.nn.Module.forward`. This is similar in concept to tokenizers
and processors in HuggingFace Transformers.

If the data is not supported, throw :exc:`TypeError`.
"""

MultiModalTokensCalc = Union[int, Callable[[InputContext], int]]
"""
Calculate the maximum number of multimodal tokens input to the language
model. This does not include the tokens that correspond to the input text.
"""

N = TypeVar("N", bound=Type[nn.Module])


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
        self._input_mappers: Dict[Type[nn.Module], MultiModalInputMapper] = {}
        self._max_mm_tokens: Dict[Type[nn.Module], MultiModalTokensCalc] = {}

    @abstractmethod
    def get_data_key(self) -> str:
        """
        Get the data key corresponding to the modality.
        """
        raise NotImplementedError

    @abstractmethod
    def _default_input_mapper(self, ctx: InputContext,
                              data: object) -> MultiModalInputs:
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
        this plugin (see :meth:`get_data_key`), the provided function is
        invoked to transform the data into a dictionary of model inputs.
        If `None` is provided, then the default input mapper is used instead.

        See also:
            :ref:`input_processing_pipeline`
            :ref:`adding_a_new_multimodal_model`
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
                  data: object) -> MultiModalInputs:
        """
        Apply an input mapper to a data passed
        to the model, transforming the data into a dictionary of model inputs.

        The model is identified by ``model_config``.

        Raises:
            TypeError: If the data type is not supported.

        See also:
            :ref:`adding_a_new_multimodal_model`
        """
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)

        mapper = self._input_mappers.get(model_cls)
        if mapper is None:
            raise KeyError(f"No input mapper in {self} is registered for "
                           f"model class {model_cls.__name__}.")

        return mapper(InputContext(model_config), data)

    def register_max_multimodal_tokens(
        self,
        max_mm_tokens: MultiModalTokensCalc,
    ):
        """
        Register the maximum number of multi-modal tokens input to the
        language model for a model class.

        See also:
            :ref:`adding_a_new_multimodal_model`
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._input_mappers:
                logger.warning(
                    "Model class %s already calculates maximum number of "
                    "tokens in %s. It is overwritten by the new one.",
                    model_cls, self)

            self._max_mm_tokens[model_cls] = max_mm_tokens

            return model_cls

        return wrapper

    def get_max_multimodal_tokens(
        self,
        model_config: ModelConfig,
        default: int,
    ):
        """
        Get the maximum number of multi-modal tokens
        for profiling the memory usage of a model.

        If this registry is not applicable to the model,
        instead return the ``default`` value.

        The model is identified by ``model_config``.

        See also:
            :ref:`adding_a_new_multimodal_model`
        """
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)

        if model_cls not in self._input_mappers:
            return default

        max_mm_tokens = self._max_mm_tokens.get(model_cls)
        if max_mm_tokens is None:
            raise KeyError(f"No maximum number of multi-modal tokens is given "
                           f"for model class {model_cls.__name__} in {self}.")

        if isinstance(max_mm_tokens, int):
            return max_mm_tokens

        return max_mm_tokens(InputContext(model_config))
