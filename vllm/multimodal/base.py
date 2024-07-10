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
    """Modality types that are predefined by vLLM."""

    image: Image.Image
    """The input image."""


MultiModalDataDict = Union[MultiModalDataBuiltins, Dict[str, Any]]
"""
A dictionary containing an item for each modality type to input.

Note:
    This dictionary also accepts modality keys defined outside
    :class:`MultiModalDataBuiltins` as long as a customized plugin is registered
    through the :class:`~vllm.multimodal.MULTIMODAL_REGISTRY`.
    Read more on that :ref:`here <adding_multimodal_plugin>`.
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
model. This does not include tokens that correspond to the input text.
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

    See also:
        :ref:`adding_multimodal_plugin`
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
        """
        Return a dictionary to be passed as keyword arguments to
        :meth:`~torch.nn.Module.forward`. This is similar in concept to
        tokenizers and processors in HuggingFace Transformers.

        If the data is not supported, throw :exc:`TypeError`.
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
            - :ref:`input_processing_pipeline`
            - :ref:`enabling_multimodal_inputs`
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
        Transform the data into a dictionary of model inputs using the
        input mapper registered for that model.

        The model is identified by ``model_config``.

        Raises:
            TypeError: If the data type is not supported.

        See also:
            - :ref:`input_processing_pipeline`
            - :ref:`enabling_multimodal_inputs`
        """
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)

        mapper = self._input_mappers.get(model_cls)
        if mapper is None:
            raise KeyError(f"No input mapper in {self} is registered for "
                           f"model class {model_cls.__name__}.")

        return mapper(InputContext(model_config), data)

    @abstractmethod
    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        """
        Calculate the maximum number of multimodal tokens input to the language
        model. This does not include tokens that correspond to the input text.
        """
        raise NotImplementedError

    def _validate_max_multimodal_tokens(self, max_mm_tokens: int):
        if max_mm_tokens < 1:
            raise ValueError("You should set the number of tokens to a "
                             f"positive integer. Found: {max_mm_tokens}")

    def register_max_multimodal_tokens(
        self,
        max_mm_tokens: Optional[MultiModalTokensCalc] = None,
    ):
        """
        Register the maximum number of multi-modal tokens input to the
        language model for a model class.

        If `None` is provided, then the default calculation is used instead.

        See also:
            :ref:`enabling_multimodal_inputs`
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._max_mm_tokens:
                logger.warning(
                    "Model class %s already calculates maximum number of "
                    "tokens in %s. It is overwritten by the new one.",
                    model_cls, self)

            if isinstance(max_mm_tokens, int):
                self._validate_max_multimodal_tokens(max_mm_tokens)

            self._max_mm_tokens[model_cls] = max_mm_tokens \
                or self._default_max_multimodal_tokens

            return model_cls

        return wrapper

    def get_max_multimodal_tokens(self, model_config: ModelConfig) -> int:
        """
        Get the maximum number of multi-modal tokens
        for profiling the memory usage of a model.

        If this registry is not applicable to the model, `0` is returned.

        The model is identified by ``model_config``.

        See also:
            :ref:`enabling_multimodal_inputs`
        """
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)

        if model_cls not in self._input_mappers:
            return 0

        max_mm_tokens = self._max_mm_tokens.get(model_cls)
        if max_mm_tokens is None:
            raise KeyError(f"No maximum number of multi-modal tokens is given "
                           f"for model class {model_cls.__name__} in {self}.")

        if callable(max_mm_tokens):
            max_mm_tokens = max_mm_tokens(InputContext(model_config))

        self._validate_max_multimodal_tokens(max_mm_tokens)

        return max_mm_tokens
