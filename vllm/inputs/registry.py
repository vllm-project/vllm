import functools
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type,
                    TypeVar)

from torch import nn
from transformers import PretrainedConfig

from vllm.logger import init_logger

from .data import LLMInputs

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VisionLanguageConfig
    from vllm.multimodal import MultiModalData
    from vllm.sequence import SequenceData

logger = init_logger(__name__)

C = TypeVar("C", bound=PretrainedConfig)


@dataclass(frozen=True)
class InputContext:
    """
    Contains information about the model which may be used to
    modify the inputs.
    """

    model_config: "ModelConfig"
    """The configuration of the model."""

    def get_multimodal_config(self) -> "VisionLanguageConfig":
        """
        Get the multimodal configuration of the model.

        Raises:
            ValueError: If the model is not multimodal.
        """

        multimodal_config = self.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError("No multimodal config found")

        return multimodal_config

    def get_hf_config(self, hf_config_type: Type[C]) -> C:
        """
        Get the HuggingFace configuration
        (:class:`transformers.PretrainedConfig`) of the model,
        additionally checking its type.

        Raises:
            ValueError: If the model is not of the specified type.
        """

        hf_config = self.model_config.hf_config
        if not isinstance(hf_config, hf_config_type):
            raise TypeError("Invalid type of HuggingFace config. "
                            f"Expected type: {hf_config_type}, but "
                            f"found type: {type(hf_config)}")

        return hf_config


N = TypeVar("N", bound=Type[nn.Module])

DummyDataFactory = Callable[[InputContext, int],
                            Tuple["SequenceData", Optional["MultiModalData"]]]
"""
Create dummy data to be inputted into the model.

Note:
    :data:`InputProcessor` is not applied to the dummy data.
"""

InputProcessor = Callable[[InputContext, LLMInputs], LLMInputs]
"""Preprocess the inputs to the model."""


class InputRegistry:
    """
    A registry to dispatch data processing
    according to the target model.
    """

    def __init__(self) -> None:
        self._dummy_factories_by_model_type: Dict[Type[nn.Module],
                                                  DummyDataFactory] = {}
        self._input_processors_by_model_type: Dict[Type[nn.Module],
                                                   InputProcessor] = {}

    def _default_dummy_data_factory(
        self,
        ctx: InputContext,
        seq_len: int,
    ) -> Tuple["SequenceData", Optional["MultiModalData"]]:
        """
        The default dummy data factory represents the longest possible text
        that can be inputted to the model.

        Note:
            :data:`InputProcessor` is not applied to the dummy data.
        """
        # Avoid circular import
        from vllm.sequence import SequenceData

        dummy_seq_data = SequenceData([0] * seq_len)
        dummy_multi_modal_data = None

        return dummy_seq_data, dummy_multi_modal_data

    def register_dummy_data(self, factory: DummyDataFactory):
        """
        Register a dummy data factory to a model class.

        During memory profiling, the provided function is invoked to create
        dummy data to be inputted into the model. The resulting memory usage
        should be an upper bound of what the model would use at inference time.
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._dummy_factories_by_model_type:
                logger.warning(
                    "Model class %s already has dummy data "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._dummy_factories_by_model_type[model_cls] = factory

            return model_cls

        return wrapper

    def dummy_data_for_profiling(self, model_config: "ModelConfig",
                                 seq_len: int):
        """
        Create dummy data for profiling the memory usage of a model.

        The model is identified by ``model_config``.

        TODO: Add guide [ref: PR #5276]
        """
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        dummy_factory = self._dummy_factories_by_model_type \
            .get(model_cls, self._default_dummy_data_factory)

        return dummy_factory(InputContext(model_config), seq_len)

    def _default_input_processor(self, ctx: InputContext,
                                 inputs: LLMInputs) -> LLMInputs:
        """The default input processor is a no-op."""
        return inputs

    def register_input_processor(self, processor: InputProcessor):
        """
        Register an input processor to a model class.

        The provided function is invoked on each input to the model. This
        happens before :meth:`~vllm.multimodal.MultiModalRegistry.map_input`.

        See also:
            :ref:`input_processing_pipeline`
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._input_processors_by_model_type:
                logger.warning(
                    "Model class %s already has input processor "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._input_processors_by_model_type[model_cls] = processor

            return model_cls

        return wrapper

    def process_input(self, model_config: "ModelConfig",
                      inputs: LLMInputs) -> LLMInputs:
        """
        Apply an input processor to an instance of model inputs.

        The model is identified by ``model_config``.

        See also:
            :ref:`input_processing_pipeline`
        """
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)

        processor = self._input_processors_by_model_type \
            .get(model_cls, self._default_input_processor)

        return processor(InputContext(model_config), inputs)

    def create_input_processor(self, model_config: "ModelConfig"):
        """
        Create an input processor (see :meth:`process_input`) for a
        specific model.
        """
        return functools.partial(self.process_input, model_config)
