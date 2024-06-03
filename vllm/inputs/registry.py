from typing import (TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type,
                    TypeVar)
from typing_extensions import Concatenate, ParamSpec

from torch import nn
from transformers import PretrainedConfig

from vllm.logger import init_logger

from .data import LLMInputs

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VisionLanguageConfig
    from vllm.multimodal import MultiModalData, MultiModalRegistry
    from vllm.sequence import SequenceData

logger = init_logger(__name__)

D = TypeVar("D", bound="MultiModalData")
N = TypeVar("N", bound=Type[nn.Module])

DummyDataFactory = Callable[["ModelConfig", int],
                            Tuple["SequenceData", Optional["MultiModalData"]]]
"""Create dummy data to be inputted into the model."""

InputProcessor = Callable[["ModelConfig", LLMInputs], LLMInputs]
"""Preprocess the inputs to the model."""

P, R = ParamSpec("P"), TypeVar("R")
C = TypeVar("C", bound=PretrainedConfig)


def _for_hf(hf_config_type: Type[C]):
    def wrapper(
        fn: Callable[Concatenate[C, P], R],
    ) -> Callable[Concatenate["ModelConfig", P], R]:

        def inner(
            model_config: "ModelConfig",
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            hf_config = model_config.hf_config
            if not isinstance(hf_config, hf_config_type):
                raise TypeError("Invalid type of HuggingFace config. "
                                f"Expected type: {hf_config_type}, but "
                                f"received type: {type(hf_config)}")

            return fn(hf_config, *args, **kwargs)

        return inner

    return wrapper


def _for_multimodal_hf(hf_config_type: Type[C]):
    def wrapper(
        factory: Callable[Concatenate["VisionLanguageConfig", C, P], R],
    ) -> Callable[Concatenate["ModelConfig", P], R]:

        def inner(
            model_config: "ModelConfig",
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            multimodal_config = model_config.multimodal_config
            if multimodal_config is None:
                raise ValueError("No multimodal config found")

            hf_config = model_config.hf_config
            if not isinstance(hf_config, hf_config_type):
                raise TypeError("Invalid type of HuggingFace config. "
                                f"Expected type: {hf_config_type}, but "
                                f"received type: {type(hf_config)}")

            return factory(multimodal_config, hf_config, *args, **kwargs)

        return inner

    return wrapper


class DummyDataFactories:
    """Contains factories for dummy data factories."""

    @classmethod
    def for_hf(cls, hf_config_type: Type[C]):
        """
        Decorate a dummy data factory that uses a specific type of
        HuggingFace config.
        
        The returned function satisfies the interface of
        :data:`DummyDataFactory`, with runtime checks being made to ensure
        the validity of the inputs.
        """

        def wrapper(
            factory: Callable[[C, int], Tuple["SequenceData",
                                              Optional["MultiModalData"]]],
        ) -> DummyDataFactory:
            return _for_hf(hf_config_type)(factory)

        return wrapper

    @classmethod
    def for_multimodal_hf(cls, hf_config_type: Type[C]):
        """
        Decorate a dummy data factory that uses multimodal config as well
        as a specific type of HuggingFace config.
        
        The returned function satisfies the interface of
        :data:`DummyDataFactory`, with runtime checks being made to ensure
        the validity of the inputs.
        """

        def wrapper(
            factory: Callable[["VisionLanguageConfig", C, int],
                              Tuple["SequenceData",
                                    Optional["MultiModalData"]]],
        ) -> DummyDataFactory:
            return _for_multimodal_hf(hf_config_type)(factory)

        return wrapper


class InputProcessors:
    """Contains factories for input processors."""

    @classmethod
    def for_hf(cls, hf_config_type: Type[C]):
        """
        Decorate an input processor that uses a specific type of
        HuggingFace config.
        
        The returned function satisfies the interface of
        :data:`InputProcessor`, with runtime checks being made to ensure
        the validity of the inputs.
        """

        def wrapper(
            processor: Callable[[C, LLMInputs], LLMInputs],
        ) -> InputProcessor:
            return _for_hf(hf_config_type)(processor)

        return wrapper

    @classmethod
    def for_multimodal_hf(cls, hf_config_type: Type[C]):
        """
        Decorate an input processor that uses multimodal config as well
        as a specific type of HuggingFace config.
        
        The returned function satisfies the interface of
        :data:`InputProcessor`, with runtime checks being made to ensure
        the validity of the inputs.
        """

        def wrapper(
            processor: Callable[["VisionLanguageConfig", C, LLMInputs],
                                LLMInputs],
        ) -> InputProcessor:
            return _for_multimodal_hf(hf_config_type)(processor)

        return wrapper


class InputRegistry:
    """
    This registry is used by model runners to dispatch data processing
    according to its modality and the target model.
    """

    def __init__(self, *, multimodal_registry: "MultiModalRegistry") -> None:
        self._multimodal_registry = multimodal_registry

        self._dummy_factories_by_model_type: Dict[Type[nn.Module],
                                                  DummyDataFactory] = {}
        self._input_processors_by_model_type: Dict[Type[nn.Module],
                                                   InputProcessor] = {}

    @property
    def MULTIMODAL(self) -> "MultiModalRegistry":
        """Access the registry for processing multimodal inputs."""
        return self._multimodal_registry

    def _default_dummy_data_factory(
        self,
        model_config: "ModelConfig",
        seq_len: int,
    ) -> Tuple["SequenceData", Optional["MultiModalData"]]:
        """Create dummy data to be inputted into the model."""
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

    def dummy_data_for_profiling(
        self,
        model_config: "ModelConfig",
        seq_len: int,
    ):
        """Create dummy data for memory profiling."""
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        dummy_factory = self._dummy_factories_by_model_type \
            .get(model_cls, self._default_dummy_data_factory)

        return dummy_factory(model_config, seq_len)

    def _default_input_processor(self, inputs: LLMInputs) -> LLMInputs:
        """Preprocess the inputs to the model."""
        return inputs

    def register_input_processor(self, processor: InputProcessor):
        """
        Register an input processor to a model class.

        The provided function is invoked on each input to the model. This
        happens before :meth:`~vllm.multimodal.MultiModalRegistry.map_input`.
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

        The model is identified by ``model_config``. ``vlm_config`` is
        for compatibility purposes and may be merged into ``model_config``
        in the near future.
        """
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)

        processor = self._input_processors_by_model_type.get(model_cls)
        if processor is None:
            raise KeyError(f"No input processor in {self} is registered for "
                           f"model class {model_cls.__name__}.")

        return processor(model_config, inputs)
