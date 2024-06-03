from typing import (TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type,
                    TypeVar)

from torch import nn
from transformers import PretrainedConfig

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VisionLanguageConfig
    from vllm.multimodal import MultiModalData, MultiModalRegistry
    from vllm.sequence import SequenceData

logger = init_logger(__name__)

D = TypeVar("D", bound="MultiModalData")
N = TypeVar("N", bound=Type[nn.Module])

DummyDataFactory = Callable[[int, "ModelConfig"],
                            Tuple["SequenceData", Optional["MultiModalData"]]]
"""Create dummy data to be inputted into the model."""

C = TypeVar("C", bound=PretrainedConfig)


class DummyDataFactories:

    @classmethod
    def for_multimodal_hf(cls, hf_config_type: Type[C]):
        """Decorates a dummy data factory that uses multimodal config as well
        as a specific type of HuggingFace config.
        
        The returned function satisfies the interface of
        :data:`DummyDataFactory`, with runtime checks being made to ensure
        the validity of the inputs."""

        def wrapper(
            factory: Callable[[int, "VisionLanguageConfig", C],
                              Tuple["SequenceData",
                                    Optional["MultiModalData"]]],
        ) -> DummyDataFactory:

            def inner(
                seq_len: int,
                model_config: "ModelConfig",
            ) -> Tuple["SequenceData", Optional["MultiModalData"]]:
                multimodal_config = model_config.multimodal_config
                if multimodal_config is None:
                    raise ValueError("No multimodal config found")

                hf_config = model_config.hf_config
                if not isinstance(hf_config, hf_config_type):
                    raise TypeError("Invalid type of HuggingFace config. "
                                    f"Expected type: {hf_config_type}, but "
                                    f"received type: {type(hf_config)}")

                return factory(seq_len, multimodal_config, hf_config)

            return inner

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

    @property
    def MULTIMODAL(self) -> "MultiModalRegistry":
        """Access the registry for processing multimodal inputs."""
        return self._multimodal_registry

    def _default_dummy_data_factory(
        self,
        seq_len: int,
        model_config: "ModelConfig",
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
        seq_len: int,
        model_config: "ModelConfig",
    ):
        """Create dummy data for memory profiling."""
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        dummy_factory = self._dummy_factories_by_model_type \
            .get(model_cls, self._default_dummy_data_factory)

        return dummy_factory(seq_len, model_config)
