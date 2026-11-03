# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, cast

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config

from .processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    TimingContext,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, ObservabilityConfig
    from vllm.model_executor.models.interfaces import SupportsMultiModal

logger = init_logger(__name__)

N = TypeVar("N", bound=type["SupportsMultiModal"])
_I = TypeVar("_I", bound=BaseProcessingInfo)
_I_co = TypeVar("_I_co", bound=BaseProcessingInfo, covariant=True)


class ProcessingInfoFactory(Protocol[_I_co]):
    """Constructs a
    [`BaseMultiModalProcessor`][vllm.multimodal.processing.BaseMultiModalProcessor]
    instance from the context.
    """

    def __call__(
        self,
        ctx: InputProcessingContext,
    ) -> _I_co: ...


class DummyInputsBuilderFactory(Protocol[_I]):  # type: ignore[misc]
    """Constructs a
    [`BaseDummyInputsBuilder`][vllm.multimodal.processing.BaseDummyInputsBuilder]
    instance from the context.
    """

    def __call__(self, info: _I) -> BaseDummyInputsBuilder[_I]: ...


class MultiModalProcessorFactory(Protocol[_I]):  # type: ignore[misc]
    """Constructs a
    [`BaseMultiModalProcessor`][vllm.multimodal.processing.BaseMultiModalProcessor]
    instance from the context.
    """

    def __call__(
        self,
        info: _I,
        dummy_inputs: BaseDummyInputsBuilder[_I],
    ) -> BaseMultiModalProcessor[_I]: ...


@dataclass(frozen=True)
class _ProcessorFactories(Generic[_I]):
    info: ProcessingInfoFactory[_I]
    processor: MultiModalProcessorFactory[_I]
    dummy_inputs: DummyInputsBuilderFactory[_I]

    def build_processor(
        self,
        ctx: InputProcessingContext,
    ):
        info = self.info(ctx)
        dummy_inputs_builder = self.dummy_inputs(info)
        return self.processor(info, dummy_inputs_builder)


class MultiModalRegistry:
    """A registry that dispatches data processing according to the model."""

    def register_processor(
        self,
        processor: MultiModalProcessorFactory[_I],
        *,
        info: ProcessingInfoFactory[_I],
        dummy_inputs: DummyInputsBuilderFactory[_I],
    ):
        """Register a multi-modal processor to a model class. The processor
        is constructed lazily, hence a factory method should be passed.

        When the model receives multi-modal data, the provided function is
        invoked to transform the data into a dictionary of model inputs.
        """

        def wrapper(model_cls: N) -> N:
            if "_processor_factory" in model_cls.__dict__:
                logger.warning(
                    "Model class %s already has a multi-modal processor "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls,
                    self,
                )

            model_cls._processor_factory = _ProcessorFactories(
                info=info,
                dummy_inputs=dummy_inputs,
                processor=processor,
            )

            return model_cls

        return wrapper

    def _get_model_cls(self, model_config: "ModelConfig") -> "SupportsMultiModal":
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        if not hasattr(model_cls, "_processor_factory"):
            raise ValueError(
                f"Model class {model_cls.__name__} has no registered "
                "multimodal processor"
            )
        return cast("SupportsMultiModal", model_cls)

    def _create_processing_ctx(
        self,
        model_config: "ModelConfig",
        tokenizer: TokenizerLike | None = None,
    ) -> InputProcessingContext:
        if tokenizer is None:
            tokenizer = cached_tokenizer_from_config(model_config)

        return InputProcessingContext(model_config, tokenizer)

    def _create_processing_info(
        self,
        model_config: "ModelConfig",
        tokenizer: TokenizerLike | None = None,
    ) -> BaseProcessingInfo:
        model_cls = self._get_model_cls(model_config)
        factories = model_cls._processor_factory
        ctx = self._create_processing_ctx(model_config, tokenizer)
        return factories.info(ctx)

    def get_processing_info(self, model_config: "ModelConfig") -> BaseProcessingInfo:
        return self._create_processing_info(model_config, tokenizer=None)

    def create_processor(
        self,
        model_config: "ModelConfig",
        *,
        tokenizer: TokenizerLike | None = None,
    ) -> BaseMultiModalProcessor[BaseProcessingInfo]:
        """Create a multi-modal processor for a specific model and tokenizer."""
        if not model_config.is_multimodal_model:
            model_name = model_config.served_model_name or model_config.model
            raise ValueError(f"{model_name} is not a multimodal model")

        model_cls = self._get_model_cls(model_config)
        factories = model_cls._processor_factory

        ctx = self._create_processing_ctx(model_config, tokenizer)

        return factories.build_processor(ctx)


class MultiModalTimingRegistry:
    def __init__(self, observability_config: "ObservabilityConfig | None") -> None:
        super().__init__()

        if observability_config and observability_config.enable_mm_processor_stats:
            self._lock = threading.Lock()
            self._ctx_by_request_id = defaultdict[str, TimingContext](TimingContext)
            self._enabled = True
        else:
            self._enabled = False

    def get(self, request_id: str) -> TimingContext:
        if not self._enabled:
            return TimingContext(enabled=False)

        with self._lock:
            return self._ctx_by_request_id[request_id]

    def stat(self) -> dict[str, dict[str, float]]:
        if not self._enabled:
            return {}

        with self._lock:
            stats = {
                req_id: ctx.get_stats_dict()
                for req_id, ctx in self._ctx_by_request_id.items()
            }
            self._ctx_by_request_id.clear()
            return stats
