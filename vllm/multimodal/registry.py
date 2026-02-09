# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing.synchronize import Lock as LockType
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar, cast

from vllm.config.multimodal import BaseDummyOptions
from vllm.config.observability import ObservabilityConfig
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config

from .cache import (
    BaseMultiModalProcessorCache,
    BaseMultiModalReceiverCache,
    MultiModalProcessorOnlyCache,
    MultiModalProcessorSenderCache,
    MultiModalReceiverCache,
    ShmObjectStoreReceiverCache,
    ShmObjectStoreSenderCache,
)
from .inputs import MultiModalInputs
from .processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, ObservabilityConfig, VllmConfig
    from vllm.model_executor.models.interfaces import SupportsMultiModal

logger = init_logger(__name__)

N = TypeVar("N", bound=type["SupportsMultiModal"])
_I = TypeVar("_I", bound=BaseProcessingInfo)
_I_co = TypeVar("_I_co", bound=BaseProcessingInfo, covariant=True)


class ProcessingInfoFactory(Protocol[_I_co]):
    """
    Constructs a
    [`BaseMultiModalProcessor`][vllm.multimodal.processing.BaseMultiModalProcessor]
    instance from the context.
    """

    def __call__(
        self,
        ctx: InputProcessingContext,
    ) -> _I_co: ...


class DummyInputsBuilderFactory(Protocol[_I]):  # type: ignore[misc]
    """
    Constructs a
    [`BaseDummyInputsBuilder`][vllm.multimodal.processing.BaseDummyInputsBuilder]
    instance from the context.
    """

    def __call__(self, info: _I) -> BaseDummyInputsBuilder[_I]: ...


class MultiModalProcessorFactory(Protocol[_I]):  # type: ignore[misc]
    """
    Constructs a
    [`BaseMultiModalProcessor`][vllm.multimodal.processing.BaseMultiModalProcessor]
    instance from the context.
    """

    def __call__(
        self,
        info: _I,
        dummy_inputs: BaseDummyInputsBuilder[_I],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> BaseMultiModalProcessor[_I]: ...


@dataclass(frozen=True)
class _ProcessorFactories(Generic[_I]):
    info: ProcessingInfoFactory[_I]
    processor: MultiModalProcessorFactory[_I]
    dummy_inputs: DummyInputsBuilderFactory[_I]

    def build_processor(
        self,
        ctx: InputProcessingContext,
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ):
        info = self.info(ctx)
        dummy_inputs_builder = self.dummy_inputs(info)
        return self.processor(info, dummy_inputs_builder, cache=cache)


class MultiModalRegistry:
    """
    A registry that dispatches data processing according to the model.
    """

    def _extract_mm_options(
        self,
        model_config: "ModelConfig",
    ) -> Mapping[str, BaseDummyOptions] | None:
        """
        Extract multimodal dummy options from model config.

        Returns None if no configurable options are found, otherwise returns
        a mapping of modality names to their dummy options.
        """
        if not model_config.multimodal_config:
            return None

        mm_options = {
            m: opt
            for m in model_config.multimodal_config.limit_per_prompt
            if (opt := model_config.multimodal_config.get_dummy_options(m)) is not None
        }

        return mm_options if len(mm_options) > 0 else None

    def supports_multimodal_inputs(self, model_config: "ModelConfig") -> bool:
        """
        Checks if the model supports multimodal inputs.
        Returns True if the model is multimodal with any non-zero supported
        modalities, otherwise returns False, effectively running in
        text-only mode.
        """
        if not model_config.is_multimodal_model:
            return False

        mm_config = model_config.get_multimodal_config()
        info = self._create_processing_info(model_config, tokenizer=None)

        # Check if all supported modalities have limit == 0
        if all(
            mm_config.get_limit_per_prompt(modality) == 0
            for modality in info.supported_mm_limits
        ):
            # If enable_mm_embeds is True, we still need MM infrastructure
            # to process pre-computed embeddings even though encoder won't run
            if mm_config.enable_mm_embeds:
                return True

            logger.info_once(
                "All limits of multimodal modalities supported by the model "
                "are set to 0, running in text-only mode."
            )
            return False

        return True

    def register_processor(
        self,
        processor: MultiModalProcessorFactory[_I],
        *,
        info: ProcessingInfoFactory[_I],
        dummy_inputs: DummyInputsBuilderFactory[_I],
    ):
        """
        Register a multi-modal processor to a model class. The processor
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
        assert hasattr(model_cls, "_processor_factory")
        return cast("SupportsMultiModal", model_cls)

    def _create_processing_ctx(
        self,
        model_config: "ModelConfig",
        observability_config: "ObservabilityConfig | None" = None,
        tokenizer: TokenizerLike | None = None,
    ) -> InputProcessingContext:
        if tokenizer is None and not model_config.skip_tokenizer_init:
            tokenizer = cached_tokenizer_from_config(model_config)

        return InputProcessingContext(
            model_config, tokenizer, observability_config=observability_config
        )

    def _create_processing_info(
        self,
        model_config: "ModelConfig",
        observability_config: "ObservabilityConfig | None" = None,
        *,
        tokenizer: TokenizerLike | None = None,
    ) -> BaseProcessingInfo:
        model_cls = self._get_model_cls(model_config)
        factories = model_cls._processor_factory
        ctx = self._create_processing_ctx(model_config, observability_config, tokenizer)
        return factories.info(ctx)

    def create_processor(
        self,
        model_config: "ModelConfig",
        observability_config: "ObservabilityConfig | None" = None,
        *,
        tokenizer: TokenizerLike | None = None,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> BaseMultiModalProcessor[BaseProcessingInfo]:
        """
        Create a multi-modal processor for a specific model and tokenizer.
        """
        if not model_config.is_multimodal_model:
            raise ValueError(f"{model_config.model} is not a multimodal model")

        model_cls = self._get_model_cls(model_config)
        factories = model_cls._processor_factory

        ctx = self._create_processing_ctx(model_config, observability_config, tokenizer)

        return factories.build_processor(ctx, cache=cache)

    def get_dummy_mm_inputs(
        self,
        model_config: "ModelConfig",
        mm_counts: Mapping[str, int],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
        processor: BaseMultiModalProcessor | None = None,
    ) -> MultiModalInputs:
        """
        Create dummy data for profiling the memory usage of a model.

        The model is identified by `model_config`.
        """
        seq_len = model_config.max_model_len

        if processor is None:
            processor = self.create_processor(model_config, cache=cache)

        processor_inputs = processor.dummy_inputs.get_dummy_processor_inputs(
            seq_len=seq_len,
            mm_counts=mm_counts,
            mm_options=self._extract_mm_options(model_config),
        )
        mm_inputs = processor.apply(
            prompt=processor_inputs.prompt,
            mm_items=processor_inputs.mm_items,
            hf_processor_mm_kwargs=processor_inputs.hf_processor_mm_kwargs,
            tokenization_kwargs=processor_inputs.tokenization_kwargs,
        )

        prompt_token_ids = mm_inputs["prompt_token_ids"]
        total_len = len(prompt_token_ids)
        if total_len < seq_len:
            prompt_token_ids.extend([0] * (seq_len - total_len))

        return mm_inputs

    def _get_cache_type(
        self,
        vllm_config: "VllmConfig",
    ) -> Literal[None, "processor_only", "lru", "shm"]:
        model_config = vllm_config.model_config
        if not self.supports_multimodal_inputs(model_config):
            return None

        # Check if the cache is disabled.
        mm_config = model_config.get_multimodal_config()
        if mm_config.mm_processor_cache_gb <= 0:
            return None

        # Check if IPC caching is supported.
        parallel_config = vllm_config.parallel_config
        is_ipc_supported = parallel_config._api_process_count == 1 and (
            parallel_config.data_parallel_size == 1
            or parallel_config.data_parallel_external_lb
        )

        if not is_ipc_supported:
            return "processor_only"

        mm_config = model_config.get_multimodal_config()
        return mm_config.mm_processor_cache_type

    def processor_cache_from_config(
        self,
        vllm_config: "VllmConfig",
    ) -> BaseMultiModalProcessorCache | None:
        """Return a `BaseMultiModalProcessorCache`, if enabled."""
        cache_type = self._get_cache_type(vllm_config)
        if cache_type is None:
            return None
        elif cache_type == "processor_only":
            return MultiModalProcessorOnlyCache(vllm_config.model_config)
        elif cache_type == "lru":
            return MultiModalProcessorSenderCache(vllm_config.model_config)
        elif cache_type == "shm":
            return ShmObjectStoreSenderCache(vllm_config)
        else:
            raise ValueError(f"Unknown cache type: {cache_type!r}")

    def processor_only_cache_from_config(
        self,
        vllm_config: "VllmConfig",
    ) -> MultiModalProcessorOnlyCache | None:
        """Return a `MultiModalProcessorOnlyCache`, if enabled."""
        cache_type = self._get_cache_type(vllm_config)
        if cache_type is None:
            return None

        return MultiModalProcessorOnlyCache(vllm_config.model_config)

    def engine_receiver_cache_from_config(
        self,
        vllm_config: "VllmConfig",
    ) -> BaseMultiModalReceiverCache | None:
        """Return a `BaseMultiModalReceiverCache` for the engine process."""
        cache_type = self._get_cache_type(vllm_config)
        if cache_type in (None, "processor_only", "shm"):
            return None
        elif cache_type == "lru":
            return MultiModalReceiverCache(vllm_config.model_config)
        else:
            raise ValueError(f"Unknown cache type: {cache_type!r}")

    def worker_receiver_cache_from_config(
        self,
        vllm_config: "VllmConfig",
        shared_worker_lock: LockType,
    ) -> BaseMultiModalReceiverCache | None:
        """Return a `BaseMultiModalReceiverCache` for the worker process."""
        cache_type = self._get_cache_type(vllm_config)
        if cache_type in (None, "processor_only", "lru"):
            return None
        elif cache_type == "shm":
            return ShmObjectStoreReceiverCache(vllm_config, shared_worker_lock)
        else:
            raise ValueError(f"Unknown cache type: {cache_type!r}")
