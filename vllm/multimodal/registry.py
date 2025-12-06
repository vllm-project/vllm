# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, cast

from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike

from .cache import BaseMultiModalProcessorCache
from .processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
)
from .profiling import (
    BaseDummyInputsBuilder,
    DummyDecoderData,
    DummyEncoderData,
    MultiModalProfiler,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, RendererConfig
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
    [`BaseDummyInputsBuilder`][vllm.multimodal.profiling.BaseDummyInputsBuilder]
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

    def supports_multimodal_inputs(self, renderer_config: "RendererConfig") -> bool:
        """
        Checks if the model supports multimodal inputs.
        Returns True if the model is multimodal with any non-zero supported
        modalities, otherwise returns False, effectively running in
        text-only mode.
        """
        model_config = renderer_config.model_config
        if not model_config.is_multimodal_model:
            return False

        info = self._create_processing_info(renderer_config, tokenizer=None)
        supported_modalities = info.get_supported_mm_limits()

        mm_config = model_config.get_multimodal_config()

        # Check if all supported modalities have limit == 0
        if all(
            mm_config.get_limit_per_prompt(modality) == 0
            for modality in supported_modalities
        ):
            logger.info_once(
                "All limits of multimodal modalities supported by the model "
                "are set to 0, running in text-only mode."
            )
            return False

        return True

    def get_max_tokens_per_item_by_modality(
        self,
        renderer_config: "RendererConfig",
        *,
        cache: BaseMultiModalProcessorCache | None = None,
        profiler_limits: Mapping[str, int] | None = None,
    ) -> Mapping[str, int]:
        """
        Get the maximum number of tokens per data item from each modality based
        on underlying model configuration.
        """
        model_config = renderer_config.model_config
        if not model_config.is_multimodal_model:
            return {}

        processor = self.create_processor(renderer_config, cache=cache)
        profiler: MultiModalProfiler = MultiModalProfiler(processor)

        seq_len = model_config.max_model_len
        profiler_limits = (
            profiler.get_mm_limits() if profiler_limits is None else profiler_limits
        )

        return profiler.get_mm_max_contiguous_tokens(
            seq_len,
            {modality: 1 for modality, limit in profiler_limits.items() if limit > 0},
        )

    def get_mm_limits_per_prompt(
        self,
        renderer_config: "RendererConfig",
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> Mapping[str, int]:
        """
        Get the maximum number of multi-modal input instances for each modality
        that are allowed per prompt for a model class.
        """
        model_config = renderer_config.model_config
        if not model_config.is_multimodal_model:
            return {}

        processor = self.create_processor(renderer_config, cache=cache)
        profiler: MultiModalProfiler = MultiModalProfiler(processor)
        return profiler.get_mm_limits()

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

    def _create_processing_info(
        self,
        renderer_config: "RendererConfig",
        *,
        tokenizer: TokenizerLike | None = None,
    ) -> BaseProcessingInfo:
        model_cls = self._get_model_cls(renderer_config.model_config)
        factories = model_cls._processor_factory

        ctx = InputProcessingContext.from_config(renderer_config, tokenizer=tokenizer)
        return factories.info(ctx)

    def create_processor(
        self,
        renderer_config: "RendererConfig",
        *,
        tokenizer: TokenizerLike | None = None,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> BaseMultiModalProcessor[BaseProcessingInfo]:
        """
        Create a multi-modal processor for a specific model and tokenizer.
        """
        model_config = renderer_config.model_config
        if not model_config.is_multimodal_model:
            raise ValueError(f"{model_config.model} is not a multimodal model")

        model_cls = self._get_model_cls(model_config)
        factories = model_cls._processor_factory

        ctx = InputProcessingContext.from_config(renderer_config, tokenizer=tokenizer)
        return factories.build_processor(ctx, cache=cache)

    def get_decoder_dummy_data(
        self,
        renderer_config: "RendererConfig",
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> DummyDecoderData:
        """
        Create dummy data for profiling the memory usage of a model.

        The model is identified by `renderer_config`.
        """
        processor = self.create_processor(renderer_config, cache=cache)
        profiler: MultiModalProfiler = MultiModalProfiler(processor)

        # Extract configurable options from multimodal config.
        # Only include modalities that use advanced option types so legacy
        # count-only behavior remains unchanged.
        mm_options = self._extract_mm_options(renderer_config.model_config)

        dummy_data = profiler.get_decoder_dummy_data(seq_len, mm_counts, mm_options)

        # Having more tokens is over-conservative but otherwise fine
        token_ids = dummy_data.prompt_token_ids
        if len(token_ids) < seq_len:
            raise AssertionError(
                f"Expected at least {seq_len} dummy tokens for profiling, "
                f"but found {len(token_ids)} tokens instead."
            )

        return dummy_data

    def get_encoder_dummy_data(
        self,
        renderer_config: "RendererConfig",
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> DummyEncoderData:
        """
        Create dummy data for profiling the memory usage of a model.

        The model is identified by `renderer_config`.
        """
        processor = self.create_processor(renderer_config, cache=cache)
        profiler: MultiModalProfiler = MultiModalProfiler(processor)

        # Extract configurable options from multimodal config.
        # Only include modalities that use advanced option types so legacy
        # count-only behavior remains unchanged.
        mm_options = self._extract_mm_options(renderer_config.model_config)

        dummy_data = profiler.get_encoder_dummy_data(seq_len, mm_counts, mm_options)

        # Having more tokens is over-conservative but otherwise fine
        token_ids = dummy_data.prompt_token_ids
        if len(token_ids) < seq_len:
            logger.warning_once(
                "Expected at least %d dummy encoder tokens for profiling, but found %d tokens instead.",  # noqa: E501
                seq_len,
                len(token_ids),
            )

        return dummy_data

    def get_encdec_max_encoder_len(self, renderer_config: "RendererConfig") -> int:
        """
        Get the maximum length of the encoder input for encoder-decoder models.
        """
        model_config = renderer_config.model_config
        if not model_config.is_encoder_decoder:
            return 0

        max_tokens = self.get_max_tokens_per_item_by_modality(renderer_config)
        if not max_tokens:
            # TODO - this function assumes encoder-decoder models are
            # multimodal. This will need to change when adding support for more
            # than whisper.
            return 0
        assert len(max_tokens) == 1, (
            "Encoder-decoder models are expected \
            to implement the multimodal interface with at most one modality."
        )

        first_modality = next(iter(max_tokens))
        return max_tokens[first_modality]
