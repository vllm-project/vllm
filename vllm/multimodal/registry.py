# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Optional, Protocol, TypeVar

import torch.nn as nn

from vllm.inputs import InputProcessingContext
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import (AnyTokenizer,
                                               cached_tokenizer_from_config)
from vllm.utils import ClassRegistry

from .cache import BaseMultiModalProcessorCache
from .processing import BaseMultiModalProcessor, BaseProcessingInfo
from .profiling import (BaseDummyInputsBuilder, DummyDecoderData,
                        DummyEncoderData, MultiModalProfiler)

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

N = TypeVar("N", bound=type[nn.Module])
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
    ) -> _I_co:
        ...


class DummyInputsBuilderFactory(Protocol[_I]):
    """
    Constructs a
    [`BaseDummyInputsBuilder`][vllm.multimodal.profiling.BaseDummyInputsBuilder]
    instance from the context.
    """

    def __call__(self, info: _I) -> BaseDummyInputsBuilder[_I]:
        ...


class MultiModalProcessorFactory(Protocol[_I]):
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
        cache: Optional[BaseMultiModalProcessorCache] = None,
    ) -> BaseMultiModalProcessor[_I]:
        ...


@dataclass(frozen=True)
class _ProcessorFactories(Generic[_I]):
    info: ProcessingInfoFactory[_I]
    processor: MultiModalProcessorFactory[_I]
    dummy_inputs: DummyInputsBuilderFactory[_I]

    def build_processor(
        self,
        ctx: InputProcessingContext,
        *,
        cache: Optional[BaseMultiModalProcessorCache] = None,
    ):
        info = self.info(ctx)
        dummy_inputs_builder = self.dummy_inputs(info)
        return self.processor(info, dummy_inputs_builder, cache=cache)


class MultiModalRegistry:
    """
    A registry that dispatches data processing according to the model.
    """

    def __init__(self) -> None:
        self._processor_factories = ClassRegistry[nn.Module,
                                                  _ProcessorFactories]()

    def supports_multimodal_inputs(self, model_config: "ModelConfig") -> bool:
        """
        Checks if the model supports multimodal inputs.
        Returns True if the model is multimodal with any non-zero supported 
        modalities, otherwise returns False, effectively running in 
        text-only mode.
        """
        if not model_config.is_multimodal_model:
            return False

        info = self._create_processing_info(model_config, tokenizer=None)
        supported_modalities = info.get_supported_mm_limits()

        mm_config = model_config.get_multimodal_config()

        # Check if all supported modalities have limit == 0
        if all(
                mm_config.get_limit_per_prompt(modality) == 0
                for modality in supported_modalities):
            logger.info_once(
                "All limits of multimodal modalities supported by the model "
                "are set to 0, running in text-only mode.")
            return False

        return True

    def get_max_tokens_per_item_by_modality(
        self,
        model_config: "ModelConfig",
        *,
        cache: Optional[BaseMultiModalProcessorCache] = None,
    ) -> Mapping[str, int]:
        """
        Get the maximum number of tokens per data item from each modality based
        on underlying model configuration.
        """
        if not model_config.is_multimodal_model:
            return {}

        processor = self.create_processor(model_config, cache=cache)
        profiler = MultiModalProfiler(processor)

        seq_len = model_config.max_model_len
        mm_limits = self.get_mm_limits_per_prompt(model_config, cache=cache)

        return profiler.get_mm_max_contiguous_tokens(
            seq_len,
            {
                modality: 1
                for modality, limit in mm_limits.items() if limit > 0
            },
        )

    def get_max_tokens_per_item_by_nonzero_modality(
        self,
        model_config: "ModelConfig",
        *,
        cache: Optional[BaseMultiModalProcessorCache] = None,
    ) -> Mapping[str, int]:
        """
        Get the maximum number of tokens per data item from each modality based
        on underlying model configuration, excluding modalities that user
        explicitly disabled via `limit_mm_per_prompt`.

        Note:
            This is currently directly used only in V1 for profiling the memory
            usage of a model.
        """
        mm_limits = self.get_mm_limits_per_prompt(model_config, cache=cache)
        max_tokens_per_item = self.get_max_tokens_per_item_by_modality(
            model_config,
            cache=cache,
        )

        return {
            key: max_tokens_per_mm_item
            for key, max_tokens_per_mm_item in max_tokens_per_item.items()
            if mm_limits[key] > 0
        }

    def get_mm_limits_per_prompt(
        self,
        model_config: "ModelConfig",
        *,
        cache: Optional[BaseMultiModalProcessorCache] = None,
    ) -> Mapping[str, int]:
        """
        Get the maximum number of multi-modal input instances for each modality
        that are allowed per prompt for a model class.
        """
        if not model_config.is_multimodal_model:
            return {}

        processor = self.create_processor(model_config, cache=cache)
        profiler = MultiModalProfiler(processor)
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
            if self._processor_factories.contains(model_cls, strict=True):
                logger.warning(
                    "Model class %s already has a multi-modal processor "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._processor_factories[model_cls] = _ProcessorFactories(
                info=info,
                dummy_inputs=dummy_inputs,
                processor=processor,
            )

            return model_cls

        return wrapper

    def _get_model_cls(self, model_config: "ModelConfig"):
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        return model_cls

    def _create_processing_ctx(
        self,
        model_config: "ModelConfig",
        tokenizer: Optional[AnyTokenizer] = None,
    ) -> InputProcessingContext:
        if tokenizer is None and not model_config.skip_tokenizer_init:
            tokenizer = cached_tokenizer_from_config(model_config)
        return InputProcessingContext(model_config, tokenizer)

    def _create_processing_info(
        self,
        model_config: "ModelConfig",
        *,
        tokenizer: Optional[AnyTokenizer] = None,
    ) -> BaseProcessingInfo:
        model_cls = self._get_model_cls(model_config)
        factories = self._processor_factories[model_cls]
        ctx = self._create_processing_ctx(model_config, tokenizer)
        return factories.info(ctx)

    def create_processor(
        self,
        model_config: "ModelConfig",
        *,
        tokenizer: Optional[AnyTokenizer] = None,
        cache: Optional[BaseMultiModalProcessorCache] = None,
    ) -> BaseMultiModalProcessor[BaseProcessingInfo]:
        """
        Create a multi-modal processor for a specific model and tokenizer.
        """
        if not model_config.is_multimodal_model:
            raise ValueError(f"{model_config.model} is not a multimodal model")

        model_cls = self._get_model_cls(model_config)
        factories = self._processor_factories[model_cls]

        ctx = self._create_processing_ctx(model_config, tokenizer)

        return factories.build_processor(ctx, cache=cache)

    def get_decoder_dummy_data(
        self,
        model_config: "ModelConfig",
        seq_len: int,
        mm_counts: Optional[Mapping[str, int]] = None,
        *,
        cache: Optional[BaseMultiModalProcessorCache] = None,
    ) -> DummyDecoderData:
        """
        Create dummy data for profiling the memory usage of a model.

        The model is identified by ``model_config``.
        """
        processor = self.create_processor(model_config, cache=cache)
        profiler = MultiModalProfiler(processor)
        dummy_data = profiler.get_decoder_dummy_data(seq_len, mm_counts)

        # Having more tokens is over-conservative but otherwise fine
        token_ids = dummy_data.prompt_token_ids
        if len(token_ids) < seq_len:
            raise AssertionError(
                f"Expected at least {seq_len} dummy tokens for profiling, "
                f"but found {len(token_ids)} tokens instead.")

        return dummy_data

    def get_encoder_dummy_data(
        self,
        model_config: "ModelConfig",
        seq_len: int,
        mm_counts: Optional[Mapping[str, int]] = None,
        *,
        cache: Optional[BaseMultiModalProcessorCache] = None,
    ) -> DummyEncoderData:
        """
        Create dummy data for profiling the memory usage of a model.

        The model is identified by ``model_config``.
        """
        processor = self.create_processor(model_config, cache=cache)
        profiler = MultiModalProfiler(processor)
        dummy_data = profiler.get_encoder_dummy_data(seq_len, mm_counts)

        # Having more tokens is over-conservative but otherwise fine
        token_ids = dummy_data.prompt_token_ids
        if len(token_ids) < seq_len:
            logger.warning_once(
                "Expected at least %d dummy encoder tokens for profiling, but found %d tokens instead.",  # noqa: E501
                seq_len,
                len(token_ids),
            )

        return dummy_data

    def get_encdec_max_encoder_len(self, model_config: "ModelConfig") -> int:
        """
        Get the maximum length of the encoder input for encoder-decoder models.
        """
        if not model_config.is_encoder_decoder:
            return 0
        max_tokens = self.\
            get_max_tokens_per_item_by_nonzero_modality(model_config)
        if not max_tokens:
            # TODO - this function assumes encoder-decoder models are
            # multimodal. This will need to change when adding support for more
            # than whisper.
            return 0
        assert len(max_tokens) == 1, "Encoder-decoder models are expected \
            to implement the multimodal interface with at most one modality."

        first_modality = next(iter(max_tokens))
        return max_tokens[first_modality]
