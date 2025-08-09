# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Generic, Optional, Protocol, TypeVar

import torch.nn as nn

from vllm.inputs import InputProcessingContext
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import (AnyTokenizer,
                                               cached_tokenizer_from_config)
from vllm.utils import ClassRegistry

from .processing import (BaseMultiModalProcessor, BaseProcessingInfo,
                         ProcessingCache)
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
        cache: Optional[ProcessingCache] = None,
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
        cache: Optional[ProcessingCache] = None,
    ):
        info = self.info(ctx)
        dummy_inputs_builder = self.dummy_inputs(info)
        return self.processor(info, dummy_inputs_builder, cache=cache)


# Make sure a different cache is used for each model config
# NOTE: ModelConfig is not hashable so it cannot be passed directly
@lru_cache(maxsize=1)
def _get_processor_cache(model_id: str, capacity_gb: int):
    return ProcessingCache(capacity_gb) if capacity_gb > 0 else None


class MultiModalRegistry:
    """
    A registry that dispatches data processing according to the model.
    """

    def __init__(self) -> None:
        self._processor_factories = ClassRegistry[nn.Module,
                                                  _ProcessorFactories]()

    def _get_processor_cache(self, model_config: "ModelConfig"):
        model_id = model_config.model
        capacity_gb = model_config.mm_processor_cache_gb
        return _get_processor_cache(model_id, capacity_gb)

    def reset_processor_cache(self, model_config: "ModelConfig") -> bool:
        """Reset the multi-modal processing cache."""
        if processor_cache := self._get_processor_cache(model_config):
            processor_cache.reset()

        return True  # Success

    def enable_mm_input_cache(self, model_config: "ModelConfig") -> bool:
        """Whether the multi-modal input cache should be enabled.
        NOTE: This is put under MultiModalRegistry on purpose to respect 
        text-only mode for multimodal models.
        """

        if not self.supports_multimodal_inputs(model_config):
            return False

        mm_config = model_config.get_multimodal_config()

        return mm_config.mm_processor_cache_gb > 0

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
    ) -> Mapping[str, int]:
        """
        Get the maximum number of tokens per data item from each modality based
        on underlying model configuration.
        """
        if not model_config.is_multimodal_model:
            return {}

        processor = self.create_processor(model_config, disable_cache=False)
        profiler = MultiModalProfiler(processor)

        seq_len = model_config.max_model_len
        mm_limits = self.get_mm_limits_per_prompt(model_config)

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
    ) -> Mapping[str, int]:
        """
        Get the maximum number of tokens per data item from each modality based
        on underlying model configuration, excluding modalities that user
        explicitly disabled via `limit_mm_per_prompt`.

        Note:
            This is currently directly used only in V1 for profiling the memory
            usage of a model.
        """
        mm_limits = self.get_mm_limits_per_prompt(model_config)

        return {
            key: max_tokens_per_mm_item
            for key, max_tokens_per_mm_item in
            self.get_max_tokens_per_item_by_modality(model_config).items()
            if mm_limits[key] > 0
        }

    def get_max_tokens_by_modality(
        self,
        model_config: "ModelConfig",
    ) -> Mapping[str, int]:
        """
        Get the maximum number of tokens from each modality
        for profiling the memory usage of a model.
        """
        mm_limits = self.get_mm_limits_per_prompt(model_config)

        return {
            key: mm_limits[key] * max_tokens_per_mm_item
            for key, max_tokens_per_mm_item in
            self.get_max_tokens_per_item_by_modality(model_config).items()
        }

    def get_max_multimodal_tokens(self, model_config: "ModelConfig") -> int:
        """
        Get the maximum number of multi-modal tokens
        for profiling the memory usage of a model.
        """
        return sum(self.get_max_tokens_by_modality(model_config).values())

    def get_mm_limits_per_prompt(
        self,
        model_config: "ModelConfig",
    ) -> Mapping[str, int]:
        """
        Get the maximum number of multi-modal input instances for each modality
        that are allowed per prompt for a model class.
        """
        if not model_config.is_multimodal_model:
            return {}

        processor = self.create_processor(model_config, disable_cache=False)
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
        disable_cache: Optional[bool] = None,
    ) -> BaseMultiModalProcessor[BaseProcessingInfo]:
        """
        Create a multi-modal processor for a specific model and tokenizer.
        """
        if not model_config.is_multimodal_model:
            raise ValueError(f"{model_config.model} is not a multimodal model")

        if disable_cache is None:
            disable_cache = not model_config.enable_mm_processor_cache

        model_cls = self._get_model_cls(model_config)
        factories = self._processor_factories[model_cls]

        ctx = self._create_processing_ctx(model_config, tokenizer)
        cache = None if disable_cache else self._get_processor_cache(
            model_config)

        return factories.build_processor(ctx, cache=cache)

    def get_decoder_dummy_data(
        self,
        model_config: "ModelConfig",
        seq_len: int,
        mm_counts: Optional[Mapping[str, int]] = None,
    ) -> DummyDecoderData:
        """
        Create dummy data for profiling the memory usage of a model.

        The model is identified by ``model_config``.
        """
        processor = self.create_processor(model_config, disable_cache=False)
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
    ) -> DummyEncoderData:
        """
        Create dummy data for profiling the memory usage of a model.

        The model is identified by ``model_config``.
        """
        processor = self.create_processor(model_config, disable_cache=False)
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
