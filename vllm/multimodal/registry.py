# SPDX-License-Identifier: Apache-2.0
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Optional, Protocol, TypeVar

import torch.nn as nn
from typing_extensions import deprecated

from vllm.envs import VLLM_MM_INPUT_CACHE_GIB
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
    """Constructs a :class:`MultiModalProcessor` instance from the context."""

    def __call__(
        self,
        ctx: InputProcessingContext,
    ) -> _I_co:
        ...


class DummyInputsBuilderFactory(Protocol[_I]):
    """
    Constructs a :class:`BaseDummyInputsBuilder` instance from the context.
    """

    def __call__(self, info: _I) -> BaseDummyInputsBuilder[_I]:
        ...


class MultiModalProcessorFactory(Protocol[_I]):
    """Constructs a :class:`MultiModalProcessor` instance from the context."""

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


class MultiModalRegistry:
    """
    A registry that dispatches data processing according to the model.
    """

    def __init__(self) -> None:
        self._processor_factories = ClassRegistry[nn.Module,
                                                  _ProcessorFactories]()

        self._processing_cache = ProcessingCache(VLLM_MM_INPUT_CACHE_GIB)

    @deprecated("Legacy input processor/mapper pipeline has been removed. "
                "Please update your model runner to use "
                "`seq_group_metadata.multi_modal_data` directly without "
                "further processing.")
    def create_input_mapper(self, model_config: "ModelConfig"):
        return lambda data, mm_processor_kwargs: data

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

        processor = self.create_processor(model_config, disable_cache=True)
        profiler = MultiModalProfiler(processor)

        seq_len = model_config.max_model_len
        mm_limits = self.get_mm_limits_per_prompt(model_config)

        return profiler.get_mm_max_tokens(
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

        See :meth:`MultiModalPlugin.get_max_multimodal_tokens` for more details.
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

        See :meth:`MultiModalPlugin.get_max_multimodal_tokens` for more details.
        """
        return sum(self.get_max_tokens_by_modality(model_config).values())

    @deprecated("Legacy input processor/mapper pipeline has been removed. "
                "Please update your model runner to use "
                "`seq_group_metadata.multi_modal_data` directly without "
                "further processing.")
    def init_mm_limits_per_prompt(
        self,
        model_config: "ModelConfig",
    ) -> None:
        pass

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

        processor = self.create_processor(model_config, disable_cache=True)
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

        See also:
            :ref:`mm-processing`
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

    @deprecated("Legacy input processor/mapper pipeline has been removed. "
                "Please update your model runner to use "
                "`seq_group_metadata.multi_modal_data` directly without "
                "further processing.")
    def has_processor(self, model_config: "ModelConfig") -> bool:
        return True

    def create_processor(
        self,
        model_config: "ModelConfig",
        *,
        tokenizer: Optional[AnyTokenizer] = None,
        disable_cache: Optional[bool] = None,
    ) -> BaseMultiModalProcessor[BaseProcessingInfo]:
        """
        Create a multi-modal processor for a specific model and tokenizer.

        See also:
            :ref:`mm-processing`
        """
        if not model_config.is_multimodal_model:
            raise ValueError(f"{model_config.model} is not a multimodal model")

        if tokenizer is None:
            tokenizer = cached_tokenizer_from_config(model_config)
        if disable_cache is None:
            disable_cache = model_config.disable_mm_preprocessor_cache

        model_cls = self._get_model_cls(model_config)
        factories = self._processor_factories[model_cls]

        ctx = InputProcessingContext(model_config, tokenizer)
        cache = None if disable_cache else self._processing_cache

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
        processor = self.create_processor(model_config, disable_cache=True)
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
        processor = self.create_processor(model_config, disable_cache=True)
        profiler = MultiModalProfiler(processor)
        dummy_data = profiler.get_encoder_dummy_data(seq_len, mm_counts)

        # Having more tokens is over-conservative but otherwise fine
        token_ids = dummy_data.prompt_token_ids
        if len(token_ids) < seq_len:
            logger.warning_once(
                f"Expected at least {seq_len} dummy encoder tokens for "
                f"profiling, but found {len(token_ids)} tokens instead.")

        return dummy_data
