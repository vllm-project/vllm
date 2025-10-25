# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, cached_tokenizer_from_config

from .cache import BaseMultiModalProcessorCache
from .processing import (
    BaseMultiModalProcessor,
    EncDecMultiModalProcessor,
    InputProcessingContext,
)
from .profiling import DummyDecoderData, DummyEncoderData

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.model_executor.models.interfaces import SupportsMultiModal

logger = init_logger(__name__)


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

        model_cls = self._get_model_cls(model_config)
        ctx = self._create_processing_ctx(model_config)
        processor_info = model_cls.processor_info(ctx)
        profiling_info = model_cls.profiling_info(processor_info)

        mm_config = model_config.get_multimodal_config()

        # Check if all supported modalities have limit == 0
        if all(
            mm_config.get_limit_per_prompt(modality) == 0
            for modality in profiling_info.supported_mm_limits
        ):
            logger.info_once(
                "All limits of multimodal modalities supported by the model "
                "are set to 0, running in text-only mode."
            )
            return False

        return True

    def get_max_tokens_per_item_by_modality(
        self,
        model_config: "ModelConfig",
        *,
        cache: BaseMultiModalProcessorCache | None = None,
        profiler_limits: Mapping[str, int] | None = None,
    ) -> Mapping[str, int]:
        """
        Get the maximum number of tokens per data item from each modality based
        on underlying model configuration.
        """
        if not model_config.is_multimodal_model:
            return {}

        processor = self.create_processor(model_config, cache=cache)

        seq_len = model_config.max_model_len
        profiler_limits = (
            processor.dummy_builder.profiling_info.allowed_mm_limits
            if profiler_limits is None
            else profiler_limits
        )

        return processor.dummy_builder.get_mm_max_contiguous_tokens(
            processor,
            seq_len,
            {modality: 1 for modality, limit in profiler_limits.items() if limit > 0},
        )

    def get_mm_limits_per_prompt(
        self,
        model_config: "ModelConfig",
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> Mapping[str, int]:
        """
        Get the maximum number of multi-modal input instances for each modality
        that are allowed per prompt for a model class.
        """
        if not model_config.is_multimodal_model:
            return {}

        model_cls = self._get_model_cls(model_config)
        ctx = self._create_processing_ctx(model_config)
        processor_info = model_cls.processor_info(ctx)
        profiling_info = model_cls.profiling_info(processor_info)

        return profiling_info.allowed_mm_limits

    def register_processor(self, *args, **kwargs):
        # TODO: Link to PR
        raise ValueError(
            "We have going to remove multi-modal registry in the near future. "
            "Please see PR# on how to update your model."
        )

    def _get_model_cls(self, model_config: "ModelConfig"):
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        return cast("SupportsMultiModal", model_cls)

    def _create_processing_ctx(
        self,
        model_config: "ModelConfig",
        tokenizer: AnyTokenizer | None = None,
    ) -> InputProcessingContext:
        if tokenizer is None and not model_config.skip_tokenizer_init:
            tokenizer = cached_tokenizer_from_config(model_config)

        return InputProcessingContext(model_config, tokenizer)

    def create_processor(
        self,
        model_config: "ModelConfig",
        *,
        tokenizer: AnyTokenizer | None = None,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> BaseMultiModalProcessor:
        """
        Create a multi-modal processor for a specific model and tokenizer.
        """
        if not model_config.is_multimodal_model:
            raise ValueError(f"{model_config.model} is not a multimodal model")

        model_cls = self._get_model_cls(model_config)

        ctx = self._create_processing_ctx(model_config, tokenizer)
        processor_info = model_cls.processor_info(ctx)
        profiling_info = model_cls.profiling_info(processor_info)
        dummy_builder = model_cls.dummy_builder(profiling_info)

        return model_cls.processor(dummy_builder, cache=cache)

    def get_decoder_dummy_data(
        self,
        model_config: "ModelConfig",
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> DummyDecoderData:
        """
        Create dummy data for profiling the memory usage of a model.

        The model is identified by `model_config`.
        """
        processor = self.create_processor(model_config, cache=cache)

        # Extract configurable options from multimodal config.
        # Only include modalities that use advanced option types so legacy
        # count-only behavior remains unchanged.
        mm_options = self._extract_mm_options(model_config)

        dummy_data = processor.dummy_builder.get_decoder_dummy_data(
            processor,
            seq_len,
            mm_counts=mm_counts,
            mm_options=mm_options,
        )

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
        model_config: "ModelConfig",
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> DummyEncoderData:
        """
        Create dummy data for profiling the memory usage of a model.

        The model is identified by `model_config`.
        """
        processor = cast(
            EncDecMultiModalProcessor,
            self.create_processor(model_config, cache=cache),
        )

        # Extract configurable options from multimodal config.
        # Only include modalities that use advanced option types so legacy
        # count-only behavior remains unchanged.
        mm_options = self._extract_mm_options(model_config)

        dummy_data = processor.dummy_builder.get_encoder_dummy_data(
            processor,
            seq_len,
            mm_counts=mm_counts,
            mm_options=mm_options,
        )

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
        max_tokens = self.get_max_tokens_per_item_by_modality(model_config)
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
