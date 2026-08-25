# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for testing both processors of the Transformers modelling backend."""

import pytest

from vllm.config import ModelConfig
from vllm.model_executor.models.transformers.multimodal import (
    LegacyMultiModalProcessor,
    MultiModalDummyInputsBuilder,
    MultiModalProcessingInfo,
    MultiModalProcessor,
    OffsetsMultiModalProcessor,
)
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.processing import InputProcessingContext
from vllm.tokenizers.registry import cached_tokenizer_from_config

offsets_only = pytest.mark.skipif(
    MultiModalProcessor is not OffsetsMultiModalProcessor,
    reason="Replacement offsets are only used from transformers 5.15.0 onwards",
)

PROCESSOR_CLASSES = [
    pytest.param(LegacyMultiModalProcessor, id="legacy"),
    pytest.param(OffsetsMultiModalProcessor, id="offsets", marks=offsets_only),
]


def create_processor(model_id: str, processor_cls):
    """Build a processor directly, because the registry only ever builds the one the
    installed transformers version selects, leaving the other path untested."""
    model_config = ModelConfig(model=model_id, model_impl="transformers")
    ctx = InputProcessingContext(
        model_config, cached_tokenizer_from_config(model_config)
    )
    info = MultiModalProcessingInfo(ctx)
    return processor_cls(info, MultiModalDummyInputsBuilder(info))


def create_cached_processor(model_id: str, processor_cls):
    """Build a processor backed by a real multi-modal processor cache, and hand the
    cache back so a test can check it was actually used."""
    model_config = ModelConfig(model=model_id, model_impl="transformers")
    model_config.multimodal_config.mm_processor_cache_gb = 4
    ctx = InputProcessingContext(
        model_config, cached_tokenizer_from_config(model_config)
    )
    info = MultiModalProcessingInfo(ctx)
    cache = MultiModalProcessorOnlyCache(model_config)
    return processor_cls(info, MultiModalDummyInputsBuilder(info), cache=cache), cache
