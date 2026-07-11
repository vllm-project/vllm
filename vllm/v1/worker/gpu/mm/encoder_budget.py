# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.multimodal.registry import MultiModalRegistry
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.core.encoder_cache_manager import compute_mm_encoder_budget


@dataclass(frozen=True)
class EncoderCacheBudget:
    """Pure encoder cache budget data for multimodal profiling."""

    encoder_compute_budget: int
    encoder_cache_size: int
    mm_max_toks_per_item: Mapping[str, int]
    mm_max_items_per_batch: Mapping[str, int]

    def get_modality_with_max_tokens(self) -> str:
        return max(
            self.mm_max_toks_per_item.items(),
            key=lambda item: (item[1], item[0]),
        )[0]

    def get_encoder_budget(self) -> int:
        return min(self.encoder_compute_budget, self.encoder_cache_size)

    @classmethod
    def compute_encoder_cache_budget(
        cls,
        vllm_config: VllmConfig,
        mm_registry: MultiModalRegistry,
    ) -> "EncoderCacheBudget":
        """Compute pure multimodal encoder cache budget data for profiling."""

        with set_default_torch_num_threads():
            processor = mm_registry.create_processor(vllm_config.model_config)
            return _compute_encoder_cache_budget(
                vllm_config,
                mm_registry,
                processor=processor,
            )


def _get_max_items_per_batch(
    scheduler_config: SchedulerConfig,
    mm_limits: Mapping[str, int],
    max_model_len: int,
    encoder_budget: int,
    modality: str,
    max_tokens_per_item: int,
) -> int:
    if max_tokens_per_item == 0 or encoder_budget == 0:
        return 0

    max_encoder_items_per_batch = encoder_budget // max_tokens_per_item
    max_items_per_prompt = max(
        1,
        min(mm_limits[modality], max_model_len // max_tokens_per_item),
    )

    max_num_reqs = scheduler_config.max_num_seqs
    if not scheduler_config.enable_chunked_prefill:
        max_num_reqs = min(
            max_num_reqs,
            scheduler_config.max_num_batched_tokens // max_tokens_per_item,
        )

    max_decoder_items_per_batch = max_num_reqs * max_items_per_prompt
    return max(
        1,
        min(max_encoder_items_per_batch, max_decoder_items_per_batch),
    )


def _compute_encoder_cache_budget(
    vllm_config: VllmConfig,
    mm_registry: MultiModalRegistry,
    processor: BaseMultiModalProcessor,
) -> EncoderCacheBudget:
    model_config = vllm_config.model_config
    scheduler_config = vllm_config.scheduler_config
    mm_config = model_config.get_multimodal_config()
    enable_mm_embeds = mm_config is not None and mm_config.enable_mm_embeds

    supported_mm_limits = processor.info.supported_mm_limits
    mm_limits = processor.info.allowed_mm_limits
    tower_modalities = {
        modality for modality in supported_mm_limits if mm_limits.get(modality, 0) > 0
    }
    embed_only_modalities = {
        modality
        for modality in supported_mm_limits
        if enable_mm_embeds and mm_limits.get(modality, 0) == 0
    }
    active_modalities = tower_modalities | embed_only_modalities

    all_mm_max_toks_per_item = get_mm_max_toks_per_item(
        model_config,
        mm_registry,
        processor,
        mm_counts=dict.fromkeys(active_modalities, 1),
    )
    active_mm_max_toks_per_item = {
        modality: all_mm_max_toks_per_item[modality]
        for modality in active_modalities
        if modality in all_mm_max_toks_per_item
    }
    tower_mm_max_toks_per_item = {
        modality: active_mm_max_toks_per_item[modality]
        for modality in tower_modalities
        if modality in active_mm_max_toks_per_item
    }

    encoder_compute_budget, encoder_cache_size = compute_mm_encoder_budget(
        scheduler_config,
        active_mm_max_toks_per_item,
    )
    encoder_budget = min(encoder_compute_budget, encoder_cache_size)
    mm_max_items_per_batch = {
        modality: _get_max_items_per_batch(
            scheduler_config,
            mm_limits,
            model_config.max_model_len,
            encoder_budget,
            modality,
            max_toks_per_item,
        )
        for modality, max_toks_per_item in tower_mm_max_toks_per_item.items()
    }

    return EncoderCacheBudget(
        encoder_compute_budget=encoder_compute_budget,
        encoder_cache_size=encoder_cache_size,
        mm_max_toks_per_item=tower_mm_max_toks_per_item,
        mm_max_items_per_batch=mm_max_items_per_batch,
    )


EncoderProfileInputFactory = Callable[
    [str, int], list[tuple[str, MultiModalKwargsItem]]
]


def get_dummy_encoder_profile_inputs(
    model_config: ModelConfig,
    mm_registry: MultiModalRegistry,
    modality: str,
    max_items_per_batch: int,
) -> list[tuple[str, MultiModalKwargsItem]]:
    dummy_mm_inputs = mm_registry.get_dummy_mm_inputs(
        model_config,
        mm_counts={modality: 1},
    )
    dummy_mm_item = dummy_mm_inputs["mm_kwargs"][modality][0]
    assert dummy_mm_item is not None, "Dummy item should be generated"

    return [(modality, dummy_mm_item)] * max_items_per_batch
