# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping

from vllm.config import ModelConfig, VllmConfig
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.multimodal.registry import MultiModalRegistry
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.core.encoder_cache_manager import compute_mm_encoder_budget


def get_mm_max_toks_per_item(
    model_config: ModelConfig,
    mm_registry: MultiModalRegistry,
    processor: BaseMultiModalProcessor,
    mm_counts: Mapping[str, int],
) -> Mapping[str, int]:
    """
    Get the maximum number of tokens per data item from each modality based
    on underlying model configuration.
    """
    max_tokens_per_item = processor.info.get_mm_max_tokens_per_item(
        seq_len=model_config.max_model_len,
        mm_counts=mm_counts,
    )
    if max_tokens_per_item is not None:
        return max_tokens_per_item

    mm_inputs = mm_registry.get_dummy_mm_inputs(
        model_config,
        mm_counts=mm_counts,
        processor=processor,
    )

    return {
        modality: sum(item.get_num_embeds for item in placeholders)
        for modality, placeholders in mm_inputs["mm_placeholders"].items()
    }


class MultiModalBudget:
    """Helper class to calculate budget information for multi-modal models."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        mm_registry: MultiModalRegistry,
    ) -> None:
        super().__init__()

        self.model_config = model_config = vllm_config.model_config
        self.scheduler_config = scheduler_config = vllm_config.scheduler_config

        self.max_model_len = model_config.max_model_len
        self.max_num_reqs = scheduler_config.max_num_seqs

        cache = mm_registry.processor_only_cache_from_config(vllm_config)
        processor = mm_registry.create_processor(model_config, cache=cache)

        self.cache = cache
        self.mm_limits = mm_limits = processor.info.allowed_mm_limits

        active_modalities = {
            modality for modality, limit in mm_limits.items() if limit > 0
        }

        with set_default_torch_num_threads():  # Avoid hang during startup
            all_mm_max_toks_per_item = get_mm_max_toks_per_item(
                model_config,
                mm_registry,
                processor,
                mm_counts=dict.fromkeys(active_modalities, 1),
            )

        # Some models (e.g., Qwen3Omni with use_audio_in_video=True) share
        # placeholders between modalities, so not all active modalities will
        # have their own entry in the returned dict. We filter to only include
        # modalities that have independent placeholder tokens.
        mm_max_toks_per_item = {
            modality: all_mm_max_toks_per_item[modality]
            for modality in active_modalities
            if modality in all_mm_max_toks_per_item
        }

        encoder_compute_budget, encoder_cache_size = compute_mm_encoder_budget(
            scheduler_config,
            mm_max_toks_per_item,
        )

        self.encoder_compute_budget = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size

        mm_max_items_per_prompt = dict[str, int]()
        mm_max_items_per_batch = dict[str, int]()

        for modality, max_toks_per_item in mm_max_toks_per_item.items():
            (
                mm_max_items_per_prompt[modality],
                mm_max_items_per_batch[modality],
            ) = self._get_max_items(modality, max_toks_per_item)

        self.mm_max_toks_per_item = mm_max_toks_per_item
        self.mm_max_items_per_prompt: Mapping[str, int] = mm_max_items_per_prompt
        self.mm_max_items_per_batch: Mapping[str, int] = mm_max_items_per_batch

    def _get_max_items(
        self,
        modality: str,
        max_tokens_per_item: int,
    ) -> tuple[int, int]:
        if max_tokens_per_item == 0:
            return 0, 0

        # Check how many items of this modality can be supported by
        # the encoder budget.
        if (encoder_budget := self.get_encoder_budget()) == 0:
            return 0, 0

        max_encoder_items_per_batch = encoder_budget // max_tokens_per_item

        # Check how many items of this modality can be supported by
        # the decoder budget.
        mm_limit = self.mm_limits[modality]

        max_items_per_prompt = max(
            1,
            min(mm_limit, self.max_model_len // max_tokens_per_item),
        )

        scheduler_config = self.scheduler_config
        max_num_reqs = self.max_num_reqs

        if not scheduler_config.enable_chunked_prefill:
            max_num_reqs = min(
                max_num_reqs,
                scheduler_config.max_num_batched_tokens // max_tokens_per_item,
            )

        max_decoder_items_per_batch = max_num_reqs * max_items_per_prompt

        max_items_per_batch = max(
            1,
            min(max_encoder_items_per_batch, max_decoder_items_per_batch),
        )

        return max_items_per_prompt, max_items_per_batch

    def get_modality_with_max_tokens(self) -> str:
        mm_max_toks_per_item = self.mm_max_toks_per_item
        modality, _ = max(mm_max_toks_per_item.items(), key=lambda x: x[1])

        return modality

    def get_encoder_budget(self) -> int:
        return min(self.encoder_compute_budget, self.encoder_cache_size)

    def reset_cache(self) -> None:
        if self.cache is not None:
            self.cache.clear_cache()
