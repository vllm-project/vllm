# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.model_executor.models.interfaces import supports_multimodal_pruning
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.rope import RopeState
from vllm.v1.worker.gpu.states import RequestState


class MultiModalPruner:
    """Recomputes M-RoPE positions for multimodal models that prune embeddings
    (e.g. Qwen2.5-VL / Qwen3-VL / Nemotron-Nano-VL Efficient Video Sampling).

    Pruning models append their mrope-position channels to the (variable-count)
    media embeddings from `embed_multimodal`. Those channels must be split off and
    used to recompute mrope positions before the embeddings are merged.
    """

    def __init__(
        self,
        model: nn.Module,
        rope_state: RopeState,
        encoder_cache: EncoderCache,
        inputs_embeds_size: int,
    ) -> None:
        self.model = model
        self.rope_state = rope_state
        self.encoder_cache = encoder_cache
        # The cleaned embedding width: pruning models append their mrope-position
        # channels as trailing columns, so embeds[:, :inputs_embeds_size] strips them.
        self.inputs_embeds_size = inputs_embeds_size

    def strip(self, mm_embeds: list[torch.Tensor]) -> list[torch.Tensor]:
        """Draft forward: strip the appended position channels only.

        Stripping is per-embedding, so no per-request segmentation is needed. The
        speculator reuses the target's already-recomputed positions, hence there is
        no position write-back here.
        """
        return [mm[:, : self.inputs_embeds_size] for mm in mm_embeds]

    def recompute(
        self,
        mm_embeds: list[torch.Tensor],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> list[torch.Tensor]:
        """Target forward: split the appended mrope-position channels off each
        request's media embeddings, recompute the corrected mrope positions, and
        stage them back into RopeState. Returns the cleaned, flattened embeddings.
        """
        cleaned: list[torch.Tensor] = []
        pos = 0
        req_idx_list = input_batch.idx_mapping_np.tolist()
        prefill_lens_list = input_batch.prefill_len_np.tolist()
        num_computed_list = input_batch.num_computed_prefill_tokens_np.tolist()
        num_scheduled_list = input_batch.num_scheduled_tokens.tolist()
        for batch_idx, req_id in enumerate(input_batch.req_ids):
            num_computed = num_computed_list[batch_idx]
            query_end = num_computed + num_scheduled_list[batch_idx]
            num_req_embeds = self._num_window_embeds(req_id, num_computed, query_end)
            if num_req_embeds == 0:
                continue
            req_embeds = mm_embeds[pos : pos + num_req_embeds]
            pos += num_req_embeds

            req_idx = req_idx_list[batch_idx]
            prefill_len = prefill_lens_list[batch_idx]
            input_ids = req_states.all_token_ids.gpu[req_idx, :prefill_len]
            mrope_positions = self.rope_state.read_prefill_positions(
                req_idx, prefill_len
            ).long()
            req_cleaned, new_positions, delta = self.model.recompute_mrope_positions(
                input_ids=input_ids,
                multimodal_embeddings=req_embeds,
                mrope_positions=mrope_positions,
                num_computed_tokens=num_computed,
            )
            self.rope_state.update_prefill_positions(req_idx, new_positions, delta)
            cleaned.extend(req_cleaned)

        assert pos == len(mm_embeds)
        return cleaned

    def _num_window_embeds(self, req_id: str, query_start: int, query_end: int) -> int:
        """Count the media items contributing embeddings to [query_start,
        query_end), mirroring EncoderRunner.gather_mm_embeddings' per-request
        windowing so the flat mm_embeds list can be re-segmented per request.

        Note: This logic is intentionally duplicated here rather than being emitted
        from gather_mm_embeddings, to keep the main path cleaner, since this is a niche
        feature.
        """
        mm_features = self.encoder_cache.mm_features[req_id]
        lo, hi = get_mm_features_in_window(
            mm_features, start=query_start, end=query_end
        )
        count = 0
        for mm_feature in mm_features[lo:hi]:
            pos_info = mm_feature.mm_position
            start_idx = max(query_start - pos_info.offset, 0)
            end_idx = min(query_end - pos_info.offset, pos_info.length)
            embeds_start, embeds_end = pos_info.get_embeds_indices_in_range(
                start_idx, end_idx
            )
            if embeds_start != embeds_end:
                count += 1
        return count


def maybe_create_mm_pruner(
    model_config: ModelConfig,
    model: nn.Module,
    rope_state: RopeState | None,
    encoder_cache: EncoderCache | None,
) -> MultiModalPruner | None:
    """Create a MultiModalPruner if the model prunes embeddings and uses M-RoPE."""
    if (
        rope_state is None
        or not rope_state.has_delta
        or encoder_cache is None
        or model_config.multimodal_config is None
        or not model_config.multimodal_config.is_multimodal_pruning_enabled()
        or not supports_multimodal_pruning(model)
    ):
        return None

    return MultiModalPruner(
        model, rope_state, encoder_cache, model_config.get_inputs_embeds_size()
    )
