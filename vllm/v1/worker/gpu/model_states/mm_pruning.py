# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.model_executor.models.interfaces import supports_multimodal_pruning
from vllm.v1.worker.gpu.input_batch import InputBatch
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
        self, model: nn.Module, rope_state: RopeState, inputs_embeds_size: int
    ) -> None:
        self.model = model
        self.rope_state = rope_state
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
        num_embeds_per_req: list[int],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> list[torch.Tensor]:
        """Target forward: split the appended mrope-position channels off each
        request's media embeddings, recompute the corrected mrope positions, and
        stage them back into RopeState. Returns the cleaned, flattened embeddings.

        num_embeds_per_req (aligned to input_batch.req_ids) gives the per-request
        segmentation of the flat mm_embeds list.
        """
        cleaned: list[torch.Tensor] = []
        pos = 0
        req_idx_list = input_batch.idx_mapping_np.tolist()
        prefill_lens_list = input_batch.prefill_len_np.tolist()
        num_computed_list = input_batch.num_computed_prefill_tokens_np.tolist()
        for num_req_embeds, req_idx, prefill_len, num_computed in zip(
            num_embeds_per_req, req_idx_list, prefill_lens_list, num_computed_list
        ):
            if num_req_embeds == 0:
                continue
            req_embeds = mm_embeds[pos : pos + num_req_embeds]
            pos += num_req_embeds

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


def maybe_create_mm_pruner(
    model_config: ModelConfig, model: nn.Module, rope_state: RopeState | None
) -> MultiModalPruner | None:
    """Create a MultiModalPruner if the model prunes embeddings and uses M-RoPE."""
    if (
        not rope_state
        or not rope_state.has_delta
        or not model_config.multimodal_config
        or not model_config.multimodal_config.is_multimodal_pruning_enabled()
        or not supports_multimodal_pruning(model)
    ):
        return None

    return MultiModalPruner(model, rope_state, model_config.get_inputs_embeds_size())
