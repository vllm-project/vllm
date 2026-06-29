# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.model_executor.models.interfaces import SupportsMultiModal, supports_realtime
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.utils import get_mm_features_in_window, group_and_batch_mm_kwargs
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.utils import sanity_check_mm_encoder_outputs


class EncoderRunner:
    def __init__(
        self,
        model: SupportsMultiModal,
        max_num_tokens: int,
        hidden_size: int,
        encoder_cache: EncoderCache,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.model = model
        self.max_num_tokens = max_num_tokens
        self.hidden_size = hidden_size
        self.encoder_cache = encoder_cache
        self.dtype = dtype
        self.device = device
        self.is_realtime = supports_realtime(model)

        self.inputs_embeds = torch.zeros(
            max_num_tokens, hidden_size, dtype=dtype, device=device
        )

    def prepare_mm_inputs(
        self, scheduled_encoder_inputs: dict[str, list[int]]
    ) -> tuple[list[str], list[tuple[str, MultiModalKwargsItem]]]:
        mm_hashes: list[str] = []
        mm_kwargs: list[tuple[str, MultiModalKwargsItem]] = []
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            mm_features = self.encoder_cache.mm_features[req_id]
            for mm_input_id in encoder_input_ids:
                mm_feature = mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue
                mm_hashes.append(mm_feature.identifier)
                mm_kwargs.append((mm_feature.modality, mm_feature.data))

        return mm_hashes, mm_kwargs

    @torch.inference_mode()
    def execute_mm_encoder(
        self, mm_kwargs: list[tuple[str, MultiModalKwargsItem]]
    ) -> list[torch.Tensor]:
        encoder_outputs: list[torch.Tensor] = []
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs, device=self.device, pin_memory=True
        ):
            batch_outputs = self.model.embed_multimodal(**mm_kwargs_batch)
            sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=num_items)
            encoder_outputs.extend(batch_outputs)
        return encoder_outputs

    def gather_mm_embeddings(
        self,
        req_ids: list[str],
        total_num_scheduled_tokens: int,
        num_scheduled_tokens: np.ndarray,
        query_start_loc: np.ndarray,
        prefill_lens: np.ndarray,
        num_computed_tokens: np.ndarray,
        draft_lookahead: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if draft_lookahead:
            num_computed_tokens = num_computed_tokens + draft_lookahead

        is_mm_embed = torch.zeros(
            total_num_scheduled_tokens, dtype=torch.bool, device="cpu"
        )

        # Whether to gather media embeddings this step.
        exclude_embeddings: list[bool] | None = None
        if not self.is_realtime:
            # Non-realtime models only have media embeddings within the prompt.
            is_decode = num_computed_tokens >= prefill_lens
            if is_decode.all():
                # All decode requests, so no need to gather any embeddings.
                return [], is_mm_embed
            exclude_embeddings = is_decode.tolist()

        query_start = num_computed_tokens.tolist()
        query_end = (num_computed_tokens + num_scheduled_tokens).tolist()

        mm_embeds: list[torch.Tensor] = []
        for i, req_id in enumerate(req_ids):
            if exclude_embeddings is not None and exclude_embeddings[i]:
                continue

            cur_query_start = query_start[i]
            cur_query_end = query_end[i]

            mm_features = self.encoder_cache.mm_features[req_id]
            lo, hi = get_mm_features_in_window(
                mm_features, start=cur_query_start, end=cur_query_end
            )
            for idx in range(lo, hi):
                mm_feature = mm_features[idx]
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                start_idx = max(cur_query_start - start_pos, 0)
                end_idx = min(cur_query_end - start_pos, num_encoder_tokens)
                assert start_idx < end_idx
                curr_embeds_start, curr_embeds_end = (
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                )
                # If there are no embeddings in the current range, we skip
                # gathering the embeddings.
                if curr_embeds_start == curr_embeds_end:
                    continue

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.encoder_outputs.get(mm_hash, None)
                if encoder_output is None:
                    # A feature starting at/after the processed boundary is only
                    # reached via the drafter's +1 look-ahead and might not be
                    # encoded yet; fall back to the token embedding for drafting.
                    if start_pos + draft_lookahead >= cur_query_end:
                        continue
                    raise RuntimeError(f"Encoder cache miss for {mm_hash}.")

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                req_start_pos = query_start_loc[i] + start_pos - cur_query_start
                is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] |= (
                    True if is_embed is None else is_embed
                )
                mm_embeds.append(mm_embeds_item)

        return mm_embeds, is_mm_embed

    @torch.inference_mode()
    def get_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        mm_embeds: list[torch.Tensor],
        is_mm_embed: torch.Tensor,
    ) -> torch.Tensor:
        x = self.model.embed_input_ids(
            input_ids, multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed
        )
        # Copy to the pre-allocated buffer for CUDA graphs.
        self.inputs_embeds[: x.shape[0]] = x
        return self.inputs_embeds
