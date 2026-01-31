# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.v1.worker.utils import sanity_check_mm_encoder_outputs


class EncoderRunner:
    def __init__(
        self,
        max_num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.max_num_tokens = max_num_tokens
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device

        self.inputs_embeds = torch.zeros(
            max_num_tokens,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        self.req_id_to_mm_features: dict[str, list[MultiModalFeatureSpec]] = {}
        self.encoder_cache: dict[str, torch.Tensor] = {}

    def reset_mm_cache(self) -> None:
        """
        Clear the multi-modal cache that was used during profiling,
        but no longer needed during inference.
        """
        # TODO: Implement MM budget
        pass

    def reset_encoder_cache(self) -> None:
        """Clear the GPU-side encoder cache storing vision embeddings.

        This should be called when model weights are updated to ensure
        stale embeddings computed with old weights are not reused.
        """
        self.encoder_cache.clear()

    def add_request(self, req_id: str, mm_features: list[MultiModalFeatureSpec]):
        self.req_id_to_mm_features[req_id] = mm_features

    def free_encoder_cache(self, mm_hash: str) -> None:
        self.encoder_cache.pop(mm_hash, None)

    def remove_request(self, req_id: str) -> None:
        self.req_id_to_mm_features.pop(req_id, None)

    def prepare_mm_inputs(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
    ) -> tuple[list[str], list[tuple[str, MultiModalKwargsItem]]]:
        mm_hashes: list[str] = []
        mm_kwargs: list[tuple[str, MultiModalKwargsItem]] = []
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            mm_features = self.req_id_to_mm_features[req_id]
            for mm_input_id in encoder_input_ids:
                mm_feature = mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue
                mm_hashes.append(mm_feature.identifier)
                mm_kwargs.append((mm_feature.modality, mm_feature.data))

        return mm_hashes, mm_kwargs

    @torch.inference_mode()
    def execute_mm_encoder(
        self,
        model: SupportsMultiModal,
        mm_hashes: list[str],
        mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
    ) -> list[torch.Tensor]:
        if not mm_hashes:
            return []

        encoder_outputs: list[torch.Tensor] = []
        for modality, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=False,
        ):
            curr_group_outputs = model.embed_multimodal(**mm_kwargs_group)
            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )
            encoder_outputs.extend(curr_group_outputs)

        # Cache the encoder outputs by mm_hash
        for mm_hash, output in zip(mm_hashes, encoder_outputs):
            self.encoder_cache[mm_hash] = output
        return encoder_outputs

    def gather_mm_embeddings(
        self,
        req_ids: list[str],
        total_num_scheduled_tokens: int,
        num_scheduled_tokens: np.ndarray,
        query_start_loc: np.ndarray,
        prefill_lens: np.ndarray,
        computed_prefill_lens: np.ndarray,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        is_prefilling = (computed_prefill_lens < prefill_lens).tolist()
        all_decode = not any(is_prefilling)
        if all_decode:
            # All decode requests, so no need to gather any embeddings.
            return [], torch.zeros(
                total_num_scheduled_tokens,
                dtype=torch.bool,
                device=self.device,
            )

        query_start = computed_prefill_lens.tolist()
        query_end = (computed_prefill_lens + num_scheduled_tokens).tolist()

        mm_embeds: list[torch.Tensor] = []
        is_mm_embed = torch.zeros(
            total_num_scheduled_tokens,
            dtype=torch.bool,
            device="cpu",
            pin_memory=True,
        )
        for i, req_id in enumerate(req_ids):
            if not is_prefilling[i]:
                # OPTIMIZATION: Skip decode requests.
                continue

            mm_features = self.req_id_to_mm_features[req_id]
            for mm_feature in mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                if start_pos >= query_end[i]:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= query_start[i]:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(query_start[i] - start_pos, 0)
                end_idx = min(query_end[i] - start_pos, num_encoder_tokens)
                assert start_idx < end_idx
                curr_embeds_start, curr_embeds_end = (
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                )
                # If there are no embeddings in the current range, we skip
                # gathering the embeddings.
                if curr_embeds_start == curr_embeds_end:
                    continue

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                req_start_pos = query_start_loc[i] + start_pos - query_start[i]
                is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (
                    True if is_embed is None else is_embed
                )
                mm_embeds.append(mm_embeds_item)

        # Copy the is_mm_embed tensor to the GPU.
        is_mm_embed = is_mm_embed.to(device=self.device, non_blocking=True)
        return mm_embeds, is_mm_embed

    @torch.inference_mode()
    def get_inputs_embeds(
        self,
        model: SupportsMultiModal,
        input_ids: torch.Tensor,
        mm_embeds: list[torch.Tensor],
        is_mm_embed: torch.Tensor,
    ) -> torch.Tensor:
        x = model.embed_input_ids(
            input_ids,
            multimodal_embeddings=mm_embeds,
            is_multimodal=is_mm_embed,
        )
        # Copy to the pre-allocated buffer for CUDA graphs.
        self.inputs_embeds[: x.shape[0]] = x
        return self.inputs_embeds
