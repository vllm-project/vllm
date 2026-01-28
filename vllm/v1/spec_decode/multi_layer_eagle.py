# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer

logger = init_logger(__name__)

PADDING_SLOT_ID = -1


class DraftInputStates:
    def __init__(
        self,
        len: int,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        self.len = len
        self.token_ids = token_ids
        self.hidden_states = hidden_states
        self.positions = positions
        self.slot_mapping = slot_mapping


class MultiLayerEagleProposer(EagleProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner)

        self.layer_num: int = getattr(
            self.speculative_config.draft_model_config.hf_text_config, "n_predict", 0
        )
        self.num_speculative_tokens: int = (
            self.speculative_config.num_speculative_tokens
        )
        if self.num_speculative_tokens != self.layer_num:
            logger.warning_once(
                "For multi_layer_eagle, num_speculative_tokens "
                "does not match layer_num, adjusting to layer_num"
            )
            self.num_speculative_tokens = self.layer_num
        self.running_req_ids: list[str] | None = None
        self.draft_input_states_pool: dict[str, DraftInputStates] = {}

    def set_running_req_ids(self, req_ids: list[str]):
        self.running_req_ids = req_ids

    def _get_draft_input_states(self, req_id: str, len: int) -> DraftInputStates:
        draft_input_states = self.draft_input_states_pool.get(req_id, None)
        assert draft_input_states is not None
        assert draft_input_states.len >= len
        return draft_input_states

    def clean_req_cache(self, req_id: str):
        self.draft_input_states_pool.pop(req_id, None)

    def adjust_input(
        self,
        batch_size: int,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, dict[str, Any]]:
        start_token_indices = common_attn_metadata.query_start_loc[:-1]
        start_token_pos = target_positions[start_token_indices]
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        query_start_loc_cpu_np = query_start_loc_cpu.numpy()
        start_token_indices_cpu = query_start_loc_cpu_np[:-1]
        end_token_indices_cpu = query_start_loc_cpu_np[1:] - 1
        last_token_indices_cpu = last_token_indices.cpu().numpy()
        start_token_pos_cpu = start_token_pos.cpu().numpy()

        prev_token_ids = target_token_ids
        prev_positions = target_positions
        prev_hidden_states = target_hidden_states

        for i in range(batch_size):
            last_token_index: int = int(last_token_indices_cpu[i])
            start_token_index: int = int(start_token_indices_cpu[i])
            end_token_index: int = int(end_token_indices_cpu[i])
            start_pos: int = int(start_token_pos_cpu[i])
            assert self.running_req_ids is not None
            req_id = self.running_req_ids[i]
            shift = min(end_token_index - last_token_index, start_pos)

            modify_last_token_index = last_token_index
            if shift > 0:

                def shift_input(
                    input: torch.Tensor,
                    cached: torch.Tensor,
                    start_token_index: int = start_token_index,
                    end_token_index: int = end_token_index,
                    shift: int = shift,
                ) -> torch.Tensor:
                    window_len = end_token_index - start_token_index + 1
                    dest = input.narrow(
                        0, start_token_index + shift, window_len - shift
                    )
                    # clone is used to ensure correctness in the case of
                    # overlap between src and dest
                    src = input.narrow(0, start_token_index, window_len - shift).clone()
                    dest.copy_(src)
                    head = input.narrow(0, start_token_index, shift)
                    head.copy_(cached[-shift:])
                    return input

                cached_input_state = self._get_draft_input_states(req_id, shift)
                prev_token_ids = shift_input(
                    prev_token_ids, cached_input_state.token_ids
                )
                prev_positions = shift_input(
                    prev_positions, cached_input_state.positions
                )
                prev_hidden_states = shift_input(
                    prev_hidden_states, cached_input_state.hidden_states
                )
                common_attn_metadata.slot_mapping = shift_input(
                    common_attn_metadata.slot_mapping, cached_input_state.slot_mapping
                )
                common_attn_metadata.seq_lens[i] -= shift
                common_attn_metadata.num_computed_tokens_cpu[i] -= shift
                common_attn_metadata.seq_lens_cpu[i] -= shift

                modify_last_token_index = last_token_index + shift
                last_token_indices[i] += shift

            cache_start_index = max(
                start_token_index, modify_last_token_index + 1 - self.layer_num
            )

            self.draft_input_states_pool[req_id] = DraftInputStates(
                len=modify_last_token_index + 1 - cache_start_index,
                token_ids=prev_token_ids[
                    cache_start_index : modify_last_token_index + 1
                ].clone(),
                hidden_states=prev_hidden_states[
                    cache_start_index : modify_last_token_index + 1
                ].clone(),
                positions=prev_positions[
                    cache_start_index : modify_last_token_index + 1
                ].clone(),
                slot_mapping=common_attn_metadata.slot_mapping[
                    cache_start_index : modify_last_token_index + 1
                ].clone(),
            )

        common_attn_metadata.max_seq_len = torch.max(
            common_attn_metadata.seq_lens
        ).item()

        if self.attn_metadata_builder is None:
            attn_metadata_builder = self._get_attention_metadata_builder()
        else:
            attn_metadata_builder = self.attn_metadata_builder

        assert isinstance(attn_metadata_builder, AttentionMetadataBuilder)

        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0
        )

        # FIXME: support hybrid kv for draft model (remove separate indexer)
        if self.draft_indexer_metadata_builder:
            draft_indexer_metadata = (
                self.draft_indexer_metadata_builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=0,
                )
            )
        else:
            draft_indexer_metadata = None

        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
        for layer_name in self.indexer_layer_names:
            assert draft_indexer_metadata is not None
            per_layer_attn_metadata[layer_name] = draft_indexer_metadata

        return (
            prev_token_ids,
            prev_positions,
            prev_hidden_states,
            attn_metadata,
            per_layer_attn_metadata,
        )

    def initial_inputs_for_forward(
        self,
        num_tokens: int,
        prev_token_ids: torch.Tensor,
        prev_positions: torch.Tensor,
        prev_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor,
        spec_step_idx: int = 0,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ):
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[: num_tokens - 1] = prev_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids
        self._set_positions(num_tokens, prev_positions)
        self.hidden_states[:num_tokens] = prev_hidden_states[:num_tokens]
        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )
        else:
            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
            )

    def draft_model_forward(
        self,
        num_tokens: int,
        per_layer_attn_metadata: dict[str, Any],
        last_token_indices: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        common_attn_metadata: CommonAttentionMetadata,
        spec_step_idx: int = 0,
    ):
        num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=num_tokens, num_tokens_padded=num_tokens
        )

        cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens_dp_padded
        )
        num_input_tokens = batch_desc.num_tokens

        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        if self.supports_mm_inputs:
            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = self.inputs_embeds[:num_input_tokens]

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(num_input_tokens),
            "hidden_states": self.hidden_states[:num_input_tokens],
            "inputs_embeds": inputs_embeds,
            "spec_step_idx": spec_step_idx,
        }

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                num_input_tokens, common_attn_metadata.slot_mapping
            ),
        ):
            last_hidden_states = self.model(**model_kwargs)

        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(
            sample_hidden_states, spec_step_idx=spec_step_idx
        )

        draft_token_ids = logits.argmax(dim=-1)

        return draft_token_ids, last_hidden_states

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        assert self.method == "mtp"
        assert self.runner is not None
        assert target_positions.dim() == 1, (
            "MultiLayerEagleProposer does not support M-RoPE yet; "
            f"got target_positions with shape {tuple(target_positions.shape)}"
        )

        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]

        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        (
            prev_token_ids,
            prev_positions,
            prev_hidden_states,
            attn_metadata,
            per_layer_attn_metadata,
        ) = self.adjust_input(
            batch_size=batch_size,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            last_token_indices=last_token_indices,
            common_attn_metadata=common_attn_metadata,
        )

        if isinstance(attn_metadata, TreeAttentionMetadata):
            raise NotImplementedError(
                "Tree attention is not supported for multi layer eagle."
            )

        if self.allowed_attn_types is not None and not isinstance(
            attn_metadata, self.allowed_attn_types
        ):
            raise ValueError(
                f"Unsupported attention metadata type for speculative "
                "decoding for multi layer eagle: "
                f"{type(attn_metadata)}. Supported types are: "
                f"{self.allowed_attn_types}"
            )

        # Generate the remaining draft tokens.
        draft_token_ids_list: list[torch.Tensor] = []

        for token_index in range(self.num_speculative_tokens):
            if token_index != 0:
                prev_token_ids = self.input_ids[:num_tokens].clone()
                next_token_ids = draft_token_ids_list[-1].int()

            self.initial_inputs_for_forward(
                num_tokens=num_tokens,
                prev_token_ids=prev_token_ids,
                prev_positions=prev_positions,
                prev_hidden_states=prev_hidden_states,
                next_token_ids=next_token_ids,
                last_token_indices=last_token_indices,
                spec_step_idx=token_index,
                mm_embed_inputs=mm_embed_inputs,
            )

            draft_token_ids, prev_hidden_states = self.draft_model_forward(
                num_tokens=num_tokens,
                per_layer_attn_metadata=per_layer_attn_metadata,
                last_token_indices=last_token_indices,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                spec_step_idx=token_index,
            )

            # Early exit if there is only one draft token to be generated.
            if self.num_speculative_tokens == 1:
                return draft_token_ids.view(-1, 1)

            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)

        return draft_token_ids

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        raise Exception(
            "speculative_config.disable_padded_drafter_batch"
            " is not supported now for MultiLayerEagleProposer."
        )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=num_tokens, num_tokens_padded=num_tokens
        )
        if use_cudagraphs:
            cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                num_tokens_dp_padded
            )
            num_input_tokens = batch_desc.num_tokens
        else:
            cudagraph_runtime_mode = CUDAGraphMode.NONE
            num_input_tokens = num_tokens_dp_padded
        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        # Make sure to use EAGLE's own buffer during cudagraph capture.
        if (
            self.attn_layer_names
            and slot_mappings is not None
            and self.attn_layer_names[0] in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        for fwd_idx in range(self.layer_num):
            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping_dict,
            ):
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]

                model_kwargs = {
                    "input_ids": input_ids,
                    "positions": self._get_positions(num_input_tokens),
                    "hidden_states": self.hidden_states[:num_input_tokens],
                    "inputs_embeds": inputs_embeds,
                    "spec_step_idx": fwd_idx,
                }

                self.model(**model_kwargs)
