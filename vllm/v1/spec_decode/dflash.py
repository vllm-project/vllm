# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer

logger = init_logger(__name__)


class DFlashProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash"
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        # Only next_token_ids and mask tokens are query tokens, all other context is K/V
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        # Positions covers both context states + query states
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        # Expand data structures related to positions, since they must hold enough
        # slots for the query tokens and a full set of context tokens
        self._slot_mapping_buffer = torch.zeros(
            self.max_positions,
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.zeros(
            self.max_positions,
            dtype=torch.int64,
            device=device,
        )
        self.arange = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32
        )

        # For DFlash we use the input embeddings to embed the mask token
        self.parallel_drafting_hidden_state_tensor = None

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        # DFlash cross-attention: context K/V from target hidden states,
        # Q from query embeddings (bonus + mask tokens).
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req
        num_all_positions = num_context + num_query_total

        # Store for build_model_inputs_first_pass to use
        self._dflash_num_context = num_context
        self._dflash_num_positions = num_all_positions

        # input_ids: [next_token, mask, mask, ...] for each request
        # First write all slots to the mask id
        self.input_ids[:num_query_total] = self.parallel_drafting_token_id
        # Then write the first slot of each query to the next_token_id
        self.input_ids[self.arange[:batch_size] * num_query_per_req] = next_token_ids

        # positions: [context_positions, query_positions]
        # Context positions are the target positions
        self.positions[:num_context] = target_positions
        # Query positions extend from each request's last position
        last_pos = target_positions[cad.query_start_loc[1:] - 1]
        offsets = self.arange[1 : num_query_per_req + 1]
        self.positions[num_context:num_all_positions] = (
            last_pos.unsqueeze(1) + offsets
        ).flatten()

        self.hidden_states[:num_context] = target_hidden_states

        # slot mapping also covers all K/V tokens (context + query)
        block_size = self.block_size
        all_positions = self.positions[:num_all_positions]
        block_numbers = all_positions // block_size

        req_lens = cad.query_start_loc[1:] - cad.query_start_loc[:-1]
        ctx_req_idx = torch.repeat_interleave(
            self.arange[:batch_size], req_lens, output_size=num_context
        )
        q_req_idx = torch.repeat_interleave(self.arange[:batch_size], num_query_per_req)
        req_idx = torch.cat([ctx_req_idx, q_req_idx])

        block_ids = cad.block_table_tensor[req_idx, block_numbers.long()]
        slot_mapping = block_ids * block_size + (all_positions % block_size)

        # skip index 0 (bonus token), sample from the indices of the mask tokens
        base = self.arange[:batch_size] * num_query_per_req
        offsets = self.arange[1 : self.num_speculative_tokens + 1]
        token_indices_to_sample = (base.unsqueeze(1) + offsets).flatten()

        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req
        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=cad.seq_lens + num_query_per_req,
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * num_query_per_req
            ),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=cad.max_seq_len + num_query_per_req,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=slot_mapping,
            causal=False,  # Non-causal attention is required for DFlash
        )

        return num_query_total, token_indices_to_sample, new_cad

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """
        Key differences to default dummy_run:
        - Only one forward pass due to parallel drafting
        - DFlash uses context states as unpadded metadata, so hidden_states will
        use the unpadded num_tokens instead of num_input_tokens
        - max_query_tokens is quite small, DFlash only sees spec tokens as queries
        - Multimodal inputs are not currently supported
        """
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
        )

        # Unpadded context + padded query
        num_positions = num_tokens + num_query_tokens

        # Make sure to use EAGLE's own buffer during cudagraph capture.
        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping_dict,
        ):
            self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_positions),
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=None,
            )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        # Input ids: (padded) query tokens
        # Positions: (unpadded) context tokens + (padded) query tokens
        # Hidden states: (unpadded) context tokens
        # Slot mapping size: unpadded number of positions
        num_positions = self._dflash_num_context + num_input_tokens
        return dict(
            input_ids=self.input_ids[:num_input_tokens],
            positions=self._get_positions(num_positions),
            hidden_states=self.hidden_states[: self._dflash_num_context],
            inputs_embeds=None,
        ), self._dflash_num_positions

    @override
    def build_per_layer_attn_metadata(
        self, cad: CommonAttentionMetadata, draft_index: int = 0
    ) -> dict[str, object]:
        per_layer_attention_metadata = super().build_per_layer_attn_metadata(
            cad, draft_index
        )
        for layer_name, attn_metadata in per_layer_attention_metadata.items():
            assert getattr(attn_metadata, "causal", None) is False, (
                f"Attention metadata for layer {layer_name} does not have"
                " non-causal support, which is required for DFlash."
                " Consider using a different attention backend, such as FlashAttention."
            )
        return per_layer_attention_metadata

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self):
        use_aux_hidden_state = True
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if dflash_config is not None:
            use_aux_hidden_state = dflash_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state
