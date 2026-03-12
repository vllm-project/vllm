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
        logger.info("DFlash is enabled.")
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash"
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
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

        # Store for propose() to use
        self._dflash_num_context = num_context
        self._dflash_num_positions = num_all_positions

        # 1. Set up query input_ids: [next_token, mask, mask, ...] x batch
        for i in range(batch_size):
            start = i * num_query_per_req
            self.input_ids[start] = next_token_ids[i]
            self.input_ids[start + 1 : start + num_query_per_req] = (
                self.parallel_drafting_token_id
            )

        # 2. Set up positions: [context_positions, query_positions]
        # Context positions are the target positions
        self.positions[:num_context] = target_positions
        # Query positions extend from each request's last position
        query_start_loc = cad.query_start_loc
        for i in range(batch_size):
            ctx_end = query_start_loc[i + 1].item()
            last_pos = target_positions[ctx_end - 1].item()
            q_start = num_context + i * num_query_per_req
            for j in range(num_query_per_req):
                self.positions[q_start + j] = last_pos + 1 + j

        # 3. Context hidden states (combined target HS)
        self.hidden_states[:num_context] = target_hidden_states

        # 4. Compute slot mapping for ALL K/V tokens (context + query)
        block_size = self.block_size
        all_positions = self.positions[:num_all_positions]
        block_numbers = all_positions // block_size
        # For context tokens, use the original block table
        # For query tokens, also use block table (same requests)
        # Build expanded block table index for gathering
        slot_mapping = torch.empty(
            num_all_positions, dtype=torch.int64, device=self.device
        )
        for i in range(batch_size):
            # Context tokens for request i
            ctx_start = query_start_loc[i].item()
            ctx_end = query_start_loc[i + 1].item()
            ctx_block_nums = block_numbers[ctx_start:ctx_end]
            ctx_block_ids = cad.block_table_tensor[i].gather(0, ctx_block_nums.long())
            slot_mapping[ctx_start:ctx_end] = (
                ctx_block_ids * block_size
                + all_positions[ctx_start:ctx_end] % block_size
            )
            # Query tokens for request i
            q_start = num_context + i * num_query_per_req
            q_end = q_start + num_query_per_req
            q_block_nums = block_numbers[q_start:q_end]
            q_block_ids = cad.block_table_tensor[i].gather(0, q_block_nums.long())
            slot_mapping[q_start:q_end] = (
                q_block_ids * block_size + all_positions[q_start:q_end] % block_size
            )

        # 5. Token indices to sample: mask positions in query output
        # Query output has num_query_per_req tokens per request.
        # Skip index 0 (bonus token), take indices 1..num_spec_tokens.
        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        for i in range(batch_size):
            base = i * num_query_per_req
            for j in range(self.num_speculative_tokens):
                token_indices_to_sample[i * self.num_speculative_tokens + j] = (
                    base + 1 + j
                )

        # 6. Build attention metadata for the query tokens.
        # DFlash uses non-causal attention.
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
            causal=False,
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
