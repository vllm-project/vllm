# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel

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

        # Dedicated buffer for query positions so the slice always starts at
        # offset 0 — required for CUDA graph address stability.
        self.query_positions = torch.zeros(
            self.max_positions,
            dtype=torch.int64,
            device=device,
        )

        # For DFlash we use the input embeddings to embed the mask token
        self.parallel_drafting_hidden_state_tensor = None

    @override
    def _raise_if_multimodal(self):
        # Override to allow multimodal inputs since DFlash supports Qwen3.5 models
        # Support for multimodal inputs has not been tested.
        pass

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

        # Copy hidden states (wide memcpy, not suitable for triton)
        self.hidden_states[:num_context] = target_hidden_states

        # Allocate output for token_indices_to_sample
        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # Launch fused triton kernel for input_ids, positions, slot_mapping,
        # and token_indices_to_sample
        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        BLOCK_SIZE = min(256, triton.next_power_of_2(max_tokens_per_req))
        num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
        grid = (batch_size, num_blocks)

        has_num_rejected = num_rejected_tokens_gpu is not None
        copy_and_expand_dflash_inputs_kernel[grid](
            # Inputs
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            # Outputs
            out_input_ids_ptr=self.input_ids,
            out_positions_ptr=self.positions,
            out_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            # Block table
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            # Metadata
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=(
                num_rejected_tokens_gpu if has_num_rejected else 0
            ),
            # Scalars
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            num_context=num_context,
            total_input_tokens=num_context,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_NUM_REJECTED=has_num_rejected,
        )

        # Save context slot_mapping for pre-insertion (clone because buffer
        # will be reused by _get_slot_mapping)
        self._dflash_context_slot_mapping = self._slot_mapping_buffer[
            :num_context
        ].clone()

        # Only query slot_mapping for the model forward pass — context KVs
        # are pre-inserted into cache before the forward.  Clone to avoid
        # aliasing with the buffer that _get_slot_mapping writes into.
        query_slot_mapping = self._slot_mapping_buffer[
            num_context:num_all_positions
        ].clone()
        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req

        # In padded mode, cad.seq_lens includes rejected tokens. Subtract
        # them so attention only sees the valid prefix of context states.
        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens + num_query_per_req,
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
            slot_mapping=query_slot_mapping,
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

        # Slot mapping sized to num_input_tokens (query only), matching
        # the K/V tensor size from the model forward.  Context KVs are
        # pre-inserted separately and don't flow through the model.
        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        # Positions must cover context (unpadded) + query (padded to
        # num_input_tokens) so that query_positions matches the model input.
        num_pos_padded = num_tokens + num_input_tokens
        all_positions = self._get_positions(num_pos_padded)
        context_states = self.hidden_states[:num_tokens]
        if all_positions.dim() == 1:
            context_positions = all_positions[:num_tokens]
            self.query_positions[:num_input_tokens] = all_positions[num_tokens:]
        else:
            context_positions = all_positions[:, :num_tokens]
            self.query_positions[:num_input_tokens] = all_positions[:, num_tokens:]

        # Run the KV projection (GEMM + norms + RoPE) for memory profiling,
        # but skip the cache write — dummy_run doesn't have valid
        # context slot_mappings.
        self.model.precompute_and_store_context_kv(context_states, context_positions)
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
                positions=self.query_positions[:num_input_tokens],
                inputs_embeds=None,
            )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int, dict[str, Any]]:
        # Input ids: (padded) query tokens
        # Positions: (unpadded) context tokens + (padded) query tokens
        # Hidden states: (unpadded) context tokens
        # Slot mapping size: unpadded number of positions
        num_context = self._dflash_num_context
        num_positions = num_context + num_input_tokens
        all_positions = self._get_positions(num_positions)
        if all_positions.dim() == 1:
            context_positions = all_positions[:num_context]
            self.query_positions[:num_input_tokens] = all_positions[num_context:]
        else:
            context_positions = all_positions[:, :num_context]
            self.query_positions[:num_input_tokens] = all_positions[:, num_context:]

        # Pre-insert context KVs directly into cache
        self.model.precompute_and_store_context_kv(
            self.hidden_states[:num_context],
            context_positions,
            self._dflash_context_slot_mapping,
        )
        return (
            dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self.query_positions[:num_input_tokens],
                inputs_embeds=None,
            ),
            num_input_tokens,
            {},
        )

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
