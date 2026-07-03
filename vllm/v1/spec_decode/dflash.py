# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig, replace
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
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
            # Request aux hidden states; DFlash turns them into context K/V.
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        # Only next_token_ids and mask tokens are query tokens, all other context is K/V
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        # Positions covers both context states + query states
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        # Separate context buffers to keep query buffer addresses stable for CUDA graphs
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._slot_mapping_buffers_by_gid: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._draft_block_size_by_gid: dict[int, int] = {}
        self._draft_block_tables: dict[int, torch.Tensor] = {}
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )

        self.arange = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32
        )

        # DFlash embeds mask tokens directly.
        self.parallel_drafting_hidden_state_tensor = None

    @override
    def allow_multiple_draft_kv_cache_groups(self) -> bool:
        return True

    @override
    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        self._draft_block_size_by_gid.clear()
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            self._draft_block_size_by_gid[gid] = (
                kernel_block_sizes[gid]
                if kernel_block_sizes is not None and gid < len(kernel_block_sizes)
                else attn_group.get_metadata_builder().kv_cache_spec.block_size
            )
        self._ensure_slot_mapping_buffers()

    def clear_draft_block_tables(self) -> None:
        self._draft_block_tables.clear()

    def set_draft_block_table(
        self,
        kv_cache_gid: int,
        block_table: torch.Tensor,
    ) -> None:
        if kv_cache_gid in self._draft_kv_cache_group_ids:
            self._draft_block_tables[kv_cache_gid] = block_table

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        return replace(
            base,
            attention_config=replace(
                base.attention_config,
                use_non_causal=True,
            ),
        )

    @override
    def _warn_if_multimodal(self):
        # Override to allow multimodal inputs since DFlash supports Qwen3.5 models
        # Support for multimodal inputs has not been tested.
        pass

    def _ensure_slot_mapping_buffers(self) -> None:
        gids = self._draft_kv_gids()

        first_gid = gids[0]
        for gid in gids:
            if gid in self._slot_mapping_buffers_by_gid:
                continue
            if gid == first_gid:
                self._slot_mapping_buffers_by_gid[gid] = (
                    self._context_slot_mapping_buffer,
                    self._slot_mapping_buffer,
                )
            else:
                self._slot_mapping_buffers_by_gid[gid] = (
                    torch.zeros(
                        self.max_num_tokens,
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    torch.zeros(
                        self.max_query_tokens,
                        dtype=torch.int64,
                        device=self.device,
                    ),
                )

    def _draft_kv_gids(self) -> list[int]:
        return self._draft_kv_cache_group_ids or [
            self.kv_cache_gid if self.kv_cache_gid >= 0 else 0
        ]

    def _get_dflash_block_table(
        self,
        kv_cache_gid: int,
        cad: CommonAttentionMetadata,
    ) -> torch.Tensor:
        block_table = self._draft_block_tables.get(kv_cache_gid)
        if block_table is not None:
            return block_table
        if kv_cache_gid == self.kv_cache_gid or self.kv_cache_gid < 0:
            return cad.block_table_tensor
        raise RuntimeError(
            "Missing DFlash KV metadata for draft KV cache group "
            f"{kv_cache_gid}. This is required when DFlash draft layers span "
            "multiple KV cache groups."
        )

    def _get_dflash_context_slot_mapping(
        self,
        num_context: int,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if not self._draft_layer_to_kv_cache_gid:
            return self._context_slot_mapping_buffer[:num_context]
        return {
            layer_name: self._slot_mapping_buffers_by_gid[
                self._draft_layer_to_kv_cache_gid[layer_name]
            ][0][:num_context]
            for layer_name in self._draft_attn_layer_names
        }

    @override
    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._ensure_slot_mapping_buffers()
        if self._draft_layer_to_kv_cache_gid:
            return {
                layer_name: self._slot_mapping_buffers_by_gid[
                    self._draft_layer_to_kv_cache_gid[layer_name]
                ][1][:num_tokens]
                for layer_name in self._draft_attn_layer_names
            }
        return super()._get_slot_mapping(num_tokens, slot_mapping)

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

        self._dflash_num_context = num_context

        # Context preprocessing does not run in a CUDA graph.
        self._dflash_hidden_states = target_hidden_states

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # Fill query inputs and per-KV-group slot mappings.
        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        BLOCK_SIZE = min(256, triton.next_power_of_2(max_tokens_per_req))
        num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
        grid = (batch_size, num_blocks)

        has_num_rejected = num_rejected_tokens_gpu is not None
        self._ensure_slot_mapping_buffers()
        draft_kv_group_ids = self._draft_kv_gids()
        for kv_cache_gid in draft_kv_group_ids:
            context_slot_mapping_buffer, query_slot_mapping_buffer = (
                self._slot_mapping_buffers_by_gid[kv_cache_gid]
            )
            block_table = self._get_dflash_block_table(kv_cache_gid, cad)
            copy_and_expand_dflash_inputs_kernel[grid](
                # Inputs
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                # Outputs
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=context_slot_mapping_buffer,
                out_query_slot_mapping_ptr=query_slot_mapping_buffer,
                out_token_indices_ptr=token_indices_to_sample,
                # Block table
                block_table_ptr=block_table,
                block_table_stride=block_table.stride(0),
                # Metadata
                query_start_loc_ptr=cad.query_start_loc,
                num_rejected_tokens_ptr=(
                    num_rejected_tokens_gpu if has_num_rejected else 0
                ),
                # Scalars
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=self._draft_block_size_by_gid.get(
                    kv_cache_gid, self.block_size
                ),
                num_query_per_req=num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=num_context,
                BLOCK_SIZE=BLOCK_SIZE,
                HAS_NUM_REJECTED=has_num_rejected,
            )

        primary_kv_cache_gid = draft_kv_group_ids[0]
        query_slot_mapping = self._slot_mapping_buffers_by_gid[primary_kv_cache_gid][1][
            :num_query_total
        ]
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
            block_table_tensor=self._get_dflash_block_table(primary_kv_cache_gid, cad),
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

        # Context and query positions use separate buffers; no copy needed.
        context_positions = self._context_positions_buffer[:num_tokens]
        # Context states will be passed directly to the precomputation without
        # going through the buffer, since no CUDA graph is used for the precomputation.
        # For the dummy run, we use the dummy buffer.
        context_states = self.hidden_states[:num_tokens]

        # Run the KV projection (GEMM + norms + RoPE) for memory profiling,
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
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        # Context and query positions/slots were written by the kernel.
        num_context = self._dflash_num_context

        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,  # Shape is already [num_context, hidden_size]
            self._context_positions_buffer[:num_context],
            self._get_dflash_context_slot_mapping(num_context),
        )
        return (
            dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            ),
            num_input_tokens,
        )

    @override
    def build_per_group_and_layer_attn_metadata(
        self, cad: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        self._ensure_slot_mapping_buffers()
        sliding_layer_names: set[str] = getattr(
            self.model, "sliding_attention_layer_names", set()
        )

        per_group: list[object] = []
        per_layer: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            kv_cache_gid = attn_group.kv_cache_group_id
            group_cad = cad.replace(
                block_table_tensor=self._get_dflash_block_table(kv_cache_gid, cad),
                slot_mapping=self._slot_mapping_buffers_by_gid[kv_cache_gid][1][
                    : cad.num_actual_tokens
                ],
                causal=False,
            )
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=group_cad,
                draft_index=draft_index,
            )
            per_group.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = attn_metadata

            # DFlash layers consume attention metadata through the per-layer
            # forward context. Keep the non-causal group metadata for
            # group-level spec decode checks, and specialize only the SWA
            # layers that need a causal sliding-window mask.
            causal_layers = sliding_layer_names & set(attn_group.layer_names)
            if causal_layers:
                causal_attn_metadata = (
                    attn_group.get_metadata_builder().build_for_drafting(
                        common_attn_metadata=group_cad.replace(causal=True),
                        draft_index=draft_index,
                    )
                )
                for layer_name in causal_layers:
                    per_layer[layer_name] = causal_attn_metadata

        for layer_name, attn_metadata in per_layer.items():
            if layer_name in sliding_layer_names:
                assert getattr(attn_metadata, "causal", None) is True, (
                    f"Attention metadata for sliding layer {layer_name} does not have"
                    " causal support, which is required for DFlash SWA."
                )
                continue
            assert getattr(attn_metadata, "causal", None) is False, (
                f"Attention metadata for layer {layer_name} does not have"
                " non-causal support, which is required for DFlash."
                " Consider using a different attention backend, such as FlashAttention."
            )
        return per_group, per_layer

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self):
        use_aux_hidden_state = True
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if dflash_config is not None:
            use_aux_hidden_state = dflash_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state
