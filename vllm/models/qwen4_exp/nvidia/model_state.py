# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from ..common.qsa import PCP_GLOBAL_METADATA_SUFFIX
from .ngram_embedding import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLEHostStagedEmbedding,
    stage_host_rows,
)


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    @staticmethod
    def _pcp_global_positions(
        local_positions: torch.Tensor,
        global_positions: torch.Tensor,
    ) -> torch.Tensor:
        if local_positions.ndim == 1:
            return global_positions
        return global_positions.unsqueeze(0).expand(local_positions.shape[0], -1)

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self._pcp_mamba_block_table_ptrs: tuple[int, ...] | None = None
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        # PLE runs inside captured regions, so these buffers keep a fixed shape
        # and address as the active request count changes between replays.
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )
        self.host_staged_modules = [
            (module, module.ngram_embedding)
            for module in model.modules()
            if isinstance(module, Qwen4ExpNGramEmbedding)
            and isinstance(module.ngram_embedding, Qwen4ExpPLEHostStagedEmbedding)
        ]

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        context = self.ngram_context
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        pcp = input_batch.pcp
        if pcp is not None:
            global_batch = pcp.global_batch
            local_positions = model_inputs.get("positions", input_batch.positions)
            model_inputs.update(
                pcp_hidden_restore_idx=pcp.hidden_restore_idx,
                pcp_local_gather_idx=pcp.local_gather_idx,
                pcp_global_positions=self._pcp_global_positions(
                    local_positions, global_batch.positions
                ),
                pcp_global_input_ids=global_batch.input_ids,
            )
        if not self.uses_ngram_embedding:
            return model_inputs
        if pcp is not None:
            global_batch = pcp.global_batch
            ngram_context = self._prepare_ngram_context(global_batch, req_states)
            model_inputs.update(
                query_start_loc=global_batch.query_start_loc,
                ngram_context=ngram_context,
            )
            if self.host_staged_modules:
                stage_host_rows(
                    self.host_staged_modules,
                    global_batch.input_ids,
                    global_batch.query_start_loc,
                    ngram_context,
                )
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc
        query_start_loc[: num_reqs_padded + 1].copy_(input_batch.query_start_loc)
        # Represent unused capacity as trailing zero-length requests.
        query_start_loc[num_reqs_padded + 1 :].copy_(input_batch.query_start_loc[-1])
        ngram_context = self._prepare_ngram_context(input_batch, req_states)
        if self.host_staged_modules:
            stage_host_rows(
                self.host_staged_modules,
                input_batch.input_ids,
                query_start_loc,
                ngram_context,
            )
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        return model_inputs

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        pcp = input_batch.pcp
        if pcp is not None:
            assert pcp.global_block_tables is not None
            ctx = getattr(self, "_mamba_ctx", None)
            if ctx is not None:
                tables = [pcp.global_block_tables[gid] for gid in ctx.mamba_group_ids]
                ptrs = tuple(table.data_ptr() for table in tables)
                if ptrs != self._pcp_mamba_block_table_ptrs:
                    assert all(
                        table.stride(0) == ctx.block_table_stride_req
                        for table in tables
                    )
                    # State copies retain the persistent per-request tables.
                    # GDN indices need the PCP batch's global token order.
                    async_tensor_h2d(list(ptrs), out=ctx.aligned_index_block_table_ptrs)
                    self._pcp_mamba_block_table_ptrs = ptrs
            input_batch = pcp.global_batch
            block_tables = pcp.global_block_tables
        super().preprocess_state(
            input_batch, block_tables, kv_cache_config, num_computed_tokens
        )

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
        metadata_builder_idx: int | None = None,
    ) -> dict[str, Any]:
        pcp = input_batch.pcp
        local_attn_groups = attn_groups
        if pcp is not None:
            # Recurrent layers run once per global request, not per PCP segment.
            local_attn_groups = [
                [
                    group
                    for group in groups
                    if group.backend.get_name()
                    not in {"GDN_ATTN", "PLE_SHORT_CONV_ATTN"}
                ]
                for groups in attn_groups
            ]
        metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            local_attn_groups,
            kv_cache_config,
            for_capture,
            ubatch_idx,
            model_specific_attn_metadata,
            metadata_builder_idx=metadata_builder_idx,
        )
        if pcp is None:
            return metadata
        assert cudagraph_mode == CUDAGraphMode.NONE
        assert pcp.global_block_tables is not None
        assert pcp.global_slot_mappings is not None
        global_metadata = super().prepare_attn(
            pcp.global_batch,
            cudagraph_mode,
            pcp.global_block_tables,
            pcp.global_slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            0,
            model_specific_attn_metadata,
            metadata_builder_idx=1,
        )
        for layer_name, layer_metadata in global_metadata.items():
            if isinstance(
                layer_metadata,
                (GDNAttentionMetadata, PleShortConvAttentionMetadata),
            ):
                metadata[layer_name] = layer_metadata
            else:
                metadata[layer_name + PCP_GLOBAL_METADATA_SUFFIX] = layer_metadata
        return metadata

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc = self.ple_query_start_loc
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1 : num_reqs + 1])
        query_start_loc[num_reqs + 1 :].fill_(num_tokens)

        ngram_context = self.ngram_context
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        return model_inputs


__all__ = ["Qwen4ExpModelState"]
