# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class MambaHybridModelState(DefaultModelState):
    """Model state for hybrid attention + Mamba / linear-attention models."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )

        self.is_spec_decode_align_mode = (
            vllm_config.num_speculative_tokens > 0
            and vllm_config.cache_config.mamba_cache_mode == "align"
        )

        # Used for align mode + spec decoding.
        self.last_block_tables: tuple[torch.Tensor, ...] | None = None
        self.last_kv_cache_config: KVCacheConfig | None = None

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        req_states: RequestState | None = None,
        scheduled_spec_decode_tokens: dict[str, list[int]] | None = None,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()

        # During CUDAGraph capture, num_decode_draft_tokens_cpu and num_accepted_tokens
        # are created by attn_metadata_builder.build_for_cudagraph_capture, so we only
        # compute them during actual (non-capture) forward execution.
        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool)
        num_decode_draft_tokens_cpu = None
        num_accepted_tokens = None
        if not for_capture:
            assert req_states is not None
            assert scheduled_spec_decode_tokens is not None
            is_prefilling[: input_batch.num_reqs] = torch.from_numpy(
                req_states.num_computed_prefill_tokens[input_batch.idx_mapping_np]
                < req_states.prefill_len.np[input_batch.idx_mapping_np]
            )
            num_decode_draft_tokens_cpu = torch.full(
                (num_reqs,),
                -1,
                dtype=torch.int32,
                device="cpu",
            )
            for batch_idx, req_id in enumerate(input_batch.req_ids):
                draft_ids = scheduled_spec_decode_tokens.get(req_id)
                if draft_ids is None:
                    continue
                req_state_idx = req_states.req_id_to_index[req_id]
                if (
                    req_states.num_computed_prefill_tokens[req_state_idx]
                    >= req_states.prefill_len.np[req_state_idx]
                ):
                    num_decode_draft_tokens_cpu[batch_idx] = len(draft_ids)

            num_accepted_tokens = torch.ones(
                num_reqs,
                dtype=self.num_accepted_tokens_gpu.dtype,
                device=self.num_accepted_tokens_gpu.device,
            )
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping
            ]

        if self.is_spec_decode_align_mode:
            # Save state needed during postprocess_state.
            self.last_block_tables = block_tables
            self.last_kv_cache_config = kv_cache_config

        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            for_cudagraph_capture=for_capture,
        )

    def postprocess_state(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
    ) -> None:
        num_accepted_tokens = torch.clamp(num_sampled, min=1)
        # Chunked prefill does not sample a token, so num_sampled can be 0.
        # Mamba treats num_accepted_tokens=1 as the neutral non-spec value.
        self.num_accepted_tokens_gpu[input_batch.idx_mapping] = num_accepted_tokens
        # The last accepted SSM state must be copied from the staging
        # block to the running block to ensure that the next step's
        # committed block read is correct.
        self._copy_ssm_staging_to_committed(input_batch, num_accepted_tokens)

    def _copy_ssm_staging_to_committed(
        self,
        input_batch: InputBatch,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        """Copy SSM state from the staging block that holds the last-accepted
        token's state back to the running block (column 0 of the state indices
        tensor).
        """
        if not self.is_spec_decode_align_mode:
            return

        assert self.last_kv_cache_config is not None
        assert self.last_block_tables is not None

        needs_copy_mask = num_accepted_tokens > 1
        if not needs_copy_mask.any():
            # No draft tokens were accepted, and thus no draft staging
            # block SSM states need to be copied over.
            return

        fwd_ctx = self.vllm_config.compilation_config.static_forward_context
        # Compute the narrow window for the Mamba block tables to get the
        # staging physical block IDs.  We iterate over Mamba kv-cache
        # groups; they share the same MambaSpec.
        for idx, group in enumerate(self.last_kv_cache_config.kv_cache_groups):
            if not isinstance(group.kv_cache_spec, MambaSpec):
                # Skip non-Mamba groups.
                continue

            block_table = self.last_block_tables[idx]
            cache_mode = self.vllm_config.cache_config.mamba_cache_mode
            state_indices_tensor = mamba_get_block_table_tensor(
                block_table,
                input_batch.seq_lens,
                group.kv_cache_spec,
                cache_mode,
            )
            num_spec_tokens = self.vllm_config.num_speculative_tokens
            state_indices_tensor = state_indices_tensor[:, : 1 + num_spec_tokens]

            # Source is the staging block for the last accepted token.
            src_block = (
                (num_accepted_tokens - 1)
                .clamp(max=state_indices_tensor.size(1) - 1)
                .to(torch.int64)
            )
            src_phys = state_indices_tensor.gather(1, src_block.unsqueeze(1)).squeeze(1)
            # Destination is the running block.
            dst_phys = state_indices_tensor[:, 0]
            # Copy for every layer in the group.
            for layer_name in group.layer_names:
                layer = fwd_ctx[layer_name]
                ssm_state = layer.kv_cache[1]
                src_idx = src_phys[needs_copy_mask].long()
                dst_idx = dst_phys[needs_copy_mask].long()
                ssm_state[dst_idx] = ssm_state[src_idx]
