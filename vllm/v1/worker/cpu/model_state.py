# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm import _custom_ops as ops
from vllm.v1.worker.gpu.mm.rope import RopeState
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.input_batch import InputBatch


class CPURopeState(RopeState):
    def prepare_positions(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        prefill_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        ops.prepare_mrope_positions_cpu(
            self.positions,
            self.prefill_positions.gpu,
            self.prefill_delta.gpu,
            idx_mapping,
            query_start_loc,
            prefill_lens,
            num_computed_tokens,
        )


class CPUDefaultModelState(DefaultModelState):
    rope_state_cls = CPURopeState


class CPUEncoderOnlyModelState(EncoderOnlyModelState):
    rope_state_cls = CPURopeState


class CPUMambaHybridModelState(MambaHybridModelState):
    rope_state_cls = CPURopeState

    def initialize_kv_cache(
        self,
        kv_cache_config: "KVCacheConfig",
        block_tables: "BlockTables",
    ) -> None:
        """Resolve align metadata without building the GPU pointer-copy context."""
        if self._align_mode:
            self._get_mamba_group_info(kv_cache_config)

    def preprocess_state(
        self,
        input_batch: "InputBatch",
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: "KVCacheConfig",
        num_computed_tokens: torch.Tensor,
    ) -> None:
        if not self._align_mode or input_batch.num_reqs == 0:
            return

        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        req_indices = input_batch.idx_mapping
        previous_state_indices = self._mamba_state_idx_gpu[req_indices]
        query_lens = input_batch.query_start_loc[1:] - input_batch.query_start_loc[:-1]
        computed_after = num_computed_tokens[req_indices] + query_lens
        current_state_indices = (
            computed_after + mamba_spec.block_size - 1
        ) // mamba_spec.block_size - 1
        self._mamba_state_idx_gpu[req_indices] = current_state_indices

        crossed = (previous_state_indices >= 0) & (
            previous_state_indices != current_state_indices
        )
        crossed_rows = crossed.nonzero(as_tuple=True)[0].tolist()
        self.num_accepted_tokens_gpu[req_indices[crossed]] = 1

        forward_context = self.vllm_config.compilation_config.static_forward_context
        for batch_idx in crossed_rows:
            src_col = previous_state_indices[batch_idx].item()
            dst_col = current_state_indices[batch_idx].item()
            for group_id in mamba_group_ids:
                block_table = block_tables[group_id]
                src_block = block_table[batch_idx, src_col].item()
                dst_block = block_table[batch_idx, dst_col].item()
                for layer_name in kv_cache_config.kv_cache_groups[group_id].layer_names:
                    for state in forward_context[layer_name].kv_cache:
                        state[dst_block].copy_(state[src_block])

    def _update_num_accepted_tokens(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
    ) -> None:
        sampled_tensor = num_sampled if isinstance(num_sampled, torch.Tensor) else None
        sampled_value = num_sampled if isinstance(num_sampled, int) else 0
        ops.update_num_accepted_tokens_cpu(
            idx_mapping,
            sampled_tensor,
            self.num_accepted_tokens_gpu,
            sampled_value,
        )
