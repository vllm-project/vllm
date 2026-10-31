# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncsByType
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.cpu.block_table import CPUBlockTables
from vllm.v1.worker.cpu.encoder_cache import EncoderCache
from vllm.v1.worker.cpu.input_batch import InputBatch
from vllm.v1.worker.cpu.model_states.default import DefaultModelState
from vllm.v1.worker.mamba_utils import (
    get_mamba_group_ids,
    get_mamba_groups,
    validate_mamba_state_copy_funcs,
)


def _get_mamba_spec(kv_cache_group: KVCacheGroupSpec, layer_name: str) -> MambaSpec:
    spec = kv_cache_group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        spec = spec.kv_cache_specs[layer_name]
    assert isinstance(spec, MambaSpec)
    return spec


class MambaHybridModelState(DefaultModelState):
    """Model state for hybrid attention and recurrent CPU models."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.cache_config = vllm_config.cache_config
        self._align_mode = self.cache_config.mamba_cache_mode == "align"
        if self._align_mode:
            self._mamba_state_idx = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._mamba_group_ids: list[int] = []
            self._mamba_spec: MambaSpec | None = None
            self._mamba_state_copy_funcs: MambaStateCopyFuncsByType | None = None

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        if self._align_mode:
            mamba_block_size = self.cache_config.mamba_block_size
            assert mamba_block_size is not None
            self._mamba_state_idx[req_index] = (
                new_req_data.num_computed_tokens - 1
            ) // mamba_block_size

    def _get_mamba_group_info(
        self, kv_cache_config: KVCacheConfig
    ) -> tuple[list[int], MambaSpec]:
        if self._mamba_spec is None:
            mamba_groups = get_mamba_groups(kv_cache_config)
            mamba_spec = next(iter(mamba_groups))
            assert all(
                spec.block_size == mamba_spec.block_size
                and spec.num_speculative_blocks == mamba_spec.num_speculative_blocks
                and spec.mamba_cache_mode == mamba_spec.mamba_cache_mode
                for spec in mamba_groups
            ), "all mamba groups must share cache scheduling parameters"
            self._mamba_group_ids = get_mamba_group_ids(mamba_groups)
            self._mamba_spec = mamba_spec
        return self._mamba_group_ids, self._mamba_spec

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: CPUBlockTables
    ) -> None:
        if not self._align_mode:
            return
        self._get_mamba_group_info(kv_cache_config)
        if self._mamba_state_copy_funcs is None:
            mamba_groups = get_mamba_groups(kv_cache_config)
            mamba_types = {spec.mamba_type for spec in mamba_groups}
            copy_funcs = self.model.get_mamba_state_copy_funcs(mamba_types)
            validate_mamba_state_copy_funcs(mamba_groups, copy_funcs)
            self._mamba_state_copy_funcs = copy_funcs

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        if not self._align_mode or input_batch.num_reqs == 0:
            return

        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        req_indices = input_batch.idx_mapping
        previous_state_indices = self._mamba_state_idx[req_indices]
        query_lens = input_batch.query_start_loc[1:] - input_batch.query_start_loc[:-1]
        computed_after = num_computed_tokens[req_indices] + query_lens
        current_state_indices = (
            computed_after + mamba_spec.block_size - 1
        ) // mamba_spec.block_size - 1
        self._mamba_state_idx[req_indices] = current_state_indices

        crossed_rows = (
            (previous_state_indices >= 0)
            & (previous_state_indices != current_state_indices)
        ).nonzero(as_tuple=True)[0]
        if crossed_rows.numel() == 0:
            return

        copy_funcs_by_type = self._mamba_state_copy_funcs
        assert copy_funcs_by_type is not None
        forward_context = self.vllm_config.compilation_config.static_forward_context
        for batch_idx in crossed_rows.tolist():
            src_col = int(previous_state_indices[batch_idx])
            dst_col = int(current_state_indices[batch_idx])
            for group_id in mamba_group_ids:
                kv_cache_group = kv_cache_config.kv_cache_groups[group_id]
                block_ids = block_tables[group_id][batch_idx].tolist()
                dst_block_id = block_ids[dst_col]
                for layer_name in kv_cache_group.layer_names:
                    mamba_type = _get_mamba_spec(kv_cache_group, layer_name).mamba_type
                    copy_funcs = copy_funcs_by_type[mamba_type]
                    states = forward_context[layer_name].kv_cache
                    if len(states) < len(copy_funcs):
                        raise ValueError(
                            f"Expected at least {len(copy_funcs)} Mamba state "
                            f"tensors, got {len(states)}"
                        )
                    for state, copy_func in zip(states, copy_funcs):
                        # This runner rejects speculative decoding, so the
                        # accepted-token offset is always zero (1 - 1).
                        copy_spec = copy_func(state, block_ids, src_col, 1)
                        ctypes.memmove(
                            state[dst_block_id].data_ptr(),
                            copy_spec.start_addr,
                            copy_spec.num_elements * state.element_size(),
                        )
