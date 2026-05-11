# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_temporal_copy_spec,
    is_conv_state_dim_first,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup


@triton.jit
def _reset_mamba_accepted_tokens_kernel(
    idx_mapping_ptr,
    state_idx_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
    num_accepted_tokens_ptr,
    num_reqs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_reqs
    req_indices = tl.load(idx_mapping_ptr + offsets, mask=mask, other=0)
    src_block_idx = tl.load(state_idx_ptr + req_indices, mask=mask, other=-1)
    num_computed = tl.load(num_computed_tokens_ptr + req_indices, mask=mask, other=0)
    query_start = tl.load(query_start_loc_ptr + offsets, mask=mask, other=0)
    query_end = tl.load(query_start_loc_ptr + offsets + 1, mask=mask, other=0)
    computed_after = num_computed + query_end - query_start
    dest_block_idx = (computed_after + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1
    should_reset = (src_block_idx >= 0) & (src_block_idx != dest_block_idx)
    tl.store(state_idx_ptr + req_indices, dest_block_idx, mask=mask)
    tl.store(
        num_accepted_tokens_ptr + req_indices,
        1,
        mask=mask & should_reset,
    )


@triton.jit
def _reset_mamba_postprocess_accepted_tokens_kernel(
    idx_mapping_ptr,
    state_idx_ptr,
    num_computed_tokens_ptr,
    num_accepted_tokens_ptr,
    num_reqs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_reqs
    req_indices = tl.load(idx_mapping_ptr + offsets, mask=mask, other=0)
    accepted = tl.load(num_accepted_tokens_ptr + req_indices, mask=mask, other=1)
    new_computed = tl.load(num_computed_tokens_ptr + req_indices, mask=mask, other=0)
    running_tokens = new_computed - accepted + 1
    aligned = (new_computed // MAMBA_BLOCK_SIZE) * MAMBA_BLOCK_SIZE
    dest_block_idx = aligned // MAMBA_BLOCK_SIZE - 1
    src_block_idx = tl.load(state_idx_ptr + req_indices, mask=mask, other=-1)
    should_reset = (aligned >= running_tokens) & (src_block_idx == dest_block_idx)
    tl.store(num_accepted_tokens_ptr + req_indices, 1, mask=mask & should_reset)


@triton.jit
def _get_mamba_copy_indices(
    batch_idx,
    idx_mapping_ptr,
    block_table_ptr,
    block_table_stride: tl.constexpr,
    state_idx_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
    num_accepted_tokens_ptr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    POSTPROCESS: tl.constexpr,
    TEMPORAL: tl.constexpr,
):
    req_idx = tl.load(idx_mapping_ptr + batch_idx)
    src_block_idx = tl.load(state_idx_ptr + req_idx)
    accepted = tl.load(num_accepted_tokens_ptr + req_idx)

    if POSTPROCESS:
        new_computed = tl.load(num_computed_tokens_ptr + req_idx)
        running_tokens = new_computed - accepted + 1
        aligned = (new_computed // MAMBA_BLOCK_SIZE) * MAMBA_BLOCK_SIZE
        should_copy = (src_block_idx >= 0) & (aligned >= running_tokens)
        accept_bias = aligned - running_tokens
        dest_block_idx = aligned // MAMBA_BLOCK_SIZE - 1
    else:
        num_computed = tl.load(num_computed_tokens_ptr + req_idx)
        query_start = tl.load(query_start_loc_ptr + batch_idx)
        query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
        computed_after = num_computed + query_end - query_start
        should_copy = src_block_idx >= 0
        accept_bias = accepted - 1
        dest_block_idx = (computed_after + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1

    src_lookup_idx = src_block_idx + accept_bias if TEMPORAL else src_block_idx
    src_block_id = tl.load(
        block_table_ptr + batch_idx * block_table_stride + src_lookup_idx,
        mask=should_copy,
        other=0,
    )
    dest_block_id = tl.load(
        block_table_ptr + batch_idx * block_table_stride + dest_block_idx,
        mask=should_copy,
        other=0,
    )
    if POSTPROCESS:
        no_copy = (src_block_idx == dest_block_idx) & (accept_bias == 0)
    else:
        no_copy = src_block_idx == dest_block_idx
    should_copy = should_copy & ~no_copy
    return should_copy, src_block_id, dest_block_id, accept_bias


@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)


@triton.jit
def _get_mamba_batched_copy_indices(
    batch_idx,
    entry_idx,
    idx_mapping_ptr,
    block_table_ptrs,
    block_table_strides_ptr,
    state_idx_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
    num_accepted_tokens_ptr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    POSTPROCESS: tl.constexpr,
    TEMPORAL: tl.constexpr,
):
    req_idx = tl.load(idx_mapping_ptr + batch_idx)
    src_block_idx = tl.load(state_idx_ptr + req_idx)
    accepted = tl.load(num_accepted_tokens_ptr + req_idx)

    if POSTPROCESS:
        new_computed = tl.load(num_computed_tokens_ptr + req_idx)
        running_tokens = new_computed - accepted + 1
        aligned = (new_computed // MAMBA_BLOCK_SIZE) * MAMBA_BLOCK_SIZE
        should_copy = (src_block_idx >= 0) & (aligned >= running_tokens)
        accept_bias = aligned - running_tokens
        dest_block_idx = aligned // MAMBA_BLOCK_SIZE - 1
    else:
        num_computed = tl.load(num_computed_tokens_ptr + req_idx)
        query_start = tl.load(query_start_loc_ptr + batch_idx)
        query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
        computed_after = num_computed + query_end - query_start
        should_copy = src_block_idx >= 0
        accept_bias = accepted - 1
        dest_block_idx = (computed_after + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1

    block_table_ptr = _load_ptr(block_table_ptrs + entry_idx, tl.int32)
    block_table_stride = tl.load(block_table_strides_ptr + entry_idx)
    src_lookup_idx = src_block_idx + accept_bias if TEMPORAL else src_block_idx
    src_block_id = tl.load(
        block_table_ptr + batch_idx * block_table_stride + src_lookup_idx,
        mask=should_copy,
        other=0,
    )
    dest_block_id = tl.load(
        block_table_ptr + batch_idx * block_table_stride + dest_block_idx,
        mask=should_copy,
        other=0,
    )
    if POSTPROCESS:
        no_copy = (src_block_idx == dest_block_idx) & (accept_bias == 0)
    else:
        no_copy = src_block_idx == dest_block_idx
    should_copy = should_copy & ~no_copy
    return should_copy, src_block_id, dest_block_id, accept_bias


@triton.jit
def _copy_mamba_temporal_batched_kernel(
    state_ptrs,
    block_table_ptrs,
    block_table_strides_ptr,
    state_block_strides_ptr,
    block_bytes_ptr,
    idx_mapping_ptr,
    state_idx_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
    num_accepted_tokens_ptr,
    num_reqs: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    POSTPROCESS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    entry_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    byte_offsets = tl.program_id(2) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)

    should_copy, src_block_id, dest_block_id, _ = _get_mamba_batched_copy_indices(
        batch_idx,
        entry_idx,
        idx_mapping_ptr,
        block_table_ptrs,
        block_table_strides_ptr,
        state_idx_ptr,
        num_computed_tokens_ptr,
        query_start_loc_ptr,
        num_accepted_tokens_ptr,
        MAMBA_BLOCK_SIZE,
        POSTPROCESS,
        True,
    )
    state_ptr = _load_ptr(state_ptrs + entry_idx, tl.uint8)
    state_block_stride = tl.load(state_block_strides_ptr + entry_idx)
    block_bytes = tl.load(block_bytes_ptr + entry_idx)
    mask = (batch_idx < num_reqs) & should_copy & (byte_offsets < block_bytes)
    values = tl.load(
        state_ptr + src_block_id * state_block_stride + byte_offsets,
        mask=mask,
        other=0,
    )
    tl.store(
        state_ptr + dest_block_id * state_block_stride + byte_offsets,
        values,
        mask=mask,
    )


@triton.jit
def _copy_mamba_conv_sd_batched_kernel(
    state_ptrs,
    block_table_ptrs,
    block_table_strides_ptr,
    state_block_strides_ptr,
    state_token_strides_ptr,
    state_lens_ptr,
    idx_mapping_ptr,
    state_idx_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
    num_accepted_tokens_ptr,
    num_reqs: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    POSTPROCESS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    entry_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    byte_offsets = tl.program_id(2) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)

    should_copy, src_block_id, dest_block_id, accept_bias = (
        _get_mamba_batched_copy_indices(
            batch_idx,
            entry_idx,
            idx_mapping_ptr,
            block_table_ptrs,
            block_table_strides_ptr,
            state_idx_ptr,
            num_computed_tokens_ptr,
            query_start_loc_ptr,
            num_accepted_tokens_ptr,
            MAMBA_BLOCK_SIZE,
            POSTPROCESS,
            False,
        )
    )
    state_ptr = _load_ptr(state_ptrs + entry_idx, tl.uint8)
    state_block_stride = tl.load(state_block_strides_ptr + entry_idx)
    state_token_stride = tl.load(state_token_strides_ptr + entry_idx)
    state_len = tl.load(state_lens_ptr + entry_idx)
    copy_bytes = tl.maximum(0, state_len - accept_bias) * state_token_stride
    mask = (batch_idx < num_reqs) & should_copy & (byte_offsets < copy_bytes)
    values = tl.load(
        state_ptr
        + src_block_id * state_block_stride
        + accept_bias * state_token_stride
        + byte_offsets,
        mask=mask,
        other=0,
    )
    tl.store(
        state_ptr + dest_block_id * state_block_stride + byte_offsets,
        values,
        mask=mask,
    )


@triton.jit
def _copy_mamba_conv_state_kernel(
    state_ptr,
    idx_mapping_ptr,
    block_table_ptr,
    state_idx_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
    num_accepted_tokens_ptr,
    num_reqs: tl.constexpr,
    block_table_stride: tl.constexpr,
    state_block_stride: tl.constexpr,
    state_dim_stride: tl.constexpr,
    state_token_stride: tl.constexpr,
    DIM: tl.constexpr,
    STATE_LEN: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    POSTPROCESS: tl.constexpr,
    BLOCK_STATE_LEN: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    dim_offsets = tl.program_id(1) * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    token_offsets = tl.arange(0, BLOCK_STATE_LEN)
    should_copy, src_block_id, dest_block_id, accept_bias = _get_mamba_copy_indices(
        batch_idx,
        idx_mapping_ptr,
        block_table_ptr,
        block_table_stride,
        state_idx_ptr,
        num_computed_tokens_ptr,
        query_start_loc_ptr,
        num_accepted_tokens_ptr,
        MAMBA_BLOCK_SIZE,
        POSTPROCESS,
        False,
    )

    src_token_offsets = token_offsets + accept_bias
    token_mask = token_offsets < STATE_LEN
    dim_mask = dim_offsets < DIM
    mask = (
        (batch_idx < num_reqs)
        & should_copy
        & token_mask[:, None]
        & dim_mask[None, :]
        & (src_token_offsets < STATE_LEN)[:, None]
    )
    values = tl.load(
        state_ptr
        + src_block_id * state_block_stride
        + dim_offsets[None, :] * state_dim_stride
        + src_token_offsets[:, None] * state_token_stride,
        mask=mask,
        other=0.0,
    )
    tl.store(
        state_ptr
        + dest_block_id * state_block_stride
        + dim_offsets[None, :] * state_dim_stride
        + token_offsets[:, None] * state_token_stride,
        values,
        mask=mask,
    )


@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
    is_prefilling: torch.Tensor
    num_accepted_tokens: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        return {"is_prefilling": self.is_prefilling[:num_reqs]}

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        if not isinstance(
            attn_metadata_builder,
            (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder),
        ):
            return {}
        return {
            "num_accepted_tokens": None
            if self.num_accepted_tokens is None
            else self.num_accepted_tokens[:num_reqs],
            "num_decode_draft_tokens_cpu": None
            if self.num_decode_draft_tokens_cpu is None
            else self.num_decode_draft_tokens_cpu[:num_reqs],
        }


_MAMBA_COMMON_COPY_FIELDS: tuple[tuple[str, Any, torch.dtype], ...] = (
    ("state_ptrs", np.uint64, torch.uint64),
    ("block_table_ptrs", np.uint64, torch.uint64),
    ("block_table_strides", np.int64, torch.int64),
    ("state_block_strides", np.int64, torch.int64),
)


class MambaCacheBatchedCopyInfo:
    def __init__(
        self,
        device: torch.device,
        extra_fields: tuple[tuple[str, Any, torch.dtype], ...],
    ) -> None:
        self.device = device
        self.count = 0
        self.max_block_bytes = 0
        self._fields = _MAMBA_COMMON_COPY_FIELDS + extra_fields
        for name, _, _ in self._fields:
            setattr(self, f"{name}_gpu", None)

    def build(
        self,
        entries: list[dict[str, int]],
        max_block_bytes: int,
    ) -> None:
        self.count = len(entries)
        self.max_block_bytes = max_block_bytes

        if self.count == 0:
            for name, _, _ in self._fields:
                setattr(self, f"{name}_gpu", None)
            return

        for name, np_dtype, torch_dtype in self._fields:
            cpu = np.empty(self.count, dtype=np_dtype)
            for i, entry in enumerate(entries):
                cpu[i] = entry[name]
            gpu = torch.empty(self.count, dtype=torch_dtype, device=self.device)
            async_copy_to_gpu(cpu, out=gpu)
            setattr(self, f"{name}_gpu", gpu)

    def gpu(self, name: str) -> torch.Tensor:
        tensor = getattr(self, f"{name}_gpu")
        assert tensor is not None
        return tensor


class MambaCacheAlignInfo:
    def __init__(self, max_num_reqs: int, device: torch.device) -> None:
        self.state_idx_gpu = torch.empty(max_num_reqs, dtype=torch.int32, device=device)
        self.current_step_block_tables: tuple[torch.Tensor, ...] | None = None
        self.current_step_kv_cache_config: KVCacheConfig | None = None
        self.group_kv_cache_config: KVCacheConfig | None = None
        self.group_ids: list[int] = []
        self.spec: MambaSpec | None = None
        self.copy_info_signature: list[tuple[int, int]] | None = None
        self.conv_dim_first_info: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.temporal_copy_info = MambaCacheBatchedCopyInfo(
            device,
            (("block_bytes", np.int64, torch.int64),),
        )
        self.conv_sd_copy_info = MambaCacheBatchedCopyInfo(
            device,
            (
                ("state_token_strides", np.int64, torch.int64),
                ("state_lens", np.int64, torch.int64),
            ),
        )


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
        self.cache_config = vllm_config.cache_config
        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        if self.cache_config.mamba_cache_mode == "align":
            self.mamba_cache_align_info = MambaCacheAlignInfo(
                self.max_num_reqs, self.device
            )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        if self.cache_config.mamba_cache_mode == "align":
            state_idx = (new_req_data.num_computed_tokens - 1) // (
                self.cache_config.block_size
            )
            self.mamba_cache_align_info.state_idx_gpu[req_index] = state_idx
        self.num_accepted_tokens_gpu[req_index] = 1

    @staticmethod
    def _get_mamba_group_ids(
        kv_cache_config: KVCacheConfig,
    ) -> tuple[list[int], MambaSpec]:
        mamba_group_ids: list[int] = []
        mamba_specs: list[MambaSpec] = []
        for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            spec = kv_cache_group.kv_cache_spec
            if isinstance(spec, MambaSpec):
                mamba_group_ids.append(i)
                mamba_specs.append(spec)
        assert mamba_specs, "no mamba layers in the model"
        assert all(mamba_specs[0] == spec for spec in mamba_specs)
        return mamba_group_ids, mamba_specs[0]

    def _get_mamba_group_info(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> tuple[list[int], MambaSpec]:
        info = self.mamba_cache_align_info
        if info.group_kv_cache_config is not kv_cache_config:
            mamba_group_ids, mamba_spec = self._get_mamba_group_ids(kv_cache_config)
            info.group_kv_cache_config = kv_cache_config
            info.group_ids = mamba_group_ids
            info.spec = mamba_spec
            info.copy_info_signature = None
        assert info.spec is not None
        return info.group_ids, info.spec

    def _copy_mamba_temporal_batched(
        self,
        input_batch: InputBatch,
        num_computed_tokens: torch.Tensor,
        mamba_spec: MambaSpec,
        *,
        postprocess: bool,
    ) -> None:
        info = self.mamba_cache_align_info
        copy_info = info.temporal_copy_info
        if copy_info.count == 0:
            return

        block_bytes = 1024
        grid = (
            copy_info.count,
            input_batch.num_reqs,
            triton.cdiv(copy_info.max_block_bytes, block_bytes),
        )
        _copy_mamba_temporal_batched_kernel[grid](
            copy_info.gpu("state_ptrs"),
            copy_info.gpu("block_table_ptrs"),
            copy_info.gpu("block_table_strides"),
            copy_info.gpu("state_block_strides"),
            copy_info.gpu("block_bytes"),
            input_batch.idx_mapping,
            info.state_idx_gpu,
            num_computed_tokens,
            input_batch.query_start_loc,
            self.num_accepted_tokens_gpu,
            input_batch.num_reqs,
            mamba_spec.block_size,
            postprocess,
            BLOCK_BYTES=block_bytes,
        )

    def _copy_mamba_conv_sd_batched(
        self,
        input_batch: InputBatch,
        num_computed_tokens: torch.Tensor,
        mamba_spec: MambaSpec,
        *,
        postprocess: bool,
    ) -> None:
        info = self.mamba_cache_align_info
        copy_info = info.conv_sd_copy_info
        if copy_info.count == 0:
            return

        block_bytes = 1024
        grid = (
            copy_info.count,
            input_batch.num_reqs,
            triton.cdiv(copy_info.max_block_bytes, block_bytes),
        )
        _copy_mamba_conv_sd_batched_kernel[grid](
            copy_info.gpu("state_ptrs"),
            copy_info.gpu("block_table_ptrs"),
            copy_info.gpu("block_table_strides"),
            copy_info.gpu("state_block_strides"),
            copy_info.gpu("state_token_strides"),
            copy_info.gpu("state_lens"),
            input_batch.idx_mapping,
            info.state_idx_gpu,
            num_computed_tokens,
            input_batch.query_start_loc,
            self.num_accepted_tokens_gpu,
            input_batch.num_reqs,
            mamba_spec.block_size,
            postprocess,
            BLOCK_BYTES=block_bytes,
        )

    def _create_mamba_copy_info(
        self,
        block_tables: tuple[torch.Tensor, ...],
        mamba_group_ids: list[int],
    ) -> None:
        info = self.mamba_cache_align_info
        signature = [
            (block_tables[group_id].data_ptr(), block_tables[group_id].stride(0))
            for group_id in mamba_group_ids
        ]
        if info.copy_info_signature == signature:
            return

        # State pointers, block table pointers, and per-state strides are stable
        # across scheduler steps. Rebuild only when the backing block table
        # tensors change, such as after a kv-cache wake-up.
        kv_cache_config = info.group_kv_cache_config
        assert kv_cache_config is not None
        forward_context = self.vllm_config.compilation_config.static_forward_context
        state_copy_funcs = self.model.get_mamba_state_copy_func()
        dim_first = is_conv_state_dim_first()
        temporal_entries: list[dict[str, int]] = []
        conv_sd_entries: list[dict[str, int]] = []
        max_temporal_block_bytes = 0
        max_conv_sd_block_bytes = 0
        info.conv_dim_first_info = []

        for mamba_group_id in mamba_group_ids:
            block_table = block_tables[mamba_group_id]
            assert block_table.stride(1) == 1
            block_table_ptr = block_table.data_ptr()
            block_table_stride = block_table.stride(0)
            layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
            for layer_name in layer_names:
                kv_caches: list[torch.Tensor] = forward_context[layer_name].kv_cache
                for state, state_copy_func in zip(kv_caches, state_copy_funcs):
                    element_size = state.element_size()
                    state_block_stride_bytes = state.stride(0) * element_size
                    if state_copy_func is get_temporal_copy_spec:
                        block_bytes = state[0].numel() * element_size
                        temporal_entries.append(
                            {
                                "state_ptrs": state.data_ptr(),
                                "block_table_ptrs": block_table_ptr,
                                "block_table_strides": block_table_stride,
                                "state_block_strides": state_block_stride_bytes,
                                "block_bytes": block_bytes,
                            }
                        )
                        max_temporal_block_bytes = max(
                            max_temporal_block_bytes, block_bytes
                        )
                    else:
                        assert state_copy_func is get_conv_copy_spec
                        if dim_first:
                            info.conv_dim_first_info.append((state, block_table))
                        else:
                            state_len = state.shape[1]
                            state_token_stride_bytes = state.stride(1) * element_size
                            conv_sd_entries.append(
                                {
                                    "state_ptrs": state.data_ptr(),
                                    "block_table_ptrs": block_table_ptr,
                                    "block_table_strides": block_table_stride,
                                    "state_block_strides": state_block_stride_bytes,
                                    "state_token_strides": state_token_stride_bytes,
                                    "state_lens": state_len,
                                }
                            )
                            max_conv_sd_block_bytes = max(
                                max_conv_sd_block_bytes,
                                state_len * state_token_stride_bytes,
                            )

        info.temporal_copy_info.build(temporal_entries, max_temporal_block_bytes)
        info.conv_sd_copy_info.build(conv_sd_entries, max_conv_sd_block_bytes)
        info.copy_info_signature = signature

    def _copy_mamba_state(
        self,
        input_batch: InputBatch,
        num_computed_tokens: torch.Tensor,
        mamba_spec: MambaSpec,
        *,
        postprocess: bool,
    ) -> None:
        if input_batch.num_reqs == 0:
            return

        info = self.mamba_cache_align_info
        num_reqs = input_batch.num_reqs

        self._copy_mamba_temporal_batched(
            input_batch,
            num_computed_tokens,
            mamba_spec,
            postprocess=postprocess,
        )
        self._copy_mamba_conv_sd_batched(
            input_batch,
            num_computed_tokens,
            mamba_spec,
            postprocess=postprocess,
        )

        for state, block_table in info.conv_dim_first_info:
            dim = state.shape[1]
            state_len = state.shape[2]
            block_dim = 128
            block_state_len = 1 << (state_len - 1).bit_length()
            grid = (num_reqs, triton.cdiv(dim, block_dim))
            _copy_mamba_conv_state_kernel[grid](
                state,
                input_batch.idx_mapping,
                block_table,
                info.state_idx_gpu,
                num_computed_tokens,
                input_batch.query_start_loc,
                self.num_accepted_tokens_gpu,
                num_reqs,
                block_table.stride(0),
                state.stride(0),
                state.stride(1),
                state.stride(2),
                dim,
                state_len,
                mamba_spec.block_size,
                postprocess,
                BLOCK_STATE_LEN=block_state_len,
                BLOCK_DIM=block_dim,
            )

    def _reset_preprocess_accepted_tokens(
        self,
        input_batch: InputBatch,
        num_computed_tokens: torch.Tensor,
        mamba_spec: MambaSpec,
    ) -> None:
        info = self.mamba_cache_align_info
        num_reqs = input_batch.num_reqs
        block_size = 256
        grid = (triton.cdiv(num_reqs, block_size),)
        _reset_mamba_accepted_tokens_kernel[grid](
            input_batch.idx_mapping,
            info.state_idx_gpu,
            num_computed_tokens,
            input_batch.query_start_loc,
            self.num_accepted_tokens_gpu,
            num_reqs,
            BLOCK_SIZE=block_size,
            MAMBA_BLOCK_SIZE=mamba_spec.block_size,
        )

    def _reset_postprocess_accepted_tokens(
        self,
        input_batch: InputBatch,
        num_computed_tokens: torch.Tensor,
        mamba_spec: MambaSpec,
    ) -> None:
        info = self.mamba_cache_align_info
        num_reqs = input_batch.num_reqs
        block_size = 256
        grid = (triton.cdiv(num_reqs, block_size),)
        _reset_mamba_postprocess_accepted_tokens_kernel[grid](
            input_batch.idx_mapping,
            info.state_idx_gpu,
            num_computed_tokens,
            self.num_accepted_tokens_gpu,
            num_reqs,
            BLOCK_SIZE=block_size,
            MAMBA_BLOCK_SIZE=mamba_spec.block_size,
        )

    def _preprocess_mamba_cache_align(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        info = self.mamba_cache_align_info

        info.current_step_block_tables = block_tables
        info.current_step_kv_cache_config = kv_cache_config
        self._create_mamba_copy_info(block_tables, mamba_group_ids)
        self._copy_mamba_state(
            input_batch,
            num_computed_tokens,
            mamba_spec,
            postprocess=False,
        )
        self._reset_preprocess_accepted_tokens(
            input_batch,
            num_computed_tokens,
            mamba_spec,
        )

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        if self.cache_config.mamba_cache_mode == "align":
            self._preprocess_mamba_cache_align(
                input_batch,
                block_tables,
                kv_cache_config,
                num_computed_tokens,
            )

    def _postprocess_mamba_cache_align(
        self,
        input_batch: InputBatch,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        info = self.mamba_cache_align_info
        block_tables = info.current_step_block_tables
        kv_cache_config = info.current_step_kv_cache_config
        assert block_tables is not None
        assert kv_cache_config is not None
        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)

        self._create_mamba_copy_info(block_tables, mamba_group_ids)
        self._copy_mamba_state(
            input_batch,
            num_computed_tokens,
            mamba_spec,
            postprocess=True,
        )
        self._reset_postprocess_accepted_tokens(
            input_batch,
            num_computed_tokens,
            mamba_spec,
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
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()

        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(
            input_batch.is_prefilling_np
        )
        # During CUDAGraph capture, num_decode_draft_tokens_cpu and num_accepted_tokens
        # are created by attn_metadata_builder.build_for_cudagraph_capture, so we only
        # compute them during actual (non-capture) forward execution.
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping
            ]

            # GDN uses >= 0 to select spec-decode rows, so non-decode rows
            # need the -1 sentinel rather than a raw zero draft count.
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            if input_batch.num_draft_tokens_per_req is not None:
                spec_decode_mask = (
                    input_batch.num_draft_tokens_per_req > 0
                ) & ~input_batch.is_prefilling_np
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask,
                    input_batch.num_draft_tokens_per_req,
                    -1,
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        mamba_attn_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )
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
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
        )

    def postprocess_state(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        # Chunked prefill does not sample a token, so num_sampled can be 0.
        # Mamba treats num_accepted_tokens=1 as the neutral non-spec value.
        self.num_accepted_tokens_gpu[input_batch.idx_mapping] = torch.clamp(
            num_sampled, min=1
        )

        if self.cache_config.mamba_cache_mode == "align":
            self._postprocess_mamba_cache_align(
                input_batch,
                num_computed_tokens,
            )
