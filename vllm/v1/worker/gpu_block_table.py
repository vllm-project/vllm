# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch
import triton
import triton.language as tl

from vllm.utils import cdiv
from vllm.v1.worker.utils import CpuGpuBuffer

PAD_SLOT_ID = -1


class BlockTables:

    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_cached_reqs: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        device: torch.device,
        pin_memory: bool,
    ):
        self.block_sizes = block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_cached_reqs = max_num_cached_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.device = device
        self.pin_memory = pin_memory

        self.num_kv_cache_groups = len(self.block_sizes)
        # [num_kv_cache_groups, max_num_reqs, max_num_blocks]
        self.block_tables: list[torch.Tensor] = []
        # [num_kv_cache_groups, max_num_cached_reqs, max_num_blocks]
        self.block_table_buffers: list[torch.Tensor] = []
        for i in range(self.num_kv_cache_groups):
            block_size = self.block_sizes[i]
            max_num_blocks = cdiv(self.max_model_len, block_size)

            block_table = torch.zeros(
                self.max_num_reqs,
                max_num_blocks,
                dtype=torch.int32,
                device=self.device,
            )
            self.block_tables.append(block_table)

            block_table_buffer = torch.zeros(
                self.max_num_cached_reqs,
                max_num_blocks,
                dtype=torch.int32,
                device=self.device,
            )
            self.block_table_buffers.append(block_table_buffer)

        self.block_table_ptrs = self._make_ptr_tensor(self.block_tables)
        self.buffer_ptrs = self._make_ptr_tensor(self.block_table_buffers)
        self.block_table_strides = torch.tensor(
            [b.stride(0) for b in self.block_tables],
            dtype=torch.int64,
            device=self.device)
        self.block_sizes_tensor = torch.tensor(self.block_sizes,
                                               dtype=torch.int32,
                                               device=self.device)
        self.num_blocks = torch.zeros(self.num_kv_cache_groups,
                                      self.max_num_reqs,
                                      dtype=torch.int32,
                                      device=self.device)
        self.slot_mappings = torch.zeros(self.num_kv_cache_groups,
                                         self.max_num_batched_tokens,
                                         dtype=torch.int64,
                                         device=self.device)

        # Misc buffers.
        self.req_indices = self._make_buffer(self.max_num_reqs,
                                             dtype=torch.int32)
        self.overwrite = self._make_buffer(self.max_num_reqs, dtype=torch.bool)
        self.cu_num_new_blocks = self._make_buffer(self.num_kv_cache_groups,
                                                   self.max_num_reqs + 1,
                                                   dtype=torch.int32)
        # NOTE(woosuk): Here, we assume that total number of new blocks
        # is ALWAYS less than max_num_batched_tokens.
        # TODO(woosuk): Rigorously verify that this assumption is correct.
        self.new_block_ids = self._make_buffer(self.num_kv_cache_groups,
                                               self.max_num_batched_tokens,
                                               dtype=torch.int32)

    def _make_buffer(self, *args, dtype: torch.dtype) -> CpuGpuBuffer:
        return CpuGpuBuffer(*args,
                            dtype=dtype,
                            pin_memory=self.pin_memory,
                            device=self.device)

    def _make_ptr_tensor(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        ptrs_tensor_cpu = torch.tensor([t.data_ptr() for t in x],
                                       dtype=torch.int64,
                                       device="cpu",
                                       pin_memory=self.pin_memory)
        return ptrs_tensor_cpu.to(self.device, non_blocking=True)

    def append_block_ids(
        self,
        # [num_reqs]
        req_indices: list[int],
        # [num_kv_cache_groups, num_reqs + 1]
        cu_num_new_blocks: list[list[int]],
        # [num_kv_cache_groups, num_new_blocks]
        new_block_ids: list[list[int]],
        # [num_reqs]
        overwrite: list[bool],
    ) -> None:
        # TODO(woosuk): Optimize & simplify this.
        num_reqs = len(req_indices)
        self.req_indices.np[:num_reqs] = req_indices
        self.overwrite.np[:num_reqs] = overwrite
        for i in range(self.num_kv_cache_groups):
            self.cu_num_new_blocks.np[i, :num_reqs + 1] = cu_num_new_blocks[i]
            n = len(new_block_ids[i])
            self.new_block_ids.np[i, :n] = new_block_ids[i]

        _append_block_ids_kernel[(num_reqs, self.num_kv_cache_groups)](
            self.req_indices.copy_to_gpu(num_reqs),
            self.cu_num_new_blocks.copy_to_gpu(),
            self.cu_num_new_blocks.gpu.stride(0),
            self.new_block_ids.copy_to_gpu(),
            self.new_block_ids.gpu.stride(0),
            self.overwrite.copy_to_gpu(num_reqs),
            self.block_table_strides,
            self.buffer_ptrs,
            self.num_blocks,
            self.num_blocks.stride(0),
            BLOCK_SIZE=1024,
        )

    def compute_block_tables(
        self,
        idx_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch_size = idx_mapping.shape[0]
        _compute_block_tables_kernel[(batch_size, self.num_kv_cache_groups)](
            idx_mapping,
            self.buffer_ptrs,
            self.block_table_ptrs,
            self.block_table_strides,
            self.num_blocks,
            self.num_blocks.stride(0),
            BLOCK_SIZE=1024,
        )
        return tuple(b[:batch_size] for b in self.block_tables)

    def compute_slot_mappings(
        self,
        cu_num_tokens: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        num_tokens = pos.shape[0]
        num_reqs = cu_num_tokens.shape[0] - 1
        num_groups = self.num_kv_cache_groups
        _compute_slot_mappings_kernel[(num_reqs + 1, num_groups)](
            num_tokens,
            self.max_num_batched_tokens,
            cu_num_tokens,
            pos,
            self.block_table_ptrs,
            self.block_table_strides,
            self.block_sizes_tensor,
            self.slot_mappings,
            self.slot_mappings.stride(0),
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )
        return tuple(x[:num_tokens] for x in self.slot_mappings)


@triton.jit
def _append_block_ids_kernel(
    # Inputs
    req_indices,  # [num_reqs]
    cu_num_new_blocks_ptr,  # [num_kv_cache_groups, num_reqs + 1]
    cu_num_new_blocks_stride,
    new_block_ids_ptr,  # [num_kv_cache_groups, num_new_blocks]
    new_block_ids_stride,
    overwrite,  # [num_reqs]
    block_table_strides,  # [num_kv_cache_groups]
    # Outputs
    block_table_buffer_ptrs,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    # Constants
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    group_id = tl.program_id(1)
    req_idx = tl.load(req_indices + batch_idx)
    do_overwrite = tl.load(overwrite + batch_idx)

    group_new_blocks_ptr = (cu_num_new_blocks_ptr +
                            group_id * cu_num_new_blocks_stride)
    start_idx = tl.load(group_new_blocks_ptr + batch_idx)
    end_idx = tl.load(group_new_blocks_ptr + batch_idx + 1)
    num_new_blocks = end_idx - start_idx

    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    if do_overwrite:
        dst_start_idx = 0
    else:
        dst_start_idx = tl.load(group_num_blocks_ptr + req_idx)
    dst_end_idx = dst_start_idx + num_new_blocks
    tl.store(group_num_blocks_ptr + req_idx, dst_end_idx)

    # Destination
    block_table_buffer_ptr = _load_ptr(block_table_buffer_ptrs + group_id,
                                       tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    buffer_row_ptr = block_table_buffer_ptr + req_idx * block_table_stride

    group_new_block_ids_ptr = (new_block_ids_ptr +
                               group_id * new_block_ids_stride)
    for i in tl.range(0, num_new_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(group_new_block_ids_ptr + start_idx + offset,
                            mask=offset < num_new_blocks)
        tl.store(buffer_row_ptr + dst_start_idx + offset,
                 block_ids,
                 mask=offset < num_new_blocks)


@triton.jit
def _compute_block_tables_kernel(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_table_ptrs,  # [num_kv_cache_groups]
    dst_block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    # kv cache group id
    group_id = tl.program_id(1)
    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)

    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    num_blocks = tl.load(group_num_blocks_ptr + req_idx)

    stride = tl.load(block_table_strides + group_id)
    src_block_table_ptr = _load_ptr(src_block_table_ptrs + group_id, tl.int32)
    src_row_ptr = src_block_table_ptr + req_idx * stride
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride

    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)


@triton.jit
def _compute_slot_mappings_kernel(
    num_tokens,
    max_num_tokens,
    cu_num_tokens,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    page_sizes,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
    slot_mappings_stride,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    # kv cache group id
    group_id = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if req_idx == tl.num_programs(0) - 1:
        # Pad remaining slots to -1. This is needed for CUDA graphs.
        for i in tl.range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset,
                     PAD_ID,
                     mask=offset < max_num_tokens)
        return

    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    page_size = tl.load(page_sizes + group_id)

    start_idx = tl.load(cu_num_tokens + req_idx)
    end_idx = tl.load(cu_num_tokens + req_idx + 1)
    for i in tl.range(start_idx, end_idx, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)
        block_indices = positions // page_size
        block_numbers = tl.load(block_table_ptr +
                                req_idx * block_table_stride + block_indices)
        slot_ids = block_numbers * page_size + positions % page_size
        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)


@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    return tl.cast(ptr, tl.pointer_type(elem_dtype))
