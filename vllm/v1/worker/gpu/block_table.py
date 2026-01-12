# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


class BlockTables:
    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        device: torch.device,
    ):
        self.block_sizes = block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.device = device

        self.num_kv_cache_groups = len(self.block_sizes)
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.block_tables: list[StagedWriteTensor] = []
        for i in range(self.num_kv_cache_groups):
            block_size = self.block_sizes[i]
            max_num_blocks = cdiv(self.max_model_len, block_size)
            block_table = StagedWriteTensor(
                (self.max_num_reqs, max_num_blocks),
                dtype=torch.int32,
                device=device,
            )
            self.block_tables.append(block_table)
        self.block_table_ptrs = self._make_ptr_tensor(
            [b.gpu for b in self.block_tables]
        )
        self.block_table_strides = torch.tensor(
            [b.gpu.stride(0) for b in self.block_tables],
            dtype=torch.int64,
            device=self.device,
        )

        self.block_sizes_tensor = torch.tensor(
            self.block_sizes, dtype=torch.int32, device=self.device
        )
        self.num_blocks = UvaBackedTensor(
            (self.num_kv_cache_groups, self.max_num_reqs),
            dtype=torch.int32,
        )

        # Block tables used for model's forward pass.
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.input_block_tables: list[torch.Tensor] = [
            torch.zeros_like(b.gpu) for b in self.block_tables
        ]
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)

        self.slot_mappings = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

    def _make_ptr_tensor(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        # NOTE(woosuk): Use uint64 instead of int64 to cover all possible addresses.
        return torch.tensor(
            [t.data_ptr() for t in x],
            dtype=torch.uint64,
            device=self.device,
        )

    def append_block_ids(
        self,
        req_index: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        for i in range(self.num_kv_cache_groups):
            start = self.num_blocks.np[i, req_index] if not overwrite else 0
            block_ids = new_block_ids[i]
            self.block_tables[i].stage_write(req_index, start, block_ids)
            self.num_blocks.np[i, req_index] = start + len(block_ids)

    def apply_staged_writes(self) -> None:
        # TODO(woosuk): This can be inefficient since it launches one kernel per
        # block table. Implement a kernel to handle all block tables at once.
        for block_table in self.block_tables:
            block_table.apply_write()
        self.num_blocks.copy_to_uva()

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        num_reqs = idx_mapping.shape[0]
        _gather_block_tables_kernel[(self.num_kv_cache_groups, num_reqs)](
            idx_mapping,
            self.block_table_ptrs,
            self.input_block_table_ptrs,
            self.block_table_strides,
            self.num_blocks.gpu,
            self.num_blocks.gpu.stride(0),
            BLOCK_SIZE=1024,  # type: ignore
        )
        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def compute_slot_mappings(
        self,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = query_start_loc.shape[0] - 1
        num_tokens = positions.shape[0]
        num_groups = self.num_kv_cache_groups
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            num_tokens,
            self.max_num_batched_tokens,
            query_start_loc,
            positions,
            self.input_block_table_ptrs,
            self.block_table_strides,
            self.block_sizes_tensor,
            self.slot_mappings,
            self.slot_mappings.stride(0),
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,  # type: ignore
        )
        return self.slot_mappings[:, :num_tokens]

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        self.slot_mappings.fill_(PAD_SLOT_ID)
        return self.slot_mappings[:, :num_tokens]


@triton.jit
def _gather_block_tables_kernel(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_table_ptrs,  # [num_kv_cache_groups]
    dst_block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
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
    # kv cache group id
    group_id = tl.program_id(0)
    req_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if req_idx == tl.num_programs(1) - 1:
        # Pad remaining slots to -1. This is needed for CUDA graphs.
        for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    page_size = tl.load(page_sizes + group_id)

    start_idx = tl.load(cu_num_tokens + req_idx)
    end_idx = tl.load(cu_num_tokens + req_idx + 1)
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)
        block_indices = positions // page_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices
        )
        slot_ids = block_numbers * page_size + positions % page_size
        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)


@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)
