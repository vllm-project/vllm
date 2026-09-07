# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch equivalents of the ``gpu/block_table.py`` Triton kernels.

These replace ``BlockTables`` methods rather than the kernels themselves: the
kernels receive device pointer arrays so they can fuse across KV cache groups,
and a torch implementation needs the tensors those pointers stand for.
"""

import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID


def gather_block_tables(
    self,
    idx_mapping: torch.Tensor,
    num_reqs_padded: int,
    out: tuple[torch.Tensor, ...] | None = None,
    out_ptrs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    if self.num_kv_cache_groups == 0:
        return ()
    if out is None:
        out = tuple(self.input_block_tables)
    num_reqs = idx_mapping.shape[0]

    req = idx_mapping[:num_reqs].long()

    for group_id in range(self.num_kv_cache_groups):
        src = self.block_tables[group_id].gpu
        dst = out[group_id]
        num_blocks = self.num_blocks.gpu[group_id]
        # Padded rows are zeroed; live rows keep whatever follows num_blocks.
        dst[num_reqs:num_reqs_padded].zero_()
        counts = num_blocks[req].long().unsqueeze(1)
        cols = torch.arange(dst.shape[1], device=dst.device).unsqueeze(0)
        dst[:num_reqs] = torch.where(cols < counts, src[req], dst[:num_reqs])

    return tuple(block_table[:num_reqs_padded] for block_table in out)


def compute_slot_mappings(
    self,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_tokens_padded: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    slot_mappings = self.slot_mappings if out is None else out
    if self.num_kv_cache_groups == 0:
        return slot_mappings[:, :num_tokens_padded]

    num_reqs = idx_mapping.shape[0]
    max_num_tokens = slot_mappings.shape[1]
    # Stale slot ids past the real token count would otherwise be treated as
    # valid during chunked prefill.
    num_tokens = int(query_start_loc[num_reqs])

    for group_id in range(self.num_kv_cache_groups):
        row = slot_mappings[group_id]
        row[num_tokens:max_num_tokens] = PAD_SLOT_ID
        if not self._slot_mapping_enabled[group_id]:
            row[:num_tokens] = PAD_SLOT_ID
            continue

        block_table = self.block_tables[group_id].gpu
        kv_block_size = self.block_sizes[group_id]
        kernel_block_size = self.kernel_block_sizes[group_id]

        # One pass over every token, with each token's request looked up by
        # run-length expanding the batch's query lengths.
        starts = query_start_loc[:num_reqs].long()
        query_lens = query_start_loc[1 : num_reqs + 1].long() - starts
        req_per_token = torch.repeat_interleave(
            idx_mapping[:num_reqs].long(), query_lens
        )
        pos = positions[:num_tokens].long()

        is_local = None
        if self.cp_size == 1:
            local_positions = pos
        else:
            virtual_block_size = kv_block_size * self.cp_size
            virtual_block_indices = pos // virtual_block_size
            virtual_block_offsets = pos % virtual_block_size
            is_local = (
                virtual_block_offsets // self.cp_interleave % self.cp_size
            ) == self.cp_rank
            rounds = virtual_block_offsets // (self.cp_interleave * self.cp_size)
            remainder = virtual_block_offsets % self.cp_interleave
            local_positions = (
                virtual_block_indices * kv_block_size
                + rounds * self.cp_interleave
                + remainder
            )

        block_indices = local_positions // kernel_block_size
        block_offsets = local_positions % kernel_block_size
        if is_local is not None:
            block_indices = torch.where(is_local, block_indices, 0)

        block_numbers = block_table[req_per_token, block_indices].long()
        slot_ids = block_numbers * kernel_block_size + block_offsets
        if is_local is not None:
            slot_ids = torch.where(is_local, slot_ids, PAD_SLOT_ID)
        row[:num_tokens] = slot_ids.to(row.dtype)

    return slot_mappings[:, :num_tokens_padded]
