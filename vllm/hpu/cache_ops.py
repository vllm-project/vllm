###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from typing import Tuple
import torch
import habana_frameworks.torch as htorch


def pad_to_full_block(data, block_size, pad_value):
    seq_dim = 1
    pad_shape = list(data.shape)
    remainder = pad_shape[seq_dim] % block_size
    if remainder == 0:
        return data
    pad_shape[seq_dim] = block_size - remainder
    pad = torch.full(pad_shape, pad_value, dtype=data.dtype, device=data.device)
    return torch.cat([data, pad], dim=seq_dim)


def initialize_cache(data, indices, cache):
    block_size = cache.size(-1)
    data = data.unflatten(0, (-1, block_size)).permute(0, 2, 3, 1)
    indices = indices.unflatten(0, (-1, block_size))[:,0]
    cache.index_copy_(0, indices, data)


def update_cache(data, indices, offsets, cache):
    prev = cache.index_select(0, indices)
    idx = offsets.view(-1, 1, 1, 1).expand(-1, data.size(1), data.size(2), -1)
    prev.scatter_(-1, idx, data.unsqueeze(-1))
    cache.index_copy_(0, indices, prev)


def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, dtype, is_prompt):
    block_size = key_cache.size(-1)
    assert slot_mapping.dim() == 2, 'This implementation requires unflattened slot_mapping!'

    if is_prompt:
        block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
        batch_size, seq_length = block_indices.shape
        key = pad_to_full_block(key.unflatten(0, (batch_size, seq_length)), block_size, 0).flatten(0, 1)
        value = pad_to_full_block(value.unflatten(0, (batch_size, seq_length)), block_size, 0).flatten(0, 1)
        block_indices = pad_to_full_block(block_indices, block_size, -1).flatten(0, 1)
        initialize_cache(key, block_indices, key_cache)
        initialize_cache(value, block_indices, value_cache)
    else:
        slot_mapping = slot_mapping.flatten()
        block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
        block_offsets = torch.fmod(slot_mapping, block_size)
        update_cache(key, block_indices, block_offsets, key_cache)
        update_cache(value, block_indices, block_offsets, value_cache)


def swap_blocks(src, dst, block_mapping):
    index_src = torch.zeros((1,), dtype=torch.int32, device=src.device)
    index_dst = torch.zeros((1,), dtype=torch.int32, device=dst.device)
    for src_idx, dst_idx in block_mapping.items():
        index_src[0] = src_idx
        index_dst[0] = dst_idx
        dst.index_put_([index_dst], src.index_select(0, index_src))
    if dst.device.type == 'hpu':
        htorch.core.mark_step()
        torch.hpu.synchronize()


def copy_blocks(key_caches, value_caches, block_mapping):
    index_src = torch.zeros((1,), dtype=torch.int32, device=key_caches[0].device)
    index_dst = torch.zeros((1,), dtype=torch.int32, device=key_caches[0].device)
    for src, dsts in block_mapping.items():
        index_src[0] = src
        for dst in dsts:
            index_dst[0] = dst
            for key_cache in key_caches:
                key_cache.index_copy_(0, index_dst, key_cache.index_select(0, index_src))
            for value_cache in value_caches:
                value_cache.index_copy_(0, index_dst, value_cache.index_select(0, index_src))
        if key_caches[0].device.type == 'hpu':
            htorch.core.mark_step()
