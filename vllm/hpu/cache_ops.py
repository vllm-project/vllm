###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from typing import Tuple
import torch
import habana_frameworks.torch as htorch


def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, is_prompt=False):
    """
    key: [num_tokens, num_heads, head_size]
    value: [num_tokens, num_heads, head_size]
    key_cache: [num_heads, head_size, block_size] * num_blocks
    value_cache: [num_heads, head_size, block_size] * num_blocks
    slot_mapping: [num_tokens]
    """
    num_tokens = key.shape[0]
    block_size = key_cache.shape[-1]
    slot_mapping = slot_mapping.to(key.device)
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    if is_prompt:
        for i in range(0, num_tokens, block_size):
            key_cache.index_put_([block_indices[i]], key[i:i+block_size].transpose(0,1).transpose(1,2))
            value_cache.index_put_([block_indices[i]], value[i:i+block_size].transpose(0,1).transpose(1,2))
    else:
        key_cache = key_cache.permute(0, 3, 1, 2)
        value_cache = value_cache.permute(0, 3, 1, 2)
        block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
        block_offsets = torch.fmod(slot_mapping, block_size)
        slot_indices = torch.stack([block_indices, block_offsets], dim=-1)
        index = torch.tensor(0, device=key.device)
        for i in range(num_tokens):
            key_cache[slot_indices[i][0], slot_indices[i][1], :, :] = key[i]
            value_cache[slot_indices[i][0], slot_indices[i][1], :, :] = value[i]
            index.add_(1)
        key_cache = key_cache.permute(0, 2, 3, 1)
        value_cache = value_cache.permute(0, 2, 3, 1)


def swap_blocks(src, dst, block_mapping):
    index_src = torch.zeros((1,), dtype=torch.int32, device=key_caches[0].device)
    index_dst = torch.zeros((1,), dtype=torch.int32, device=key_caches[0].device)
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
