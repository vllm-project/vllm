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
