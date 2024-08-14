###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from functools import wraps

import habana_frameworks.torch as htorch
import torch

from vllm.hpu.cache_ops import insert_or_update_cache


def with_mark_steps(fn):

    @wraps(fn)
    def wrapped(*args, **kwargs):
        htorch.core.mark_step()
        result = fn(*args, **kwargs)
        del args
        del kwargs
        htorch.core.mark_step()
        return result

    return wrapped


class Matmul(torch.nn.Module):

    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class Softmax(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dim=None, inv_head=None):
        return torch.softmax(x, dim)


class VLLMKVCache(torch.nn.Module):

    def __init__(self):
        super(VLLMKVCache, self).__init__()

    def forward(self, input, cache, num_kv_cache_passes, num_slots_available,
                block_indices, block_offset):
        insert_or_update_cache(input, cache, num_kv_cache_passes,
                               num_slots_available, block_indices,
                               block_offset)
        return cache

    def fetch_from_cache(self, cache, blocks, permutations):
        return [
            cache.index_select(0, blocks[:, i]).permute(permutations)
            for i in range(blocks.size(1))
        ]
