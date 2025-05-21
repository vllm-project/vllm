# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch

from vllm.model_executor.models.constant_size_cache import ConstantSizeCache


@dataclass
class MinimaxCacheParams:
    minimax_cache: torch.Tensor = torch.Tensor()
    state_indices_tensor: torch.Tensor = torch.Tensor()

    def at_layer_idx(self, layer_idx):
        return MinimaxCacheParams(self.minimax_cache[layer_idx, ...],
                                  self.state_indices_tensor)


class MinimaxCacheManager(ConstantSizeCache):

    def __init__(self, dtype, cache_shape):
        super().__init__(cache_shape[1])  # max_batch_size is cache_shape[1]
        self._minimax_cache = torch.empty(size=cache_shape,
                                          dtype=dtype,
                                          device="cuda")

    @property
    def cache(self):
        return self._minimax_cache

    def _copy_cache(self, from_index: int, to_index: int):
        assert len(self.cache) > 0
        for cache_t in self.cache:
            cache_t[:, to_index].copy_(cache_t[:, from_index],
                                       non_blocking=True)
