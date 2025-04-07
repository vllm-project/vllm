# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional

import torch


def get_swapper_class(uri: str,
                      use_mla: bool = False,
                      local_rank: Optional[int] = 0):
    if uri.startswith("redis"):
        from vllm.v1.core.swapper.redis import RedisSwapper
        return RedisSwapper(uri, use_mla, local_rank)
    elif uri.startswith("valkey"):
        from vllm.v1.core.swapper.valkey import ValkeySwapper
        return ValkeySwapper(uri, use_mla, local_rank)
    else:
        raise RuntimeError("only support redis swapper.")


def get_kv_cache_key(hash_id: int, rank_id: int, layer_id: int, key_type: str):
    return str(hash_id) + "/rank" + str(rank_id) + "/layer" + str(
        layer_id) + "/" + key_type


class SwapperBase(ABC):

    @abstractmethod
    def swap_in_mha(self, req_id: str, block_mapping: dict[int, int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def swap_out_mha(self, block_mapping: dict[int, int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def exist(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reg_mr(self, tensors: list[torch.Tensor]):
        raise NotImplementedError

    def dreg_mr(self, tensors: list[torch.Tensor]):
        pass

    @abstractmethod
    def get_loaded_reqs(self):
        raise NotImplementedError

    @abstractmethod
    def get_saved_blocks(self):
        raise NotImplementedError
