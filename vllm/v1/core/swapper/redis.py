# SPDX-License-Identifier: Apache-2.0

import copy
import queue
import threading
from typing import Optional
from urllib.parse import urlparse

import torch

from vllm.utils import tensor_from_bytes, tensor_to_bytes
from vllm.v1.core.swapper.base_swapper import SwapperBase, get_kv_cache_key


class RedisSwapper(SwapperBase):

    def __init__(self,
                 url: str,
                 use_mla: bool = False,
                 rank: Optional[int] = 0):
        import redis

        parsed_url = urlparse(url)
        self.swapper = redis.Redis(host=parsed_url.hostname,
                                   port=parsed_url.port)
        self.rank = rank
        self.device = f"cuda:{self.rank}"
        self.use_mla = use_mla
        if self.use_mla:
            raise RuntimeError("mla kv cache offload not support at present.")

        self.swap_in_queue: queue.Queue[tuple[str, dict[int,
                                                        int]]] = queue.Queue()
        self.swap_out_queue: queue.Queue[dict[int, int]] = queue.Queue()

        self.loaded_reqs: list[str] = []
        self.saved_blocks: list[str] = []

        self.kv_caches: Optional[list[torch.Tensor]] = None

        threading.Thread(target=self._swap_in_mha_loop).start()
        threading.Thread(target=self._swap_out_mha_loop).start()

    def reg_mr(self, tensors: list[torch.Tensor]):
        self.kv_caches = tensors

    def swap_in_mha(self, req_id: str, block_mapping: dict[int, int]) -> None:
        assert self.kv_caches is not None, "please reg_mr first."
        self.swap_in_queue.put_nowait((req_id, block_mapping))

    def swap_out_mha(self, block_mapping: dict[int, int]) -> None:
        assert self.kv_caches is not None, "please reg_mr first."
        self.swap_out_queue.put_nowait(block_mapping)

    def exist(self, key: str) -> bool:
        return self.swapper.exists(key) == 1

    def _swap_in_mha_loop(self):
        while True:
            req_id, block_mapping = self.swap_in_queue.get()
            for hash, block_id in block_mapping.items():
                for i in range(len(self.kv_caches)):
                    layer_cache = self.kv_caches[i]
                    key_cache = layer_cache[0]
                    val_cache = layer_cache[1]

                    key_cache_key = get_kv_cache_key(
                        hash, self.rank, i, "key")  # type: ignore[arg-type]
                    val_cache_key = get_kv_cache_key(
                        hash, self.rank, i, "val")  # type: ignore[arg-type]

                    key_cache_bytes = self.swapper.get(key_cache_key)
                    val_cache_bytes = self.swapper.get(val_cache_key)

                    gpu_key_cache = tensor_from_bytes(key_cache_bytes).to(
                        self.device)
                    gpu_val_cache = tensor_from_bytes(val_cache_bytes).to(
                        self.device)

                    key_cache[block_id] = gpu_key_cache
                    val_cache[block_id] = gpu_val_cache

            self.loaded_reqs.append(req_id)

    def _swap_out_mha_loop(self):
        while True:
            block_mapping = self.swap_out_queue.get()
            for block_id, hash in block_mapping.items():
                for i in range(len(self.kv_caches)):
                    layer_cache = self.kv_caches[i]
                    key_cache = layer_cache[0]
                    val_cache = layer_cache[1]

                    block_key_cache = key_cache[block_id]
                    block_val_cache = val_cache[block_id]

                    key_cache_key = get_kv_cache_key(
                        hash, self.rank, i, "key")  # type: ignore[arg-type]
                    val_cache_key = get_kv_cache_key(
                        hash, self.rank, i, "val")  # type: ignore[arg-type]

                    block_key_cache_bytes = tensor_to_bytes(block_key_cache)
                    block_val_cache_bytes = tensor_to_bytes(block_val_cache)

                    self.swapper.set(key_cache_key, block_key_cache_bytes)
                    self.swapper.set(val_cache_key, block_val_cache_bytes)

                self.swapper.set(str(hash), 1)
                self.saved_blocks.append(str(block_id))

    def get_loaded_reqs(self):
        res = copy.deepcopy(self.loaded_reqs)
        self.loaded_reqs.clear()
        return res

    def get_saved_blocks(self):
        res = copy.deepcopy(self.saved_blocks)
        self.saved_blocks.clear()
        return res
