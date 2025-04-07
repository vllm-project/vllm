# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from urllib.parse import urlparse

from vllm.v1.core.swapper.base_swapper import SwapperBase
import vllm._custom_ops as ops 

import torch

class ValkeySwapper(SwapperBase):

    def __init__(self,
                 url: str,
                 use_mla: bool = False,
                 rank: Optional[int] = 0):
        parsed_url = urlparse(url)
        self.swapper = ops.valkey_connect(parsed_url.hostname,
                                   parsed_url.port, rank=rank)
        self.rank = rank
        self.device = f"cuda:{self.rank}"
        self.use_mla = use_mla
        if self.use_mla:
            raise RuntimeError("mla kv cache offload not support at present.")

    def reg_mr(self, tensors: list[torch.Tensor]):
        ops.valkey_reg_mr(self.swapper, tensors)

    def swap_in_mha(self, req_id: str, block_mapping: dict[int, int]) -> None:
        hashs = []
        blocks = []
        for hash, block in block_mapping.items():
            hashs.append(hash)
            blocks.append(block)

        ops.valkey_swap_in(self.swapper, req_id, hashs, blocks)

    def swap_out_mha(self, block_mapping: dict[int, int]) -> None:
        hashs = []
        blocks = []
        for block, hash in block_mapping.items():
            hashs.append(hash)
            blocks.append(block)

        ops.valkey_swap_out(self.swapper, blocks, hashs)

    def exist(self, key: str) -> bool:
        return ops.valkey_exist(key)

    def get_loaded_reqs(self):
        return ops.get_loaded_reqs()

    def get_saved_blocks(self):
        return ops.get_saved_blocks()