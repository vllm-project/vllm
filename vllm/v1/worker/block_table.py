# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List

import numpy as np
import torch
from triton import cdiv

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.kv_cache_interface import BlockIDList, FullAttentionSpec, KVCacheConfig, KVCacheSpec

logger = init_logger(__name__)


class BlockTable:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kv_cache_spec: KVCacheSpec,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_tokens = max_num_tokens
        self.max_num_blocks_per_req = cdiv(max_model_len,
                                           kv_cache_spec.block_size)
        self.pin_memory = pin_memory
        self.device = device

        self.block_table = torch.zeros(
            (max_num_reqs, self.max_num_blocks_per_req),
            device=self.device,
            dtype=torch.int32,
        )
        self.block_table_cpu = torch.zeros(
            (max_num_reqs, self.max_num_blocks_per_req),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.block_table_np = self.block_table_cpu.numpy()
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        self.slot_mapping_cpu = torch.zeros(self.max_num_tokens,
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=self.pin_memory)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()

    def append_row(
        self,
        block_ids: List[int],
        row_idx: int,
    ) -> None:
        if not block_ids:
            return
        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.block_table_np[row_idx, start:start + num_blocks] = block_ids
        self.num_blocks_per_row[row_idx] = start + num_blocks

    def add_row(self, block_ids: List[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        self.block_table_np[tgt, :num_blocks] = self.block_table_np[
            src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    def commit(self, num_reqs: int) -> None:
        self.block_table[:num_reqs].copy_(self.block_table_cpu[:num_reqs],
                                          non_blocking=True)

    def clear(self) -> None:
        self.block_table.fill_(0)
        self.block_table_cpu.fill_(0)

    def get_device_tensor(self) -> torch.Tensor:
        """Ruturns the device tensor of the block table."""
        return self.block_table

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table_cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table_np


class GroupedBlockTable:

    def __init__(self, max_num_reqs: int, max_model_len: int,
                 max_num_tokens: int, pin_memory: bool, device: torch.device,
                 kv_cache_config: KVCacheConfig):
        self.block_tables = [
            BlockTable(max_num_reqs, max_model_len, max_num_tokens, pin_memory,
                       device, g.kv_cache_spec) for g in kv_cache_config.groups
        ]
        for f_name in ('move_row', 'commit', 'clear'):
            setattr(self, f_name, self._make_grouped_func(f_name))

        for f_name in ('append_row', 'add_row'):
            # NOTE: requires to pass block_ids as the first argument
            setattr(self, f_name,
                    self._make_grouped_func_with_block_ids(f_name))

    def _make_grouped_func(self, f_name):

        def grouped_func(*args, **kwargs):
            for block_table in self.block_tables:
                getattr(block_table, f_name)(*args, **kwargs)

        return grouped_func

    def _make_grouped_func_with_block_ids(self, f_name):

        def grouped_func(block_ids: BlockIDList, *args, **kwargs):
            for i, block_table in enumerate(self.block_tables):
                getattr(block_table, f_name)(block_ids.get_group(i), *args,
                                             **kwargs)

        return grouped_func

    def __getitem__(self, idx) -> BlockTable:
        return self.block_tables[idx]
