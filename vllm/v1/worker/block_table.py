# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Concatenate, ParamSpec, Union

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class BlockTable:

    def __init__(
        self,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_tokens: int,  # TODO
        pin_memory: bool,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_tokens = max_num_tokens
        self.pin_memory = pin_memory
        self.device = device

        self.block_table = torch.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            device=self.device,
            dtype=torch.int32,
        )
        self.block_table_cpu = torch.zeros(
            (max_num_reqs, max_num_blocks_per_req),
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
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        if not block_ids:
            return
        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table_np[row_idx, start:start + num_blocks] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        self.block_table_np[tgt, :num_blocks] = self.block_table_np[
            src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        num_blocks_src = self.num_blocks_per_row[src]
        num_blocks_tgt = self.num_blocks_per_row[tgt]
        self.num_blocks_per_row[src] = num_blocks_tgt
        self.num_blocks_per_row[tgt] = num_blocks_src

        self.block_table_np[[src, tgt]] = self.block_table_np[[tgt, src]]

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


P = ParamSpec("P")


class MultiLayerBlockTable:
    move_row: Callable[P, None]
    commit: Callable[P, None]
    clear: Callable[P, None]

    append_row: Callable[Concatenate[list[int], P], None]
    add_row: Callable[Concatenate[list[int], P], None]

    def __init__(self, max_num_reqs: int, max_num_blocks_per_req: list[int],
                 max_num_tokens: int, pin_memory: bool, device: torch.device,
                 kv_cache_config: KVCacheConfig) -> None:
        self.block_tables = [
            BlockTable(max_num_reqs, max_num_blocks_per_req[i], max_num_tokens,
                       pin_memory, device)
            for i in range(len(kv_cache_config.kv_cache_groups))
        ]
        # For methods that just pass the arguments to each BlockTable.
        for f_name in ("move_row", "swap_row", "commit", "clear"):
            setattr(self, f_name, self._make_broadcast_func(f_name))
        # For methods that require a block_ids as the first argument.
        for f_name in ("append_row", "add_row"):
            setattr(self, f_name,
                    self._make_broadcast_func_with_block_ids(f_name))

    def _make_broadcast_func(self, f_name: str) -> Callable[P, None]:

        def broadcast_func(*args: P.args, **kwargs: P.kwargs) -> None:
            for block_table in self.block_tables:
                getattr(block_table, f_name)(*args, **kwargs)

        return broadcast_func

    def _make_broadcast_func_with_block_ids(
            self, f_name: str) -> Callable[Concatenate[list[int], P], None]:

        def broadcast_func(block_ids: list[int], *args: P.args,
                           **kwargs: P.kwargs) -> None:
            for i, block_table in enumerate(self.block_tables):
                getattr(block_table, f_name)(block_ids[i], *args, **kwargs)

        return broadcast_func

    def __getitem__(self, idx: int) -> "BlockTable":
        return self.block_tables[idx]


def initialize_block_table(
    max_num_reqs: int,
    max_model_len: int,
    max_num_tokens: int,
    pin_memory: bool,
    device: torch.device,
    kv_cache_config: KVCacheConfig,
) -> Union[BlockTable, MultiLayerBlockTable]:
    max_num_blocks_per_req = [
        cdiv(max_model_len, g.kv_cache_spec.block_size)
        for g in kv_cache_config.kv_cache_groups
    ]
    if len(kv_cache_config.kv_cache_groups) == 1:
        return BlockTable(max_num_reqs, max_num_blocks_per_req[0],
                          max_num_tokens, pin_memory, device)
    else:
        return MultiLayerBlockTable(max_num_reqs, max_num_blocks_per_req,
                                    max_num_tokens, pin_memory, device,
                                    kv_cache_config)
