from typing import List

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class BlockTable:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_blocks_per_req: int,
        pin_memory: bool,
        device: torch.device,
        num_kv_cache_groups: int,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.pin_memory = pin_memory
        self.device = device
        self.num_kv_cache_groups = num_kv_cache_groups

        self.block_table = torch.zeros(
            (num_kv_cache_groups, max_num_reqs, max_num_blocks_per_req),
            device=self.device,
            dtype=torch.int32,
        )
        self.block_table_cpu = torch.zeros(
            (num_kv_cache_groups, max_num_reqs, max_num_blocks_per_req),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.block_table_np = self.block_table_cpu.numpy()
        self.num_blocks_per_row = np.zeros((num_kv_cache_groups, max_num_reqs),
                                           dtype=np.int32)

    def append_row(
        self,
        row_idx: int,
        block_ids: List[List[int]],
    ) -> None:
        for i, (num_blocks, block_ids_of_group) in enumerate(
                zip(self.num_blocks_per_row[:, row_idx], block_ids)):
            num_new_blocks = len(block_ids_of_group)
            self.block_table_np[i, row_idx, num_blocks:num_blocks +
                                num_new_blocks] = block_ids_of_group
            self.num_blocks_per_row[i, row_idx] = num_blocks + num_new_blocks

    def add_row(self, row_idx: int, block_ids: List[List[int]]) -> None:
        self.num_blocks_per_row[:, row_idx] = 0
        self.append_row(row_idx, block_ids)

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[:, src]
        self.block_table_np[:, tgt, :max(num_blocks)] = \
            self.block_table_np[:, src, :max(num_blocks)]
        self.num_blocks_per_row[:, tgt] = num_blocks

    def commit(self, num_reqs: int) -> None:
        # NOTE: an alternative is
        # self.block_table[:, :num_reqs].copy_(
        #   self.block_table_cpu[:, :num_reqs], non_blocking=True)
        # but it will be a blocking copy when num_kv_cache_groups>1.
        for i in range(self.num_kv_cache_groups):
            self.block_table[i, :num_reqs].copy_(
                self.block_table_cpu[i, :num_reqs], non_blocking=True)

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
