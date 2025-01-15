from typing import List, Set

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger

logger = init_logger(__name__)


class GPUBlockTable:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_blocks_per_req: int,
        pin_memory: bool,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_blocks_per_req = max_num_blocks_per_req
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
            pin_memory=False,
        )
        self.block_table_np = self.block_table_cpu.numpy()
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        self.block_table_diff_np = np.zeros(
            (max_num_reqs, 2),
            dtype=np.int32,
        )
        self.diff_rows: Set[int] = set()

        self.append_row_indices = torch.zeros(
            (max_num_reqs, 2),
            dtype=torch.int32,
            device=self.device,
        )
        self.append_row_indices_cpu = torch.zeros_like(
            self.append_row_indices,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.append_row_indices_np = self.append_row_indices_cpu.numpy()
        self.append_cumsums = torch.zeros(
            (max_num_reqs + 1,),
            dtype=torch.int32,
            device=self.device,
        )
        self.append_cumsums_cpu = torch.zeros_like(
            self.append_cumsums,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.append_cumsums_np = self.append_cumsums_cpu.numpy()
        self.append_data = torch.zeros(
            (max_num_reqs * max_num_blocks_per_req,),
            dtype=torch.int32,
            device=self.device,
        )
        self.append_data_cpu = torch.zeros_like(
            self.append_data,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.append_data_np = self.append_data_cpu.numpy()

    def append_row(
        self,
        row_idx: int,
        start: int,
        block_ids: List[int],
    ) -> None:
        num_blocks = len(block_ids)
        self.block_table_np[row_idx, start:start + num_blocks] = block_ids
        self.num_blocks_per_row[row_idx] = start + num_blocks

        self.block_table_diff_np[row_idx, 0] = start
        self.block_table_diff_np[row_idx, 1] = num_blocks
        self.diff_rows.add(row_idx)

    def add_row(self, row_idx: int, block_ids: List[int]) -> None:
        self.append_row(row_idx, 0, block_ids)

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        self.block_table_np[tgt, :num_blocks] = self.block_table_np[
            src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

        self.block_table_diff_np[tgt, 0] = 0
        self.block_table_diff_np[tgt, 1] = num_blocks
        self.diff_rows.discard(src)
        self.diff_rows.add(tgt)

    def commit(self, num_reqs: int) -> None:
        if not self.diff_rows:
            return

        cu_end = 0
        self.append_cumsums_np[0] = 0
        for i, row_idx in enumerate(self.diff_rows):
            start, num_blocks = self.block_table_diff_np[row_idx]
            assert num_blocks > 0

            self.append_row_indices_np[i, 0] = row_idx
            self.append_row_indices_np[i, 1] = start
            cu_start = self.append_cumsums_np[i]
            cu_end = cu_start + num_blocks
            self.append_cumsums_np[i + 1] = cu_end
            self.append_data_np[cu_start:cu_end] = self.block_table_np[
                row_idx, start:start + num_blocks]

        ops.block_table_appends(
            self.append_row_indices,
            self.append_row_indices_cpu,
            self.append_cumsums,
            self.append_cumsums_cpu,
            self.append_data,
            self.append_data_cpu,
            self.block_table,
            len(self.diff_rows),
            cu_end,
        )
        self.diff_rows.clear()

    def clear(self) -> None:
        self.block_table.fill_(0)
        self.block_table_cpu.fill_(0)

        self.diff_rows.clear()
        self.block_table_diff_np.fill(0)

        self.append_row_indices.fill_(0)
        self.append_row_indices_cpu.fill_(0)
        self.append_cumsums.fill_(0)
        self.append_cumsums_cpu.fill_(0)
        self.append_data.fill_(0)
        self.append_data_cpu.fill_(0)

    def get_device_tensor(self) -> torch.Tensor:
        """Ruturns the device tensor of the block table."""
        return self.block_table

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table_cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table_np
