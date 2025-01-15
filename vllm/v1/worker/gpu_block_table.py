from typing import List

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils import get_cuda_view_from_cpu_tensor, is_uva_available

logger = init_logger(__name__)


class BlockTable:

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
            pin_memory=pin_memory,
        )
        self.block_table_np = self.block_table_cpu.numpy()
        self.num_blocks_per_row = np.zeros((max_num_reqs,), dtype=np.int32)

        # UVA requires pinned memory.
        self.use_uva = is_uva_available() and pin_memory
        if self.use_uva:
            logger.info("Using Unified Virtual Addressing (UVA) for block "
                        "table transfer.")
            self.block_table_diff = torch.zeros((max_num_reqs, 2),
                                                dtype=torch.int32,
                                                device="cpu",
                                                pin_memory=True)
            self.block_table_diff_np = self.block_table_diff.numpy()

            self.block_table_cpu_cuda_view = get_cuda_view_from_cpu_tensor(
                self.block_table_cpu)
            self.block_table_diff_cuda_view = get_cuda_view_from_cpu_tensor(
                self.block_table_diff)
        else:
            logger.warning("Unified Virtual Addressing (UVA) is not supported "
                           "in the current environment. This may result in "
                           "lower performance.")

    def add_row(self, row_idx: int, block_ids: List[int]) -> None:
        num_blocks = len(block_ids)
        self.block_table_np[row_idx, :num_blocks] = block_ids
        self.num_blocks_per_row[row_idx] = num_blocks
        if self.use_uva:
            self.block_table_diff_np[row_idx, 0] = 0
            self.block_table_diff_np[row_idx, 1] = num_blocks

    def append_row(
        self,
        row_idx: int,
        start: int,
        block_ids: List[int],
    ) -> None:
        num_blocks = len(block_ids)
        self.block_table_np[row_idx, start:start + num_blocks] = block_ids
        self.num_blocks_per_row[row_idx] = start + num_blocks
        if self.use_uva:
            self.block_table_diff_np[row_idx, 0] = start
            self.block_table_diff_np[row_idx, 1] = num_blocks

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        self.block_table_np[tgt, :num_blocks] = \
            self.block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks
        if self.use_uva:
            # Append-and-move is allowed.
            self.block_table_diff_np[tgt, 0] = 0
            self.block_table_diff_np[tgt, 1] = num_blocks
            # Clear the source row.
            self.block_table_diff_np[src].fill(0)

    def commit(self, num_reqs: int) -> None:
        if self.use_uva:
            # Only copy the diff to the GPU.
            ops.copy_subranges(
                self.block_table_cpu_cuda_view,
                self.block_table_diff_cuda_view,
                self.block_table,
                num_reqs,
            )
        else:
            # Copy the entire block table to the GPU.
            # NOTE(woosuk): This can be a performance bottleneck when the block
            # table is large.
            self.block_table[:num_reqs].copy_(self.block_table_cpu[:num_reqs],
                                              non_blocking=True)
        self.clear_diff()

    def clear(self) -> None:
        self.block_table.fill_(0)
        self.block_table_cpu.fill_(0)
        self.num_blocks_per_row.fill(0)
        if self.use_uva:
            self.block_table_diff.fill_(0)

    def clear_diff(self) -> None:
        if self.use_uva:
            self.block_table_diff_np.fill(0)

    def cuda(self) -> torch.Tensor:
        return self.block_table

    def cpu(self) -> torch.Tensor:
        return self.block_table_cpu

    def numpy(self) -> np.ndarray:
        return self.block_table_np
