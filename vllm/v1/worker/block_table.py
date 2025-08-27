# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import numpy as np
import torch

from vllm.distributed import get_cp_group
from vllm.logger import init_logger
from vllm.utils import cdiv

logger = init_logger(__name__)


class BlockTable:

    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
    ):
        self.block_size = block_size
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_batched_tokens = max_num_batched_tokens
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

        self.slot_mapping_cpu = torch.zeros(self.max_num_batched_tokens,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=self.pin_memory)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.slot_mapping = torch.zeros(self.max_num_batched_tokens,
                                        dtype=torch.int64,
                                        device=self.device)

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

    def compute_slot_mapping(self, req_indices: np.ndarray,
                             positions: np.ndarray) -> None:
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size`
        # here because M (max_model_len) is not necessarily divisible by
        # block_size.
        block_table_indices = (req_indices * self.max_num_blocks_per_req +
                               positions // self.block_size)
        block_numbers = self.block_table_np.ravel()[block_table_indices]
        block_offsets = positions % self.block_size
        np.add(block_numbers * self.block_size,
               block_offsets,
               out=self.slot_mapping_np[:req_indices.shape[0]])

    def commit_block_table(self, num_reqs: int) -> None:
        self.block_table[:num_reqs].copy_(self.block_table_cpu[:num_reqs],
                                          non_blocking=True)

    def commit_slot_mapping(self, num_tokens: int) -> None:
        self.slot_mapping[:num_tokens].copy_(
            self.slot_mapping_cpu[:num_tokens], non_blocking=True)

    def clear(self) -> None:
        self.block_table.fill_(0)
        self.block_table_cpu.fill_(0)

    def get_device_tensor(self) -> torch.Tensor:
        """Returns the device tensor of the block table."""
        return self.block_table

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table_cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table_np


class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(self, max_num_reqs: int, max_model_len: int,
                 max_num_batched_tokens: int, pin_memory: bool,
                 device: torch.device, block_sizes: list[int]) -> None:
        self.cp_world_size = get_cp_group().world_size
        self.cp_rank = get_cp_group().rank_in_group
        # Note(hc): each cp rank only store (max_model_len//cp_world_size) tokens in kvcache,
        # so the block_size which used for calc max_num_blocks_per_req must be multiplied by cp_world_size.
        self.block_tables = [
            BlockTable(block_size, max_num_reqs,
                       cdiv(max_model_len, block_size * self.cp_world_size),
                       max_num_batched_tokens, pin_memory, device)
            for block_size in block_sizes
        ]

    def append_row(self, block_ids: tuple[list[int], ...],
                   row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
            self, req_indices: np.ndarray, positions: np.ndarray,
            num_reqs: int, num_decodes: int,
            num_computed_tokens_cpu: np.ndarray, seq_lens_np: np.ndarray,
            cp_num_computed_tokens_cpu: np.ndarray,
            cp_local_token_select_indices_np: np.ndarray) -> Optional[int]:
        # Note(hc): cp_local_token_cnt records the number of KV entries will be stored on the current CP rank.
        cp_local_token_cnt: Optional[int] = None
        if self.cp_world_size > 1:
            assert len(self.block_tables) == 1
            block_size = self.block_tables[0].block_size
            block_table_array = self.block_tables[0].block_table_np.ravel()
            max_num_blocks_per_req = self.block_tables[
                0].max_num_blocks_per_req
            slot_mapping_np = self.block_tables[0].slot_mapping_np
            cp_local_token_cnt = 0
            cu_token_idx = 0
            for req_idx in range(num_reqs):
                context_len = num_computed_tokens_cpu[req_idx]
                seq_len = seq_lens_np[req_idx]
                # calculate context lens under CP for prefill reqs
                if req_idx >= num_decodes:
                    cp_prefill_context_len = cdiv(context_len,
                                                  self.cp_world_size)
                    cp_num_computed_tokens_cpu[
                        req_idx] = cp_prefill_context_len
                for token_idx in range(context_len, seq_len):
                    target_cp_rank = token_idx % self.cp_world_size
                    cp_context_len = token_idx // self.cp_world_size
                    if self.cp_rank <= target_cp_rank:
                        cp_context_len += 1
                    # update context length for decode reqs
                    if req_idx < num_decodes:
                        seq_lens_np[req_idx] = cp_context_len
                    # update slot_mapping for both prefill & decode reqs
                    if self.cp_rank == target_cp_rank:
                        position = cp_context_len - 1
                        block_table_indice = req_idx * max_num_blocks_per_req + position // block_size
                        block_number = block_table_array[block_table_indice]
                        block_offset = position % block_size
                        slot_mapping_np[
                            cp_local_token_cnt] = block_number * block_size + block_offset
                        cp_local_token_select_indices_np[
                            cp_local_token_cnt] = cu_token_idx + token_idx - context_len
                        cp_local_token_cnt += 1
                cu_token_idx += seq_len - context_len
        else:
            for block_table in self.block_tables:
                block_table.compute_slot_mapping(req_indices, positions)
        return cp_local_token_cnt

    def commit_block_table(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    def commit_slot_mapping(self, num_tokens: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_slot_mapping(num_tokens)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]
