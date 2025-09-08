# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.distributed import get_dcp_group
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
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device
        import os
        physical_block_size = block_size
        self.block_size =  int(os.environ.get('VLLM_KERNEL_BLOCK_SIZE', block_size))
        # Hybrid block table support
        if physical_block_size != block_size:
            self.physical_block_size = physical_block_size
            self.logical_block_size = block_size
            self.blocks_per_phys_block = physical_block_size // block_size
            if physical_block_size % block_size != 0:
                raise ValueError(
                    f"Physical block size {physical_block_size} must be divisible "
                    f"by logical block size {block_size}")
            self.use_hybrid_blocks = True
            logger.info(f"Hybrid block table: physical={physical_block_size}, "
                       f"logical={block_size}, split_ratio={self.blocks_per_phys_block}")
        else:
            self.physical_block_size = block_size
            self.logical_block_size = block_size
            self.blocks_per_phys_block = 1
            self.use_hybrid_blocks = False

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
        
        # Physical block table (only used in hybrid mode)
        if self.use_hybrid_blocks:
            self.physical_block_table = torch.zeros(
                (max_num_reqs, max_num_blocks_per_req),
                device=self.device,
                dtype=torch.int32,
            )
            self.physical_block_table_cpu = torch.zeros(
                (max_num_reqs, max_num_blocks_per_req),
                device="cpu",
                dtype=torch.int32,
                pin_memory=pin_memory,
            )
            self.physical_block_table_np = self.physical_block_table_cpu.numpy()
        else:
            # In non-hybrid mode, physical table points to logical table
            self.physical_block_table = self.block_table
            self.physical_block_table_cpu = self.block_table_cpu
            self.physical_block_table_np = self.block_table_np

        self.slot_mapping_cpu = torch.zeros(self.max_num_batched_tokens,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=self.pin_memory)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.slot_mapping = torch.zeros(self.max_num_batched_tokens,
                                        dtype=torch.int64,
                                        device=self.device)
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

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
        
        # In hybrid mode, store physical blocks and convert to logical
        if self.use_hybrid_blocks:
            # Store physical blocks in physical table
            self.physical_block_table_np[row_idx, start:start + num_blocks] = block_ids
            # Convert to logical blocks and store in logical table
            logical_blocks = self._convert_physical_to_logical_blocks(np.array(block_ids))
            # Store all logical blocks, but respect array bounds
            logical_num = min(len(logical_blocks), self.max_num_blocks_per_req - start)
            if logical_num > 0:
                self.block_table_np[row_idx, start:start + logical_num] = logical_blocks[:logical_num]
        else:
            # Normal mode - just store the blocks directly in both tables
            self.block_table_np[row_idx, start:start + num_blocks] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        self.block_table_np[tgt, :num_blocks] = self.block_table_np[
            src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks
        
        # Also move physical blocks if in hybrid mode
        if self.use_hybrid_blocks:
            self.physical_block_table_np[tgt, :num_blocks] = self.physical_block_table_np[
                src, :num_blocks]

    def swap_row(self, src: int, tgt: int) -> None:
        num_blocks_src = self.num_blocks_per_row[src]
        num_blocks_tgt = self.num_blocks_per_row[tgt]
        self.num_blocks_per_row[src] = num_blocks_tgt
        self.num_blocks_per_row[tgt] = num_blocks_src

        self.block_table_np[[src, tgt]] = self.block_table_np[[tgt, src]]
        
        # Also swap physical blocks if in hybrid mode
        if self.use_hybrid_blocks:
            self.physical_block_table_np[[src, tgt]] = self.physical_block_table_np[[tgt, src]]

    def compute_slot_mapping(self, req_indices: np.ndarray,
                             positions: np.ndarray) -> None:
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size`
        # here because M (max_model_len) is not necessarily divisible by
        # block_size.
        if self.dcp_world_size > 1:
            # Note(hc): The DCP implement store kvcache with an interleave
            # style, the kvcache for the token whose token_idx is i is
            # always stored on the GPU whose dcp_rank equals i % cp_world_size:

            # Use a "virtual block" which equals to world_size * block_size
            # for block_table_indices calculation.
            virtual_block_size = self.block_size * self.dcp_world_size
            block_table_indices = (req_indices * self.max_num_blocks_per_req +
                                   positions // virtual_block_size)
            block_numbers = self.block_table_np.ravel()[block_table_indices]
            # Use virtual_block_size for mask calculation, which marks local
            # tokens.
            virtual_block_offsets = positions % virtual_block_size
            mask = virtual_block_offsets % self.dcp_world_size == self.dcp_rank
            # Calcuate local block_offsets
            block_offsets = virtual_block_offsets // self.dcp_world_size
            # Calcuate slot_mapping
            slot_mapping = block_numbers * self.block_size + block_offsets
            # Write final slots, use -1 for not-local
            self.slot_mapping_np[:req_indices.shape[0]] = np.where(
                mask, slot_mapping, -1)
        else:
            block_table_indices = (req_indices * self.max_num_blocks_per_req +
                                   positions // self.block_size)
            block_numbers = self.block_table_np.ravel()[block_table_indices]
            block_offsets = positions % self.block_size
            np.add(block_numbers * self.block_size,
                   block_offsets,
                   out=self.slot_mapping_np[:req_indices.shape[0]])

    def commit_block_table(self, num_reqs: int) -> None:
        # If in hybrid mode, synchronize logical table with physical table first
        if self.use_hybrid_blocks:
            self.sync_physical_and_logical_tables(num_reqs)
        
        self.block_table[:num_reqs].copy_(self.block_table_cpu[:num_reqs],
                                          non_blocking=True)
        
        # Also commit physical blocks if in hybrid mode
        if self.use_hybrid_blocks:
            self.physical_block_table[:num_reqs].copy_(self.physical_block_table_cpu[:num_reqs],
                                                       non_blocking=True)

    def commit_slot_mapping(self, num_tokens: int) -> None:
        self.slot_mapping[:num_tokens].copy_(
            self.slot_mapping_cpu[:num_tokens], non_blocking=True)

    def clear(self) -> None:
        self.block_table.fill_(0)
        self.block_table_cpu.fill_(0)
        
        # Also clear physical blocks if in hybrid mode
        if self.use_hybrid_blocks:
            self.physical_block_table.fill_(0)
            self.physical_block_table_cpu.fill_(0)

    def _convert_physical_to_logical_blocks(self, physical_blocks: np.ndarray) -> np.ndarray:
        """Convert physical block IDs to logical block IDs."""
        if not self.use_hybrid_blocks or self.blocks_per_phys_block == 1:
            return physical_blocks
        
        # Create logical block IDs by splitting each physical block
        logical_blocks = []
        for phys_block in physical_blocks:
            if phys_block == 0:  # Handle empty blocks (block 0 is always empty)
                logical_blocks.append(0)
            else:
                # Convert physical block to multiple logical blocks
                # Physical block 1 becomes logical blocks [1*split_ratio, 1*split_ratio+1, ...]
                # But we need to account for the fact that block 0 is special
                base_logical = (phys_block - 1) * self.blocks_per_phys_block + 1
                logical_blocks.extend(range(base_logical, base_logical + self.blocks_per_phys_block))
        
        return np.array(logical_blocks, dtype=np.int32)
    
    def _convert_logical_to_physical_blocks(self, logical_blocks: np.ndarray) -> np.ndarray:
        """Convert logical block IDs back to physical block IDs."""
        if not self.use_hybrid_blocks or self.blocks_per_phys_block == 1:
            return logical_blocks
        
        # Convert logical blocks back to physical blocks (reverse mapping)
        physical_blocks = []
        seen_phys = set()
        
        for logic_block in logical_blocks:
            if logic_block == 0:  # Handle empty blocks
                phys_block = 0
            else:
                # Convert logical block back to physical block
                # Logical block 1 becomes physical block 1
                # Logical blocks [1, 2, ..., split_ratio] become physical block 1
                phys_block = (logic_block - 1) // self.blocks_per_phys_block + 1
            
            # Only add unique physical blocks
            if phys_block not in seen_phys:
                physical_blocks.append(phys_block)
                seen_phys.add(phys_block)
        
        return np.array(physical_blocks, dtype=np.int32)
    
    def sync_physical_and_logical_tables(self, num_reqs: int) -> None:
        """Synchronize logical block table with physical block table."""
        if not self.use_hybrid_blocks:
            return
        
        # Convert physical blocks to logical blocks for each request
        for req_idx in range(num_reqs):
            num_blocks = self.num_blocks_per_row[req_idx]
            if num_blocks > 0:
                physical_blocks = self.physical_block_table_np[req_idx, :num_blocks]
                logical_blocks = self._convert_physical_to_logical_blocks(physical_blocks)
                
                # Update logical block table with converted blocks
                logical_num = min(len(logical_blocks), self.max_num_blocks_per_req)
                self.block_table_np[req_idx, :logical_num] = logical_blocks[:logical_num]
                # Note: num_blocks_per_row stays the same as physical blocks count
    
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
        # Note(hc): each dcp rank only store
        # (max_model_len//dcp_world_size) tokens in kvcache,
        # so the block_size which used for calc max_num_blocks_per_req
        # must be multiplied by dcp_world_size.
        try:
            dcp_world_size = get_dcp_group().world_size
        except AssertionError:
            # DCP might not be initialized in testing
            dcp_world_size = 1

        self.block_tables = [
            BlockTable(block_size, max_num_reqs,
                       cdiv(max_model_len, block_size * dcp_world_size),
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

    def compute_slot_mapping(self, req_indices: np.ndarray,
                             positions: np.ndarray) -> None:
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(req_indices, positions)

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
