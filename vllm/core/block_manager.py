"""A block manager that manages token blocks."""
from typing import Dict, List, Optional, Set
from collections import defaultdict
import enum

from vllm.block import PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device


class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        self.free_blocks: List[PhysicalTokenBlock] = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size)
            self.free_blocks.append(block)

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]


class AllocStatus(enum.Enum):
    """Result for BlockSpaceManager.can_allocate

    1. Ok: seq_group can be allocated now.
    2. Later: seq_group cannot be allocated.
      The capacity of allocator is larger than seq_group required.
    3. Never: seq_group can never be allocated.
      The seq_group is too large to allocated in GPU.
    """
    OK = enum.auto()
    LATER = enum.auto()
    NEVER = enum.auto()


class BlockSpaceManager:
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.block_sliding_window = None
        if sliding_window is not None:
            assert sliding_window % block_size == 0, (sliding_window,
                                                      block_size)
            self.block_sliding_window = sliding_window // block_size

        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(Device.GPU, block_size,
                                            num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size,
                                            num_cpu_blocks)
        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[int, BlockTable] = {}

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs()[0]
        num_required_blocks = len(seq.logical_token_blocks)
        if self.block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.block_sliding_window)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def allocate(self, seq_group: SequenceGroup) -> None:
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = seq_group.get_seqs()[0]

        # Allocate new physical token blocks that will store the prompt tokens.
        block_table: BlockTable = []
        for logical_idx in range(len(seq.logical_token_blocks)):
            if (self.block_sliding_window is not None
                    and logical_idx >= self.block_sliding_window):
                block = block_table[logical_idx % self.block_sliding_window]
            else:
                block = self.gpu_allocator.allocate()
            # Set the reference counts of the token blocks.
            block.ref_count = seq_group.num_seqs()
            block_table.append(block)

        # Assign the block table for each sequence.
        for seq in seq_group.get_seqs():
            self.block_tables[seq.seq_id] = block_table.copy()

    def can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_preallocated_slots: int = 0) -> bool:
        """Determine whether there is enough space to append new slots to the
        running sequences in the sequence group.

        Args:
            seq_group: The sequence group whose running sequences will be used
                in the determination.
            num_preallocated_slots: The number of slots beyond the sequence
                length that will be allocated. Used when a worker emits more
                than one token per scheduler invocation.
        """
        running_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        max_num_new_blocks = self._get_num_new_blocks_required_to_append(
            running_seqs, num_preallocated_slots)
        return max_num_new_blocks <= self.gpu_allocator.get_num_free_blocks()

    def _get_num_new_blocks_required_to_append(
            self, seqs: List[Sequence], num_preallocated_slots: int) -> int:
        """Calculate the number of new blocks required to append new tokens.

        Args:
            seqs: The list of sequences to be used in the calculation.
            num_preallocated_slots: The number of slots beyond the sequence
                length that will be allocated. Used when a worker emits more
                than one token per scheduler invocation.

        """
        max_num_new_slots_per_seq = [
            seq.get_num_unprocessed_token_ids() + num_preallocated_slots
            for seq in seqs
        ]

        # For simplicity, we assume each new slot consumes a new block (either
        # by COW or new allocation). This is the worst case --  a better
        # heuristic could be used.
        max_num_new_blocks = sum(max_num_new_slots_per_seq)
        return max_num_new_blocks

    def append_slots(self,
                     seq: Sequence,
                     num_preallocated_slots: int = 0) -> Dict[int, List[int]]:
        """Allocate physical slots for new tokens.

        Args:
            seq: The sequence that needs allocation to store new tokens.
            num_preallocated_slots: The number of slots beyond the sequence
                length that will be allocated. Used when a worker emits more
                than one token per scheduler invocation.
        """
        seq.ensure_num_empty_slots(num_preallocated_slots)

        num_new_blocks = 0
        while len(self.block_tables[seq.seq_id]) < len(
                seq.logical_token_blocks):
            self._append_block(self.block_tables[seq.seq_id])
            num_new_blocks += 1

        # Even if no new blocks were added, make sure the last block is
        # appendable.
        num_blocks_to_check_appendable = max(num_new_blocks, 1)
        return self._ensure_last_blocks_are_appendable(
            self.block_tables[seq.seq_id], num_blocks_to_check_appendable)

    def _append_block(self, block_table: BlockTable) -> None:
        """Append a block to the block table. May allocate a new block or re-use
        a block when configured with a sliding window.
        """
        if (self.block_sliding_window
                and len(block_table) >= self.block_sliding_window):
            # re-use a block
            block_table.append(block_table[len(block_table) %
                                           self.block_sliding_window])
            return

        # The sequence has a new logical block.
        # Allocate a new physical block.
        block = self.gpu_allocator.allocate()
        block_table.append(block)

    def _ensure_last_blocks_are_appendable(
            self, block_table: BlockTable,
            num_blocks_to_check: int) -> Dict[int, List[int]]:
        """Ensure the last blocks in the block table are appendable, e.g. if the
        blocks are owned by a single sequence.

        The blocks which are not appendable are replaced with new blocks. The
        copy-on-write source and destination block numbers are then returned.
        """
        cow_src_dst = defaultdict(list)
        for i in range(
                len(block_table) - num_blocks_to_check, len(block_table)):
            # We want to check if a token can be appended to this block.
            block = block_table[i]
            assert block.device == Device.GPU
            if block.ref_count == 1:
                # Not shared with other sequences. Appendable.
                continue

            # The block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            new_block = self.gpu_allocator.allocate()
            block_table[i] = new_block
            self.gpu_allocator.free(block)

            cow_src_dst[block.block_number].append(new_block.block_number)

        return dict(cow_src_dst)

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1

    def _get_physical_blocks(
            self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            blocks.update(self.block_tables[seq.seq_id])
        return list(blocks)

    def can_swap_in(self,
                    seq_group: SequenceGroup,
                    num_preallocated_slots: int = 0) -> bool:
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block per new slot right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        swapped_seqs = seq_group.get_seqs(status=SequenceStatus.SWAPPED)
        max_num_new_blocks = self._get_num_new_blocks_required_to_append(
            swapped_seqs, num_preallocated_slots)

        blocks = self._get_physical_blocks(seq_group)

        num_required_blocks = len(blocks) + max_num_new_blocks
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # CPU block -> GPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for cpu_block in block_table:
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self.gpu_allocator.allocate()
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                self.cpu_allocator.free(cpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # GPU block -> CPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for gpu_block in block_table:
                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_allocator.allocate()
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                self.gpu_allocator.free(gpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping

    def _free_block_table(self, block_table: BlockTable) -> None:
        for block in set(block_table):
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

        # Sequence tracks which tokens have been saved to KV.
        # Clear it as the physical block data may be overwritten.
        seq.reset_processed_tokens()

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()
