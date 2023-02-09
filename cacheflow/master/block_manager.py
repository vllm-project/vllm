from typing import Dict, Iterable, List, Optional, Set, Tuple

from cacheflow.block import PhysicalTokenBlock
from cacheflow.sequence import Sequence
from cacheflow.sequence import SequenceGroup
from cacheflow.sequence import SequenceStatus
from cacheflow.utils import Device


class BlockAllocator:

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        assert block_size in [8, 16, 32]
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        # TODO(woosuk): Make this a priority queue.
        self.free_blocks = [
            PhysicalTokenBlock(device=device, block_number=i, block_size=block_size)
            for i in range(num_blocks)
        ]

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError('Out of memory! '
                             f'No more free blocks are available.')
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError('Double free! '
                             f'The block {block} is already freed.')
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]


class BlockSpaceManager:

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.gpu_allocator = BlockAllocator(Device.GPU, block_size, num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size, num_cpu_blocks)

        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[int, BlockTable] = {}

    def can_allocate(self, seq_group: SequenceGroup) -> bool:
        seq = seq_group.seqs[0]
        num_required_blocks = len(seq.logical_token_blocks)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        return num_required_blocks <= num_free_gpu_blocks

    def allocate(self, seq_group: SequenceGroup) -> None:
        # Here, we assume that all sequences in the group have the same prompt.
        seq = seq_group.seqs[0]

        # Allocate new physical token blocks that will store the prompt tokens.
        block_table: BlockTable = []
        for _ in range(len(seq.logical_token_blocks)):
            block = self.gpu_allocator.allocate()
            # Set the reference counts of the token blocks.
            block.ref_count = seq_group.num_seqs()
            block_table.append(block)

        # Assign the block table for each sequence.
        for seq in seq_group.seqs:
            self.block_tables[seq.seq_id] = block_table.copy()

    def can_append(self, seq_group: SequenceGroup) -> bool:
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.SERVING)
        return num_seqs <= num_free_gpu_blocks

    def append(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for the new token."""
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]

        if len(block_table) < len(logical_blocks):
            # The sequence has a new logical block.
            # Allocate a new physical block.
            block = self.gpu_allocator.allocate()
            block_table.append(block)
            return None

        # We want to append the token to the last physical block.
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # Append.
            return None
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            new_block = self.gpu_allocator.allocate()
            block_table.append(new_block)
            self.gpu_allocator.free(last_block)
            return last_block.block_number, new_block.block_number

    def fork(self, src_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[src_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1

    def _get_physical_blocks(self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.seqs:
            if seq.status == SequenceStatus.FINISHED:
                continue
            block_table = self.block_tables[seq.seq_id]
            for block in block_table:
                blocks.add(block)
        return list(blocks)

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.gpu_allocator.get_num_free_blocks()

    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # src_block_number -> dst_block_number
        mapping: Dict[int, int] = {}
        for seq in seq_group.seqs:
            if seq.status == SequenceStatus.FINISHED:
                continue
            block_table = self.block_tables[seq.seq_id]

            for cpu_block in block_table:
                if cpu_block in mapping:
                    continue
                gpu_block = self.gpu_allocator.allocate()
                mapping[cpu_block.block_number] = gpu_block.block_number
                # Free the CPU block swapped in to GPU.
                self.cpu_allocator.free(cpu_block)
        return mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # src_block_number -> dst_block_number
        mapping: Dict[int, int] = {}
        for seq in seq_group.seqs:
            if seq.status == SequenceStatus.FINISHED:
                continue
            block_table = self.block_tables[seq.seq_id]

            for gpu_block in block_table:
                if gpu_block.block_number in mapping:
                    continue
                cpu_block = self.cpu_allocator.allocate()
                mapping[gpu_block.block_number] = cpu_block.block_number
                # Free the GPU block swapped out to CPU.
                self.gpu_allocator.free(gpu_block)
        return mapping

    def _free_blocks(self, blocks: Iterable[PhysicalTokenBlock]) -> None:
        for block in blocks:
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)

    def free(self, seq: Sequence) -> None:
        block_table = self.block_tables[seq.seq_id]
        self._free_blocks(block_table)
        del self.block_tables[seq.seq_id]

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_blocks(block_table)
        self.block_tables.clear()
