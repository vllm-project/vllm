import collections
import copy
import math
from typing import Dict, List, Optional, Set, Tuple

from cacheflow.block import PhysicalTokenBlock
from cacheflow.sequence import Sequence
from cacheflow.sequence import SequenceGroup
from cacheflow.sequence import SequenceStatus
from cacheflow.utils import Device

_MAX_SEQ_LEN = 2048


class BuddyAllocator:

    def __init__(
        self,
        device: Device,
        token_block_size: int,
        num_token_blocks: int,
    ) -> None:
        self.device = device
        self.token_block_size = token_block_size
        self.num_token_blocks = num_token_blocks

        self.min_block_size = 1
        self.max_block_size = _MAX_SEQ_LEN // token_block_size
        self.size_to_free_blocks: Dict[int, List[int]] = collections.defaultdict(list)
        self.addr_to_size: Dict[int, int] = {}

        buddy_size = self.max_block_size
        last_start_addr = 0
        start_addrs = []
        while buddy_size >= 1:
            new_start_addrs = []
            while last_start_addr + buddy_size <= self.num_token_blocks:
                new_start_addrs.append(last_start_addr)
                last_start_addr += buddy_size

            self.size_to_free_blocks[buddy_size] = new_start_addrs
            for addr in new_start_addrs:
                self.addr_to_size[addr] = buddy_size
            start_addrs.extend(new_start_addrs)
            buddy_size //= 2

    def can_allocate(self, sizes: List[int]) -> bool:
        # FIXME(woosuk): Must be fixed for performance.
        size_to_free_blocks = copy.deepcopy(self.size_to_free_blocks)
        addr_to_size = copy.deepcopy(self.addr_to_size)
        for size in sizes:
            try:
                self.allocate(size, size_to_free_blocks, addr_to_size)
            except ValueError:
                return False
        return True

    def _resize(self, size: int) -> int:
        # Bump up the size to the next power of 2.
        size = 2 ** math.ceil(math.log2(size)) 
        # Make sure the size is not smaller than the min block size.
        size = max(size, self.min_block_size)
        return size

    def allocate(
        self,
        size: int,
        size_to_free_blocks: Optional[Dict[int, List[int]]] = None,
        addr_to_size: Optional[Dict[int, int]] = None,
    ) -> List[PhysicalTokenBlock]:
        if size_to_free_blocks is None:
            size_to_free_blocks = self.size_to_free_blocks
        if addr_to_size is None:
            addr_to_size = self.addr_to_size

        size = self._resize(size)
        if size > self.max_block_size:
            raise ValueError(
                f'Size {size} is larger than max_block_size {self.max_block_size}.')

        # Find the smallest block that can fit the size.
        i = size
        while True:
            if len(size_to_free_blocks[i]) > 0:
                # Found a block.
                start = size_to_free_blocks[i].pop()
                addr_to_size[start] = size

                # Split the block.
                while i > size:
                    i //= 2
                    size_to_free_blocks[i].append(start + i)
                    addr_to_size[start + i] = i

                # Return the blocks.
                physical_blocks = []
                for j in range(size):
                    physical_block = PhysicalTokenBlock(
                        device=self.device,
                        block_number=start + j,
                        block_size=self.token_block_size,
                    )
                    physical_block.ref_count = 1
                    physical_blocks.append(physical_block)
                return physical_blocks
            else:
                i *= 2
                if i > self.max_block_size:
                    raise ValueError(f'Cannot find a block of size {size}.')

    def free(self, start: int) -> None:
        size = self.addr_to_size[start]
        del self.addr_to_size[start]

        # Merge the block with its buddy.
        while size < self.max_block_size:
            buddy = start ^ size
            if buddy in self.addr_to_size and self.addr_to_size[buddy] == size:
                # Found a buddy.
                if buddy in self.size_to_free_blocks[size]:
                    self.size_to_free_blocks[size].remove(buddy)
                    del self.addr_to_size[buddy]
                    size *= 2
                    start = min(start, buddy)
                else:
                    break
            else:
                break
        self.size_to_free_blocks[size].append(start)
        self.addr_to_size[start] = size

    def get_num_free_blocks(self) -> int:
        total = 0
        for size, free_blocks in self.size_to_free_blocks.items():
            total += size * len(free_blocks)
        return total


BlockTable = List[PhysicalTokenBlock]


class BuddyBlockSpaceManager:

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        len_estimator: str,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.len_estimator = len_estimator

        self.gpu_allocator = BuddyAllocator(
            Device.GPU, block_size, num_gpu_blocks)

        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[int, BlockTable] = {}

        # Mapping src physical block number -> List[dst physical block number].
        self.forked: Dict[int, List[int]] = {}

    def _oracle(self, seq_group: SequenceGroup) -> int:
        return seq_group.max_num_steps

    def _next_power_of_two(self, seq_group: SequenceGroup) -> int:
        output_len = seq_group.max_num_steps
        return 1 << (output_len - 1).bit_length()

    def _constant(self, seq_group: SequenceGroup) -> int:
        # FIXME
        return _MAX_SEQ_LEN

    def _compute_allocation_size(self, seq_group: SequenceGroup) -> int:
        if self.len_estimator == 'oracle':
            output_len = self._oracle(seq_group)
        elif self.len_estimator == 'power2':
            output_len = self._next_power_of_two(seq_group)
        elif self.len_estimator == 'constant':
            output_len = self._constant(seq_group)
        seq = seq_group.seqs[0]
        seq_len = min(seq.get_len() + output_len, _MAX_SEQ_LEN)
        size = (seq_len + self.block_size - 1) // self.block_size
        return size

    def can_allocate(self, seq_group: SequenceGroup) -> bool:
        # NOTE: Here we assume that all sequences in the group have the same prompt.
        size = self._compute_allocation_size(seq_group)
        return self.gpu_allocator.can_allocate([size] * len(seq_group.seqs))

    def allocate(self, seq_group: SequenceGroup) -> None:
        # NOTE: Here we assume that all sequences in the group have the same prompt.
        size = self._compute_allocation_size(seq_group)
        for seq in seq_group.seqs:
            self.block_tables[seq.seq_id] = self.gpu_allocator.allocate(size)

    def can_append(self, seq_group: SequenceGroup) -> bool:
        return True

    def append(self, seq: Sequence) -> Dict[int, List[int]]:
        ret: Dict[int, List[int]] = {}
        block_table = self.block_tables[seq.seq_id]
        for block in block_table:
            if block.block_number in self.forked:
                assert block.block_number not in ret
                ret[block.block_number] = self.forked[block.block_number]
                del self.forked[block.block_number]
        return ret

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        src_block_table = self.block_tables[parent_seq.seq_id]
        dst_block_table = self.block_tables[child_seq.seq_id]
        for src_block, dst_block in zip(src_block_table, dst_block_table):
            if src_block.block_number in self.forked:
                self.forked[src_block.block_number].append(dst_block.block_number)
            else:
                self.forked[src_block.block_number] = [dst_block.block_number]

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        return False

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        return False

    def _free_block_table(self, block_table: BlockTable) -> None:
        block = block_table[0]
        self.gpu_allocator.free(block.block_number)
        for block in block_table:
            if block.block_number in self.forked:
                del self.forked[block.block_number]

    def free(self, seq: Sequence) -> None:
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        num_blocks = len(seq.logical_token_blocks)
        block_table = block_table[:num_blocks]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.num_total_cpu_blocks
