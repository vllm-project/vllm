"""A block manager that manages token blocks."""
from typing import Dict, List, Optional, Tuple

from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device

SeqId = int


class BlockSpaceManagerV2(BlockSpaceManager):
    """BlockSpaceManager which manages the allocation of KV cache.

    It owns responsibility for allocation, swapping, allocating memory for
    autoregressively-generated tokens, and other advanced features such as
    prefix caching, forking/copy-on-write, and sliding-window memory allocation.

    The current implementation is partial; in particular prefix caching and
    sliding-window are not feature complete. This class implements the design
    described in https://github.com/vllm-project/vllm/pull/3492.

    Args:
        block_size (int): The size of each memory block.
        num_gpu_blocks (int): The number of memory blocks allocated on GPU.
        num_cpu_blocks (int): The number of memory blocks allocated on CPU.
        watermark (float, optional): The threshold used for memory swapping.
            Defaults to 0.01.
        sliding_window (Optional[int], optional): The size of the sliding
            window. Defaults to None.
        enable_caching (bool, optional): Flag indicating whether caching is
            enabled. Defaults to False.
    """

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        assert sliding_window is None, "Sliding window not yet supported"

        self.block_sliding_window = None

        self.watermark = watermark
        assert watermark >= 0.0

        assert not enable_caching, "Prefix caching not yet supported"
        self.enable_caching = enable_caching

        self.watermark_blocks = int(watermark * num_gpu_blocks)

        self.block_allocator = CpuGpuBlockAllocator.create(
            # Currently, only naive blocks are supported (no prefix caching).
            allocator_type="naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )

        self.block_tables: Dict[SeqId, BlockTable] = {}

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]

        num_required_blocks = BlockTable.get_num_required_blocks(
            seq.get_token_ids(),
            block_size=self.block_size,
        )

        assert self.block_sliding_window is None
        if self.block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.block_sliding_window)

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            device=Device.GPU)

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def allocate(self, seq_group: SequenceGroup) -> None:
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        assert not (set(seq.seq_id for seq in waiting_seqs)
                    & self.block_tables.keys()), "block table already exists"

        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = waiting_seqs[0]

        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=self.block_allocator,
        )
        assert self.block_sliding_window is None
        block_table.allocate(seq.get_token_ids())
        self.block_tables[seq.seq_id] = block_table

        # Assign the block table for each sequence.
        for seq in waiting_seqs[1:]:
            self.block_tables[seq.seq_id] = block_table.fork()

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            Device.GPU)
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def append_slot(
        self,
        seq: Sequence,
    ) -> Optional[Tuple[int, int]]:

        block_table = self.block_tables[seq.seq_id]

        # Get unseen token ids.
        num_full_slots = block_table.num_full_slots
        unseen_token_ids = seq.get_token_ids()[num_full_slots:]
        assert unseen_token_ids

        block_table.append_token_ids(unseen_token_ids)

        # Return any copy-on-writes.
        _ = self.block_allocator.clear_copy_on_writes()

        # TODO extend append_slot interface to append_slots
        # @cadedaniel will do in https://github.com/vllm-project/vllm/pull/3250

        return None

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        self.block_tables[seq.seq_id].free()
        del self.block_tables[seq.seq_id]

    def get_block_table(self, seq: Sequence) -> List[int]:
        assert seq.seq_id in self.block_tables
        block_ids = self.block_tables[seq.seq_id].physical_block_ids
        assert all(b is not None for b in block_ids)
        return block_ids

    def access_all_blocks_in_seq(self, seq, now):
        # TODO add prefix caching support.
        # Tracked here https://github.com/vllm-project/vllm/issues/3667
        pass

    def mark_blocks_as_computed(self, seq_group: SequenceGroup):
        # We ignore the sequence group as its not necessary. After the batch is
        # formed by the scheduler, we do not need to mark blocks from individual
        # sequence groups as computed -- all blocks in the batch can be marked
        # as computed.
        self.block_allocator.mark_blocks_as_computed()

    def get_common_computed_block_ids(self, seqs: List[Sequence]) -> List[int]:
        """Determine which blocks for which we skip prefill.

        With prefix caching we can skip prefill for previously-generated blocks.
        Currently, the attention implementation only supports skipping cached
        blocks if they are a contiguous prefix of cached blocks.

        This method determines which blocks can be safely skipped for all
        sequences in the sequence group.
        """
        seq_block_ids = [
            self.block_tables[seq.seq_id].physical_block_ids for seq in seqs
        ]
        return self.block_allocator.get_common_computed_block_ids(
            seq_block_ids)

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.fork()

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        return False

    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        raise NotImplementedError

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        return False

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        raise NotImplementedError

    def get_num_free_gpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.GPU)

    def get_num_free_cpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.CPU)
