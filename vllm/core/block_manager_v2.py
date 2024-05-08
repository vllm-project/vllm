"""A block manager that manages token blocks."""
from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Tuple

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

    Lookahead slots
        The block manager has the notion of a "lookahead slot". These are slots
        in the KV cache that are allocated for a sequence. Unlike the other
        allocated slots, the content of these slots is undefined -- the worker
        may use the memory allocations in any way.

        In practice, a worker could use these lookahead slots to run multiple
        forward passes for a single scheduler invocation. Each successive
        forward pass would write KV activations to the corresponding lookahead
        slot. This allows low inter-token latency use-cases, where the overhead
        of continuous batching scheduling is amortized over >1 generated tokens.

        Speculative decoding uses lookahead slots to store KV activations of
        proposal tokens.

        See https://github.com/vllm-project/vllm/pull/3250 for more information
        on lookahead scheduling.

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

        self.enable_caching = enable_caching

        self.watermark_blocks = int(watermark * num_gpu_blocks)

        self.block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="prefix_caching" if enable_caching else "naive",
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

    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int) -> bool:
        """Determine if there is enough space in the GPU KV cache to continue
        generation of the specified sequence group.

        We use a worst-case heuristic: assume each touched block will require a
        new allocation (either via CoW or new block). We can append slots if the
        number of touched blocks is less than the number of free blocks.

        "Lookahead slots" are slots that are allocated in addition to the slots
        for known tokens. The contents of the lookahead slots are not defined.
        This is used by speculative decoding when speculating future tokens.
        """

        num_touched_blocks = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            block_table = self.block_tables[seq.seq_id]

            num_touched_blocks += (
                block_table.get_num_blocks_touched_by_append_slots(
                    token_ids=block_table.get_unseen_token_ids(
                        seq.get_token_ids()),
                    num_lookahead_slots=num_lookahead_slots,
                ))

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            Device.GPU)
        return num_touched_blocks <= num_free_gpu_blocks

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:

        block_table = self.block_tables[seq.seq_id]

        block_table.append_token_ids(
            token_ids=block_table.get_unseen_token_ids(seq.get_token_ids()),
            num_lookahead_slots=num_lookahead_slots,
        )

        # Return any new copy-on-writes.
        new_cows = self.block_allocator.clear_copy_on_writes()
        return new_cows

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
        return block_ids  # type: ignore

    def access_all_blocks_in_seq(self, seq: Sequence, now: float):
        # Update the last accessed time of all the blocks accessed
        # in this step.
        # And the accessed time is only useful for prefix caching now,
        # as it support internal evictor policy for which cached
        # block could be refilled, to keep cached content could be reused
        # at max extend.
        if self.enable_caching:
            block_table = self.block_tables[seq.seq_id]
            block_ids = []
            for block_id in block_table.physical_block_ids:
                block_ids.append(block_id)
            self.block_allocator.mark_blocks_as_accessed(
                block_ids,  # type: ignore
                now)

    def mark_blocks_as_computed(self, seq_group: SequenceGroup):
        # The only need for mark block as computed is for prefix caching,
        # while currently we could determine whether one block is computed
        # or not by check whether it has content hash.
        # So this function is useless for block_v2.
        pass

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
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
        # NOTE(sang): This assumes seq_block_ids doesn't contain any None.
        return self.block_allocator.get_common_computed_block_ids(
            seq_block_ids)  # type: ignore

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.fork()

    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        return AllocStatus.LATER

    def swap_in(self, seq_group: SequenceGroup,
                num_lookahead_slots: int) -> List[Tuple[int, int]]:
        raise NotImplementedError

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        return False

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        raise NotImplementedError

    def get_num_free_gpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.GPU)

    def get_num_free_cpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.CPU)
