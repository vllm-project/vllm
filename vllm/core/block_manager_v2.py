"""A block manager that manages token blocks."""
from itertools import chain
from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Tuple

from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import Block
from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device

SeqId = int
EncoderSeqId = str


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

        self.sliding_window = sliding_window
        # max_block_sliding_window is the max number of blocks that need to be
        # allocated
        self.max_block_sliding_window = None
        if sliding_window is not None:
            # +1 here because // rounds down
            num_blocks = sliding_window // block_size + 1
            # +1 here because the last block may not be full,
            # and so the sequence stretches one more block at the beginning
            # For example, if sliding_window is 3 and block_size is 4,
            # we may need 2 blocks when the second block only holds 1 token.
            self.max_block_sliding_window = num_blocks + 1

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
        self.cross_block_tables: Dict[EncoderSeqId, BlockTable] = {}

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = BlockTable.get_num_required_blocks(
            seq.get_token_ids(),
            block_size=self.block_size,
        )

        if seq_group.is_encoder_decoder():
            num_required_blocks += BlockTable.get_num_required_blocks(
                seq_group.get_encoder_seq().get_token_ids(),
                block_size=self.block_size,
            )

        if self.max_block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.max_block_sliding_window)

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

    def _allocate_sequence(self, seq: Sequence) -> BlockTable:
        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=self.block_allocator,
            max_block_sliding_window=self.max_block_sliding_window,
        )
        block_table.allocate(seq.get_token_ids())

        return block_table

    def allocate(self, seq_group: SequenceGroup) -> None:

        # Allocate self-attention block tables for decoder sequences
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        assert not (set(seq.seq_id for seq in waiting_seqs)
                    & self.block_tables.keys()), "block table already exists"

        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = waiting_seqs[0]
        block_table: BlockTable = self._allocate_sequence(seq)
        self.block_tables[seq.seq_id] = block_table

        # Assign the block table for each sequence.
        for seq in waiting_seqs[1:]:
            self.block_tables[seq.seq_id] = block_table.fork()

        # Allocate cross-attention block table for encoder sequence
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # encoder prompt.
        request_id = seq_group.request_id

        assert (request_id
                not in self.cross_block_tables), \
                "block table already exists"

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        if seq_group.is_encoder_decoder():
            block_table = self._allocate_sequence(seq_group.get_encoder_seq())
            self.cross_block_tables[request_id] = block_table

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
            num_computed_slots=seq.data.get_num_computed_tokens(),
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

    def free_cross(self, seq_group: SequenceGroup) -> None:
        request_id = seq_group.request_id
        if request_id not in self.cross_block_tables:
            # Already freed or hasn't been scheduled yet.
            return
        self.cross_block_tables[request_id].free()
        del self.cross_block_tables[request_id]

    def get_block_table(self, seq: Sequence) -> List[int]:
        assert seq.seq_id in self.block_tables
        block_ids = self.block_tables[seq.seq_id].physical_block_ids
        assert all(b is not None for b in block_ids)
        return block_ids  # type: ignore

    def get_cross_block_table(self, seq_group: SequenceGroup) -> List[int]:
        request_id = seq_group.request_id
        assert request_id in self.cross_block_tables
        block_ids = self.cross_block_tables[request_id].physical_block_ids
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
        """Returns the AllocStatus for the given sequence_group 
        with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for the given sequence group.
        """
        return self._can_swap(seq_group, Device.GPU, SequenceStatus.SWAPPED,
                              num_lookahead_slots)

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        """Returns the block id mapping (from CPU to GPU) generated by
        swapping in the given seq_group with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap in.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from CPU 
                to GPU.
        """
        blocks = self._get_blocks_for_swap(seq_group, SequenceStatus.SWAPPED)
        current_swap_mapping = self.block_allocator.swap(
            blocks=blocks, source_device=Device.CPU, dest_device=Device.GPU)

        block_number_mapping = {
            self.block_allocator.get_physical_block_id(Device.CPU,
                                                       cpu_block_id):
            self.block_allocator.get_physical_block_id(Device.GPU,
                                                       gpu_block_id)
            for cpu_block_id, gpu_block_id in current_swap_mapping.items()
        }
        # convert to list of tuples once here
        return list(block_number_mapping.items())

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        """Returns whether we can swap out the given sequence_group 
        with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap in.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            bool: Whether it's possible to swap out current sequence group.
        """
        alloc_status = self._can_swap(seq_group, Device.CPU,
                                      SequenceStatus.RUNNING)
        if alloc_status == AllocStatus.OK:
            return True
        return False

    def swap_out(self, sequence_group: SequenceGroup) -> List[Tuple[int, int]]:
        """Returns the block id mapping (from GPU to CPU) generated by
        swapping out the given sequence_group with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from 
                GPU to CPU.
        """
        blocks = self._get_blocks_for_swap(sequence_group,
                                           SequenceStatus.RUNNING)
        current_swap_mapping = self.block_allocator.swap(
            blocks=blocks, source_device=Device.GPU, dest_device=Device.CPU)
        block_number_mapping = {
            self.block_allocator.get_physical_block_id(Device.GPU,
                                                       gpu_block_id):
            self.block_allocator.get_physical_block_id(Device.CPU,
                                                       cpu_block_id)
            for gpu_block_id, cpu_block_id in current_swap_mapping.items()
        }
        # convert to list of tuples once here
        return list(block_number_mapping.items())

    def get_num_free_gpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.GPU)

    def get_num_free_cpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.CPU)

    def _can_swap(self,
                  seq_group: SequenceGroup,
                  device: Device,
                  status: SequenceStatus,
                  num_lookahead_slots: int = 0) -> AllocStatus:
        """Returns the AllocStatus for swapping in/out the given sequence_group 
        on to the 'device'.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.
            device (Device): device to swap the 'seq_group' on.
            status (SequenceStatus): The status of sequence which is needed
                for action. RUNNING for swap out and SWAPPED for swap in
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for swapping in/out the given 
                sequence_group on to the 'device'.
        """
        blocks = self._get_blocks_for_swap(seq_group, status)
        num_blocks_touched = self.block_allocator.get_num_blocks_touched(
            blocks, device, num_lookahead_slots)
        watermark_blocks = 0
        if device == Device.GPU:
            watermark_blocks = self.watermark_blocks
        if self.block_allocator.get_num_total_blocks(
                device) < num_blocks_touched:
            return AllocStatus.NEVER
        elif self.block_allocator.get_num_free_blocks(
                device) - num_blocks_touched >= watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def _get_blocks_for_swap(self, seq_group: SequenceGroup,
                             status: SequenceStatus) -> List[Block]:
        """Returns the list of blocks those are touched by the seq_group
        
        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.
            status (SequenceStatus): The status of sequence which is needed
                for action. RUNNING for swap out and SWAPPED for swap in
        
        Returns:
            The list of blocks those are touched by the seq_group.
        """
        blocks: Dict[int, List[Block]] = {}
        for seq in seq_group.get_seqs(status=status):
            block_table = self.block_tables[seq.seq_id]
            if block_table.blocks is not None:
                blocks[seq.seq_id] = block_table.blocks
        combined_blocks = list(chain(*blocks.values()))
        return combined_blocks
