import math
from typing import List, Optional

from vllm.core.block.common import PhysicalBlockTable, VirtualBlockTable
from vllm.core.block.interfaces import (Block, DeviceAwareBlockAllocator,
                                        CachePolicy)
from vllm.utils import Device, cdiv, chunk_list


class CachePolicyBase(CachePolicy):
    """This cache policy always allocates new blocks to append new tokens.

    Args:
        block_size (int): The maximum number of tokens that can be stored in a
            single block.
        block_allocator (DeviceAwareBlockAllocator): The block allocator used to
            manage memory for the blocks.
        physical_block_table (Optional[List[Block]], optional): An optional list
            of existing blocks to initialize the PhysicalBlockTable with. If not
            provided, an empty PhysicalBlockTable is created.

    Attributes:
        _block_size (int): The maximum number of tokens that can be stored in a
            single block.
        _allocator (DeviceAwareBlockAllocator): The block allocator used to
            manage memory for the blocks.
        _physical_block_table (PhysicalBlockTable): The list of blocks managed
            by this PhysicalBlockTable.
    """

    def __init__(
        self,
        block_size: int,
        block_allocator: DeviceAwareBlockAllocator,
        physical_block_table: Optional[PhysicalBlockTable] = None,
        virtual_block_table: Optional[VirtualBlockTable] = None,
    ):
        self._block_size = block_size
        self._allocator = block_allocator
        if physical_block_table is None:
            physical_block_table = PhysicalBlockTable()
        self._physical_block_table: PhysicalBlockTable = physical_block_table
        if virtual_block_table is None:
            virtual_block_table = VirtualBlockTable(block_size)
        self._virtual_block_table: VirtualBlockTable = virtual_block_table

    def add_tokens_prefill(self,
                           token_ids: List[int],
                           device: Device = Device.GPU,
                           extra_hash: Optional[int] = None) -> None:
        """Allocates memory blocks for storing the given sequence of token IDs
        in prefill stage only.

        This method allocates the required number of blocks to store the given
        sequence of token IDs.

        Args:
            token_ids (List[int]): The sequence of token IDs to be stored.
            device (Device, optional): The device on which the blocks should be
                allocated. Defaults to Device.GPU.
            extra_hash (Optional[int]): The hash value of additional
                factors, such as adapters, that influence the block hash
                in the prefix-caching block.
        """
        assert not self._is_allocated
        assert token_ids
        token_chunks = chunk_list(token_ids, self._block_size)
        blocks = self._allocate_blocks_for_token_ids(prev_block=None,
                                                     token_chunks=token_chunks,
                                                     device=device,
                                                     extra_hash=extra_hash)
        self.update_blocks(blocks)
        self._virtual_block_table.append_tokens(blocks, len(token_ids))

    def update_blocks(self, blocks: List[Block]) -> None:
        """Resets the table to the newly provided blocks 
        """
        self._physical_block_table.update(blocks)

    def add_tokens_decode(self,
                          token_ids: List[int],
                          num_lookahead_slots: int = 0,
                          extra_hash: Optional[int] = None) -> None:
        """Add a sequence of token IDs to the existing blocks in the
        PhysicalBlockTable.

        This method appends the given sequence of token IDs to the existing
        blocks in the PhysicalBlockTable. If there is not enough space in the
        existing blocks, new blocks are allocated using the
        `_ensure_num_empty_slots` method to accommodate the additional tokens.

        The token IDs are divided into chunks of size `block_size` (except for
        the first chunk, which may be smaller), and each chunk is appended to a
        separate block.

        Args:
            token_ids (List[int]): The sequence of token IDs to be appended.
            num_lookahead_slots (int): The number of lookahead slots to allocate
                in speculative decoding or chunked prefill.
            extra_hash (Optional[int]): The hash value of additional
                factors such as adapters that influence the block, apart
                from the token_ids.
        """
        assert self._is_allocated, "no blocks have been allocated"
        assert self.num_physical_blocks > 0

        # Ensure there are enough empty slots for the new tokens plus
        # lookahead slots
        self._ensure_num_empty_slots(num_empty_slots=len(token_ids) +
                                     num_lookahead_slots,
                                     extra_hash=extra_hash)

        # Update the blocks with the new tokens
        first_block_idx = self.num_tokens // self._block_size
        token_blocks = self._chunk_token_blocks(token_ids)

        for i, token_block in enumerate(token_blocks):
            block = self._physical_block_table.append_tokens(
                first_block_idx + i, token_block)
            self._virtual_block_table.append_tokens([block], len(token_block))

    def _ensure_num_empty_slots(self,
                                num_empty_slots: int,
                                extra_hash: Optional[int] = None) -> None:
        """Ensures that the PhysicalBlockTable has at least the specified number
        of empty slots available.

        This method checks if the PhysicalBlockTable has enough empty slots
        (i.e., available space) to accommodate the requested number of tokens.
        If not, it allocates additional blocks on the GPU to ensure that the
        required number of empty slots is available.

        Args:
            num_empty_slots (int): The minimum number of empty slots required.
            extra_hash (Optional[int]): The hash value of additional
                factors such as adapters that influence the block, apart
                from the token_ids.
        """
        # Currently the block table only supports
        # appending tokens to GPU blocks.
        device = Device.GPU
        assert self._is_allocated

        if self._num_empty_slots >= num_empty_slots:
            return

        slots_to_allocate = num_empty_slots - self._num_empty_slots
        blocks_to_allocate = cdiv(slots_to_allocate, self._block_size)

        for _ in range(blocks_to_allocate):
            assert self.num_physical_blocks > 0
            self._physical_block_table.append(
                self._allocator.allocate_mutable_block(
                    prev_block=self._physical_block_table[-1],
                    device=device,
                    extra_hash=extra_hash))

    def fork(self) -> "CachePolicy":
        """Creates a new PhysicalBlockTable instance with a copy of the blocks
        from the current instance.

        This method creates a new PhysicalBlockTable instance with the same
        block size, block allocator, and a copy of the blocks from the current
        instance. The new PhysicalBlockTable has its own independent set of
        blocks, but shares the same underlying memory allocation with the
        original PhysicalBlockTable.

        Returns:
            PhysicalBlockTable: A new PhysicalBlockTable instance with a copy
                of the blocks from the current instance.
        """
        assert self._is_allocated
        assert self.num_physical_blocks > 0
        physical_block_table = PhysicalBlockTable(
            self._allocator.fork(self._physical_block_table[-1]))
        virtual_block_table = self._virtual_block_table.fork()
        return CachePolicyFactory.fork(
            self,
            physical_block_table=physical_block_table,
            virtual_block_table=virtual_block_table)

    def free(self) -> None:
        """Frees the memory occupied by the blocks in the PhysicalBlockTable.

        This method iterates over all the blocks in the `_physical_block_table`
        list and calls the `free` method of the `_allocator` object to release
        the memory occupied by each block. After freeing all the blocks, the
        `_physical_block_table` list is set to `None`.
        """
        for block in self.blocks:
            self._allocator.free(block)
        self._physical_block_table.reset()
        self._virtual_block_table.reset()

    @property
    def physical_block_ids(self) -> List[int]:
        """Returns a list of physical block indices for the blocks in the
        PhysicalBlockTable.

        This property returns a list of integers, where each integer represents
        the physical block index of a corresponding block in the
        `_physical_block_table` list. The physical block index is a unique
        identifier for the memory location occupied by the block.

        Returns:
            List[int]: A list of physical block indices for the blocks in the
                PhysicalBlockTable.
        """
        return self._physical_block_table.ids()

    @property
    def num_physical_blocks(self) -> int:
        return len(self._physical_block_table)

    @property
    def slot_mappings(self) -> List[int]:
        return self._virtual_block_table.slot_mappings

    @property
    def num_tokens(self) -> int:
        return self._virtual_block_table.num_tokens

    def get_unseen_token_ids(self, sequence_token_ids: List[int]) -> List[int]:
        """Get the number of "unseen" tokens in the sequence.

        Unseen tokens are tokens in the sequence corresponding to this block
        table, but are not yet appended to this block table.

        Args:
            sequence_token_ids (List[int]): The list of token ids in the
                sequence.

        Returns:
            List[int]: The postfix of sequence_token_ids that has not yet been
                appended to the block table.
        """

        # The token ids in sequence are append-only, the unseen token ids are
        # the ones after the processed ones in block table.
        return sequence_token_ids[self.num_full_slots:]

    def _allocate_blocks_for_token_ids(
            self,
            prev_block: Optional[Block],
            token_chunks: List[List[int]],
            device: Device,
            extra_hash: Optional[int] = None) -> List[Block]:
        blocks: List[Block] = []

        block_token_ids = []
        tail_token_ids = []
        for cur_token_ids in token_chunks:
            if len(cur_token_ids) == self._block_size:
                block_token_ids.append(cur_token_ids)
            else:
                tail_token_ids.append(cur_token_ids)

        if block_token_ids:
            blocks.extend(
                self._allocator.allocate_immutable_blocks(
                    prev_block,
                    block_token_ids=block_token_ids,
                    device=device,
                    extra_hash=extra_hash))
            prev_block = blocks[-1]

        if tail_token_ids:
            assert len(tail_token_ids) == 1
            cur_token_ids = tail_token_ids[0]

            block = self._allocator.allocate_mutable_block(
                prev_block=prev_block, device=device, extra_hash=extra_hash)
            block.append_token_ids(cur_token_ids)

            blocks.append(block)

        return blocks

    def _get_all_token_ids(self) -> List[int]:
        # NOTE: This function is O(seq_len); use sparingly.
        token_ids: List[int] = []

        if not self._is_allocated:
            return token_ids

        for block in self.blocks:
            token_ids.extend(block.token_ids)

        return token_ids

    @property
    def _is_allocated(self) -> bool:
        return len(self._physical_block_table) > 0

    @property
    def blocks(self) -> List[Block]:
        return self._physical_block_table.list()

    @property
    def _num_empty_slots(self) -> int:
        assert self._is_allocated
        return len(
            self._physical_block_table) * self._block_size - self.num_tokens

    @property
    def num_full_slots(self) -> int:
        """Returns the total number of tokens currently stored in the
        PhysicalBlockTable.

        Returns:
            int: The total number of tokens currently stored in the PhysicalBlockTable.
        """
        return self.num_tokens

    def get_num_blocks_touched_by_append_slots(
            self, token_ids: List[int], num_lookahead_slots: int) -> int:
        """Determine how many blocks will be "touched" by appending the token
        ids.

        This is required for the scheduler to determine whether a sequence can
        continue generation, or if it must be preempted.
        """
        # Math below is equivalent to:
        # all_token_ids = token_ids + [-1] * num_lookahead_slots
        # token_blocks = self._chunk_token_blocks(all_token_ids)
        # return len(token_blocks)

        num_token_ids = len(token_ids) + num_lookahead_slots
        first_chunk_size = self._block_size - (self.num_tokens %
                                               self._block_size)
        num_token_blocks = (1 + math.ceil(
            (num_token_ids - first_chunk_size) / self._block_size))
        return num_token_blocks

    def _chunk_token_blocks(self, token_ids: List[int]) -> List[List[int]]:
        """Split the token ids into block-sized chunks so they can be easily
        appended to blocks. The first such "token block" may have less token ids
        than the block size, since the last allocated block may be partially
        full.

        If no token ids are provided, then no chunks are returned.
        """

        if not token_ids:
            return []

        first_chunk_size = self._block_size - (self.num_tokens %
                                               self._block_size)
        token_blocks = [token_ids[:first_chunk_size]]
        token_blocks.extend(
            chunk_list(token_ids[first_chunk_size:], self._block_size))
        return token_blocks


class CachePolicySlidingWindow(CachePolicyBase):
    """This cache policy has a sliding-window context and a fixed cache space
    as a result.

    Args:
        num_sliding_window_blocks (int): The number of blocks to keep around
            for a sequence. It should at least fit the sliding window size of
            the context.
    """

    def __init__(
        self,
        block_size: int,
        block_allocator: DeviceAwareBlockAllocator,
        physical_block_table: Optional[PhysicalBlockTable] = None,
        virtual_block_table: Optional[VirtualBlockTable] = None,
        num_sliding_window_blocks: Optional[int] = None,
    ):
        super().__init__(block_size, block_allocator, physical_block_table,
                         virtual_block_table)
        assert num_sliding_window_blocks is not None
        self._num_sliding_window_blocks = num_sliding_window_blocks

    def add_tokens_prefill(self,
                           token_ids: List[int],
                           device: Device = Device.GPU,
                           extra_hash: Optional[int] = None) -> None:
        """Allocate memory blocks for storing the given sequence of token IDs
        in prefill stage only.

        This method allocates the required number of blocks to store the given
        sequence of token IDs only inside the sliding window, or
        _num_sliding_window_blocks to be exact.
        """
        assert not self._is_allocated
        assert token_ids

        block_start_idx = 0
        token_chunks = list(chunk_list(token_ids, self._block_size))
        num_evicted_chunks = len(
            token_chunks) - self._num_sliding_window_blocks

        if num_evicted_chunks > 0:
            num_windows = len(token_chunks) // self._num_sliding_window_blocks
            last_window_end = num_windows * self._num_sliding_window_blocks
            last_window_start = last_window_end - self._num_sliding_window_blocks
            last_window_chunks = token_chunks[
                last_window_start:last_window_end]
            remainder_chunks = token_chunks[last_window_end:]

            # The remainder chunks cannot fill up a window, we need to rotate
            # these chunks back to the front of the last full sliding window.
            if len(remainder_chunks) > 0:
                chunk_idx = len(remainder_chunks[:-1])
                last_window_chunks[:chunk_idx] = remainder_chunks[:chunk_idx]
                last_window_chunks[
                    chunk_idx][:len(remainder_chunks[chunk_idx])] = (
                        remainder_chunks[chunk_idx])
                block_start_idx = chunk_idx
                if len(remainder_chunks[chunk_idx]) == self._block_size:
                    block_start_idx = chunk_idx + 1
            token_chunks = last_window_chunks

        blocks = self._allocate_blocks_for_token_ids(prev_block=None,
                                                     token_chunks=token_chunks,
                                                     device=device,
                                                     extra_hash=extra_hash)
        self.update_blocks(blocks)

        num_evicted_tokens = 0
        if num_evicted_chunks > 0:
            # Chronologically, we maintain the chunk order in the sequence to
            # be added into the block tables.
            blocks = blocks[block_start_idx:] + blocks[:block_start_idx]

            # Allocate null blocks to represent the evicted tokens.
            null_block = self._allocator.allocate_or_get_null_block()
            evicted_blocks = [null_block] * num_evicted_chunks
            num_evicted_tokens = (
                len(token_ids) -
                self._num_sliding_window_blocks * self._block_size)
            self._virtual_block_table.append_tokens(evicted_blocks,
                                                    num_evicted_tokens,
                                                    evicted=True)
            # Partially filled block actually appears twice due to rotation.
            if num_evicted_tokens % self._block_size != 0:
                blocks.append(blocks[0])

        self._virtual_block_table.append_tokens(
            blocks, len(token_ids[num_evicted_tokens:]))

    def add_tokens_decode(self,
                          token_ids: List[int],
                          num_lookahead_slots: int = 0,
                          extra_hash: Optional[int] = None) -> None:
        """Add a sequence of token IDs to the blocks in the PhysicalBlockTable
        by rotating the blocks when appending new tokens when the sliding window
        is full. This means the currently oldest tokens are evicted and replaced
        with new tokens.

        """
        assert self._is_allocated, "no blocks have been allocated"
        assert self.num_physical_blocks > 0

        # Rotate and reuse blocks beyond sliding window so that no new blocks
        # are needed
        assert self.num_physical_blocks <= self._num_sliding_window_blocks
        if self.num_physical_blocks < self._num_sliding_window_blocks:
            # Ensure there are enough empty slots for the new tokens plus
            # lookahead slots
            self._ensure_num_empty_slots(num_empty_slots=len(token_ids) +
                                         num_lookahead_slots,
                                         extra_hash=extra_hash)

        # Update the blocks with the new tokens
        first_block_idx = (self.num_tokens // self._block_size %
                           self.num_physical_blocks)
        token_blocks = self._chunk_token_blocks(token_ids)

        slot_offsets = [0] * len(token_blocks)
        slot_offsets[0] = self.num_tokens % self._block_size
        for i, (slot_offset,
                token_block) in enumerate(zip(slot_offsets, token_blocks),
                                          start=first_block_idx):
            i %= self.num_physical_blocks
            block = self._physical_block_table.insert_tokens(
                i, slot_offset, token_block)
            self._virtual_block_table.insert_tokens(block, slot_offset,
                                                    len(token_block))


class CachePolicyFactory:

    @staticmethod
    def create(
        num_sliding_window_blocks: Optional[int],
        block_size: int,
        block_allocator: DeviceAwareBlockAllocator,
        physical_block_table: Optional[PhysicalBlockTable] = None,
        virtual_block_table: Optional[VirtualBlockTable] = None,
    ) -> "CachePolicy":
        if num_sliding_window_blocks is None:
            return CachePolicyBase(
                block_size=block_size,
                block_allocator=block_allocator,
                physical_block_table=physical_block_table,
                virtual_block_table=virtual_block_table,
            )
        else:
            return CachePolicySlidingWindow(
                block_size=block_size,
                block_allocator=block_allocator,
                num_sliding_window_blocks=num_sliding_window_blocks,
                physical_block_table=physical_block_table,
                virtual_block_table=virtual_block_table,
            )

    @staticmethod
    def fork(
        instance: CachePolicy,
        physical_block_table: PhysicalBlockTable,
        virtual_block_table: VirtualBlockTable,
    ) -> "CachePolicy":
        if hasattr(instance, "_num_sliding_window_blocks"):
            num_sliding_window_blocks = instance._num_sliding_window_blocks  # type: ignore
        else:
            num_sliding_window_blocks = None

        return CachePolicyFactory.create(
            block_size=instance._block_size,  # type: ignore
            block_allocator=instance._allocator,  # type: ignore
            num_sliding_window_blocks=num_sliding_window_blocks,
            physical_block_table=physical_block_table,
            virtual_block_table=virtual_block_table,
        )
