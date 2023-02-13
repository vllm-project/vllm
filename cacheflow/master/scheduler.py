from typing import Dict, List, Tuple

from cacheflow.master.block_manager import BlockSpaceManager
from cacheflow.sequence import Sequence
from cacheflow.sequence import SequenceGroup
from cacheflow.sequence import SequenceStatus


class Scheduler:

    def __int__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        self.block_manager = BlockSpaceManager(
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
        )

        # Serving sequence groups (FIFO).
        self.serving: List[SequenceGroup] = []
        # Mapping: group_id -> num_steps.
        self.num_steps: Dict[int, int] = {}
        # Mapping: group_id -> max_num_steps.
        self.max_num_steps: Dict[int, int] = {}
        # Mapping: group_id -> stop_token_ids.
        self.stop_token_ids: Dict[int, List[int]] = {}

        # Swapped sequence groups (LIFO).
        self.swapped: List[SequenceGroup] = []

        # Pending sequence groups (FIFO).
        self.queue: List[SequenceGroup] = []

    def _free_seq(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.FINISHED
        self.block_manager.free(seq)

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.seqs:
            seq.status = SequenceStatus.RUNNING
        self.serving.append(seq_group)

    def _append(self, seq_group: SequenceGroup) -> None:
        for seq in seq_group.seqs:
            if seq.status == SequenceStatus.FINISHED:
                continue
            ret = self.block_manager.append(seq)
            if ret is not None:
                src_block, dst_block = ret
                # TODO: Issue COPY commands to the workers.

    def _swap_in(self, seq_group: SequenceGroup) -> None:
        # TODO: Issue SWAP_IN commands to the workers.
        self.block_manager.swap_in(seq_group)
        self.block_manager.append(seq_group)
        for seq in seq_group.seqs:
            if seq.status == SequenceStatus.SWAPPED:
                seq.status = SequenceStatus.RUNNING
        self.serving.append(seq_group)

    def _swap_out(self, seq_group: SequenceGroup) -> None:
        assert self.block_manager.can_swap_out(seq_group)
        # TODO: Issue SWAP_OUT commands to the workers.
        self.block_manager.swap_out(seq_group)
        for seq in seq_group.seqs:
            if seq.status == SequenceStatus.RUNNING:
                seq.status = SequenceStatus.SWAPPED
        self.swapped.append(seq_group)

    def step(self) -> None:
        # 1. Prepare new slots for the running sequences.
        # NOTE: Here we implicitly assume FCFS scheduling.
        # That is, the most recently added sequence group is the first
        # to be swapped out.
        victim_idx = len(self.serving) - 1
        for i, seq_group in enumerate(self.serving):
            if i > victim_idx:
                # The i-th sequence group has already been swapped out.
                break
            # OOM. Swap out the victim sequence groups.
            while not self.block_manager.can_append(seq_group):
                victim_seq_group = self.serving[victim_idx]
                self._swap_out(victim_seq_group)
                victim_idx -= 1
                if i > victim_idx:
                    # No other sequence groups can be swapped out.
                    break
            else:
                self._append(seq_group)
        self.serving = self.serving[:victim_idx + 1]

        # 2. Swap in the swapped sequences if possible.
        # NOTE: Here we implicitly assume FCFS scheduling.
        # The swapped sequences are in LIFO order.
        for i, seq_group in enumerate(reversed(self.swapped)):
            if self.block_manager.can_swap_in(seq_group):
                self._swap_in(seq_group)
            else:
                # OOM. Stop swapping.
                self.swapped = self.swapped[:len(self.swapped) - i]
                break
        else:
            # All swapped sequences are swapped in.
            self.swapped.clear()

        # 3. Join new sequences if possible.
        # NOTE: Here we implicitly assume FCFS scheduling.
        # TODO(woosuk): Add a heuristic to control the maximum batch size.
        if not self.swapped:
            for i, seq_group in enumerate(self.queue):
                if self.block_manager.can_allocate(seq_group):
                    self._allocate(seq_group)
                else:
                    # FIXME: Consider the race condition.
                    self.queue = self.queue[i:]
                    break

    def post_step(
        self,
        next_tokens: Dict[int, Tuple[int, int]],
    ) -> None:
        # Update the running sequences and free blocks.
        for seq_group in self.serving:
            group_id = seq_group.group_id
            self.num_steps[group_id] += 1
            stop_token_ids = self.stop_token_ids[group_id]

            for seq in seq_group.seqs:
                if seq.status == SequenceStatus.FINISHED:
                    continue

                parent_seq_id, next_token = next_tokens[seq.seq_id]
                if seq.seq_id != parent_seq_id:
                    # The sequence is a fork of the parent sequence (beam search).
                    # Free the current sequence.
                    self.block_manager.free(seq)
                    # Fork the parent sequence.
                    parent_seq = seq_group.find(parent_seq_id)
                    seq.logical_token_blocks = parent_seq.logical_token_blocks.copy()
                    self.block_manager.fork(parent_seq, seq)

                # Append a new token to the sequence.
                seq.append(next_token)

                # Check if the sequence has generated a stop token.
                if next_token in stop_token_ids:
                    self._free_seq(seq)
                    continue

                # Check if the sequence has reached the maximum number of steps.
                if self.num_steps[group_id] == self.max_num_steps[group_id]:
                    self._free_seq(seq)
                    continue

        # Update the serving states.
        serving: List[SequenceGroup] = []
        for seq_group in self.serving:
            if seq_group.num_seqs(status=SequenceStatus.RUNNING) == 0:
                del self.num_steps[seq_group.group_id]
                del self.max_num_steps[seq_group.group_id]
                del self.stop_token_ids[seq_group.group_id]
                # TODO: Return the seq_group to the client.
            else:
                serving.append(seq_group)
        self.serving = serving
