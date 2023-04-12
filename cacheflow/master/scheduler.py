import enum
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

from cacheflow.master.block_manager import BlockSpaceManager
from cacheflow.master.policy import PolicyFactory
from cacheflow.sampling_params import SamplingParams
from cacheflow.sequence import Sequence
from cacheflow.sequence import SequenceGroup
from cacheflow.sequence import SequenceGroupInputs
from cacheflow.sequence import SequenceOutputs
from cacheflow.sequence import SequenceStatus


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


class Scheduler:

    def __init__(
        self,
        controllers: List,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        max_num_batched_tokens: int,
        max_num_sequences: int,
        collect_stats: bool,
        do_memory_analysis: bool = False,
    ) -> None:
        self.controllers = controllers
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        self.collect_stats = collect_stats
        self.do_memory_analysis = do_memory_analysis

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name='fcfs')
        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
        )

        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Mapping: group_id -> num_steps.
        self.num_steps: Dict[int, int] = {}
        # Mapping: group_id -> sampling params.
        self.sampling_params: Dict[int, SamplingParams] = {}
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []

        # Performance-related statistics.
        self.stats = Stats(num_gpu_blocks, num_cpu_blocks)

    def add_sequence_groups(
        self,
        seq_groups: List[Tuple[SequenceGroup, SamplingParams]],
    ) -> None:
        # Add sequence groups to the waiting queue.
        for seq_group, sampling_params in seq_groups:
            self.waiting.append(seq_group)
            self.sampling_params[seq_group.group_id] = sampling_params

    def _schedule(
        self,
    ) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, List[int]], List[int]]:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.time()

        # NOTE(woosuk): We prioritize the sequence groups in the RUNNING state
        # in order to minimize the preemption overheads.
        # Preemption happens only when there is no available slot to keep all
        # the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.pop(0)
            while not self.block_manager.can_append(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                self._append(seq_group, blocks_to_copy)
                running.append(seq_group)
        self.running = running

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        while self.swapped:
            seq_group = self.swapped[0]
            # If the sequence group has been preempted in this step, stop.
            if seq_group in preempted:
                break
            # If the sequence group cannot be swapped in, stop.
            if not self.block_manager.can_swap_in(seq_group):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
            if len(self.running) + num_seqs > self.max_num_sequences:
                break

            seq_group = self.swapped.pop(0)
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append(seq_group, blocks_to_copy)
            self.running.append(seq_group)

        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running
        )

        # Join waiting sequences if possible.
        prompt_group_ids: List[int] = []
        # NOTE(woosuk): The sequence groups in the SWAPPED state are strictly
        # prioritized over the sequence groups in the WAITING state.
        # This is because we want to bound the amount of CPU memory taken by
        # the swapped sequence groups.
        if not self.swapped:
            self.waiting = self.policy.sort_by_priority(now, self.waiting)
            while self.waiting:
                seq_group = self.waiting[0]
                # If the sequence group has been preempted in this step, stop.
                if seq_group in preempted:
                    break
                # If the sequence group cannot be allocated, stop.
                if not self.block_manager.can_allocate(seq_group):
                    break

                # If the number of batched tokens exceeds the limit, stop.
                num_prompt_tokens = seq_group.seqs[0].get_len()
                if (num_batched_tokens + num_prompt_tokens
                    > self.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_seqs = seq_group.num_seqs(status=SequenceStatus.WAITING)
                if len(self.running) + num_seqs > self.max_num_sequences:
                    break

                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_batched_tokens += num_prompt_tokens
                prompt_group_ids.append(seq_group.group_id)

        if self.collect_stats:
            if self.running or blocks_to_swap_in or blocks_to_swap_out:
                self.stats.timestamps.append(now - self.stats.start_time)
                self.stats.input_lens.append(num_batched_tokens)
                self.stats.swap_out_lens.append(len(blocks_to_swap_out) * self.block_size)
                self.stats.swap_in_lens.append(len(blocks_to_swap_in) * self.block_size)
                self.stats.num_preemption.append(len(preempted))
                self.stats.num_swapped.append(len(self.swapped))
                self.stats.num_running.append(len(self.running))
                self.stats.num_waiting.append(len(self.waiting))

                num_free_gpu_blocks = self.block_manager.get_num_free_gpu_blocks()
                num_used_gpu_blocks = self.num_gpu_blocks - num_free_gpu_blocks
                self.stats.gpu_cache_usage.append(num_used_gpu_blocks / self.num_gpu_blocks)
                num_free_cpu_blocks = self.block_manager.get_num_free_cpu_blocks()
                num_used_cpu_blocks = self.num_cpu_blocks - num_free_cpu_blocks
                self.stats.cpu_cache_usage.append(num_used_cpu_blocks / self.num_cpu_blocks)

                if self.do_memory_analysis:
                    block_tables = self.block_manager.block_tables
                    num_logical_blocks = 0
                    num_logical_tokens = 0
                    num_physical_blocks = 0
                    num_physical_tokens = 0
                    physical_block_numbers = set()
                    num_reserved_tokens = 0
                    for seq_group in self.running:
                        group_id = seq_group.group_id
                        sampling_params = self.sampling_params[group_id]
                        max_num_steps = sampling_params.max_num_steps
                        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                            num_logical_blocks += len(seq.logical_token_blocks)
                            num_logical_tokens += seq.get_len()

                            seq_id = seq.seq_id
                            block_table = block_tables[seq_id]
                            for i, block in enumerate(block_table):
                                if block.block_number in physical_block_numbers:
                                    continue
                                physical_block_numbers.add(block.block_number)
                                num_physical_blocks += 1
                                num_physical_tokens += seq.logical_token_blocks[i].num_tokens
                    
                    assert num_physical_blocks == num_used_gpu_blocks
                    self.stats.num_logical_blocks.append(num_logical_blocks)
                    self.stats.num_logical_tokens.append(num_logical_tokens)
                    self.stats.num_physical_blocks.append(num_physical_blocks)
                    self.stats.num_physical_tokens.append(num_physical_tokens)
                    self.stats.num_reserved_tokens.append(num_reserved_tokens)

        return (blocks_to_swap_in,
                blocks_to_swap_out,
                blocks_to_copy,
                prompt_group_ids)

    def step(self) -> List[SequenceGroup]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_output = self._schedule()
        blocks_to_swap_in = scheduler_output[0]
        blocks_to_swap_out = scheduler_output[1]
        blocks_to_copy = scheduler_output[2]
        prompt_group_ids = scheduler_output[3]

        # Create input data structures.
        input_seq_groups: List[SequenceGroupInputs] = []
        updated_seq_groups: List[SequenceGroup] = self.running.copy()

        for seq_group in self.running:
            group_id = seq_group.group_id
            is_prompt = group_id in prompt_group_ids

            input_tokens: Dict[int, List[int]] = {}
            seq_logprobs: Dict[int, float] = {}
            block_tables: Dict[int, List[int]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                if is_prompt:
                    input_tokens[seq_id] = seq.get_token_ids()
                else:
                    input_tokens[seq_id] = [seq.get_last_token_id()]
                seq_logprobs[seq_id] = seq.cumulative_logprobs
                # NOTE(woosuk): Sequences in the same group have the same
                # sequence length
                seq_len = seq.get_len()

            input_seq_group = SequenceGroupInputs(
                group_id=group_id,
                is_prompt=is_prompt,
                input_tokens=input_tokens,
                context_len=seq_len,
                seq_logprobs=seq_logprobs,
                sampling_params=self.sampling_params[group_id],
                block_tables=block_tables,
            )
            input_seq_groups.append(input_seq_group)

        # Execute the first stage of the pipeline.
        if input_seq_groups or blocks_to_swap_in or blocks_to_swap_out:
            # Swap in and swap out should never happen at the same time.
            assert not (blocks_to_swap_in and blocks_to_swap_out)
            self.controllers[0].execute_stage(
                input_seq_groups,
                blocks_to_swap_in=blocks_to_swap_in,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
            )

        return updated_seq_groups

    def post_step(
        self,
        seq_outputs: Dict[int, SequenceOutputs],
    ) -> None:
        # Update the running sequences and free blocks.
        for seq_group in self.running:
            group_id = seq_group.group_id
            self.num_steps[group_id] += 1
            stop_token_ids = self.sampling_params[group_id].stop_token_ids

            # Process beam search results before processing the next tokens.
            for seq in seq_group.seqs:
                if seq.status == SequenceStatus.FINISHED:
                    continue

                output = seq_outputs[seq.seq_id]
                if seq.seq_id != output.parent_seq_id:
                    # The sequence is a fork of the parent sequence (beam search).
                    # Free the current sequence.
                    self.block_manager.free(seq)
                    # Fork the parent sequence.
                    parent_seq = seq_group.find(output.parent_seq_id)
                    parent_seq.fork(seq)
                    self.block_manager.fork(parent_seq, seq)

            # Process the next tokens.
            for seq in seq_group.seqs:
                if seq.status == SequenceStatus.FINISHED:
                    continue

                # Append a new token to the sequence.
                output = seq_outputs[seq.seq_id]
                seq.append(output.output_token, output.logprobs)

                # Check if the sequence has generated a stop token.
                if output.output_token in stop_token_ids:
                    self._free_seq(seq)
                    continue

                # Check if the sequence has reached the maximum number of steps.
                max_num_steps = self.sampling_params[group_id].max_num_steps
                if self.num_steps[group_id] == max_num_steps:
                    self._free_seq(seq)
                    continue

        # Update the running sequences.
        running: List[SequenceGroup] = []
        for seq_group in self.running:
            if seq_group.is_finished():
                self._free_seq_group(seq_group)
            else:
                running.append(seq_group)
        self.running = running

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.seqs:
            seq.status = SequenceStatus.RUNNING
        # FIXME(woosuk): Support interactive generation.
        if seq_group.group_id not in self.num_steps:
            self.num_steps[seq_group.group_id] = 0

    def _append(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not supported. In such a case,
        # we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
            if len(seqs) == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            assert False, 'Invalid preemption mode.'

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        self.waiting.append(seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        for seq in seqs:
            seq.status = SequenceStatus.SWAPPED
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _free_seq(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.FINISHED
        self.block_manager.free(seq)

    def _free_seq_group(self, seq_group: SequenceGroup) -> None:
        group_id = seq_group.group_id
        del self.num_steps[group_id]
        del self.sampling_params[group_id]

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        assert self.block_manager.can_swap_out(seq_group)
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def reset_stats(self) -> None:
        self.stats.reset(self.num_gpu_blocks, self.num_cpu_blocks)

    def save_stats(
        self,
        output_dir: str,
    ) -> None:
        assert self.collect_stats, 'Statistics collection is disabled.'
        self.stats.save(output_dir)


class Stats:

    def __init__(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        self.start_time: float = time.time()
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks

        self.timestamps: List[float] = []
        self.input_lens: List[int] = []
        self.swap_out_lens: List[int] = []
        self.swap_in_lens: List[int] = []
        self.num_preemption: List[int] = []
        self.num_waiting: List[int] = []
        self.num_running: List[int] = []
        self.num_swapped: List[int] = []
        self.gpu_cache_usage: List[float] = []
        self.cpu_cache_usage: List[float] = []

        self.num_logical_blocks: List[int] = []
        self.num_logical_tokens: List[int] = []
        self.num_physical_blocks: List[int] = []
        self.num_physical_tokens: List[int] = []
        self.num_reserved_tokens: List[int] = []

    def reset(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        self.__init__(num_gpu_blocks, num_cpu_blocks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time,
            'num_gpu_blocks': self.num_gpu_blocks,
            'num_cpu_blocks': self.num_cpu_blocks,
            'timestamps': self.timestamps,
            'input_lens': self.input_lens,
            'swap_out_lens': self.swap_out_lens,
            'swap_in_lens': self.swap_in_lens,
            'num_preemption': self.num_preemption,
            'num_waiting': self.num_waiting,
            'num_running': self.num_running,
            'num_swapped': self.num_swapped,
            'gpu_cache_usage': self.gpu_cache_usage,
            'cpu_cache_usage': self.cpu_cache_usage,
            'num_logical_blocks': self.num_logical_blocks,
            'num_logical_tokens': self.num_logical_tokens,
            'num_physical_blocks': self.num_physical_blocks,
            'num_physical_tokens': self.num_physical_tokens,
            'num_reserved_tokens': self.num_reserved_tokens,
        }

    def save(self, output_dir: str) -> None:
        with open(os.path.join(output_dir, 'stats.pkl'), 'wb') as f:
            pickle.dump(self.to_dict(), f)
