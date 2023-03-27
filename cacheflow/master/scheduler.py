import time
from typing import Dict, List

from cacheflow.master.block_manager import BlockSpaceManager
from cacheflow.master.frontend import Frontend
from cacheflow.master.policy import PolicyFactory
from cacheflow.sampling_params import SamplingParams
from cacheflow.sequence import Sequence
from cacheflow.sequence import SequenceGroup
from cacheflow.sequence import SequenceGroupInputs
from cacheflow.sequence import SequenceOutputs
from cacheflow.sequence import SequenceStatus


class Scheduler:

    def __init__(
        self,
        frontend: Frontend,
        controllers: List,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        max_num_batched_tokens: int,
    ) -> None:
        self.frontend = frontend
        self.controllers = controllers
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.max_num_batched_tokens = max_num_batched_tokens

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy('fcfs')
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
        # NOTE(woosuk): This is not used for now.
        self.swapped: List[SequenceGroup] = []

    def _fetch_requests(self) -> int:
        inputs = self.frontend.get_inputs()
        for seq_group, sampling_params in inputs:
            self.waiting.append(seq_group)
            self.sampling_params[seq_group.group_id] = sampling_params
        return len(inputs)

    def step(self) -> None:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fetch new requests.
        num_requests = self._fetch_requests()

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
        while self.running:
            seq_group = self.running.pop(0)
            while not self.block_manager.can_append(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop(-1)
                    self._preempt(victim_seq_group)
                    self.waiting.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group)
                    self.waiting.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                self._append(seq_group, blocks_to_copy)
                running.append(seq_group)
        self.running = running

        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running
        )

        # Join new sequences if possible.
        prompt_group_ids: List[int] = []
        self.waiting = self.policy.sort_by_priority(now, self.waiting)
        # FIXME(woosuk): This does not work if sequence groups have more than
        # one sequence.
        while self.waiting:
            seq_group = self.waiting[0]
            # If the sequence group cannot be allocated, stop joining.
            if not self.block_manager.can_allocate(seq_group):
                break

            # If the number of batched tokens exceeds the limit, stop joining.
            num_prompt_tokens = seq_group.seqs[0].get_len()
            if (num_batched_tokens + num_prompt_tokens
                > self.max_num_batched_tokens):
                break

            seq_group = self.waiting.pop(0)
            self._allocate(seq_group)
            self.running.append(seq_group)
            num_batched_tokens += num_prompt_tokens
            prompt_group_ids.append(seq_group.group_id)

        # Create input data structures.
        input_seq_groups: List[SequenceGroupInputs] = []
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
        if input_seq_groups:
            self.controllers[0].execute_stage(
                input_seq_groups,
                blocks_to_copy=blocks_to_copy,
                blocks_to_swap_in={},
                blocks_to_swap_out={},
            )

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
                self._return(seq_group)
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

    def _preempt(self, seq_group: SequenceGroup) -> None:
        # NOTE(woosuk): There are two preemption mechanisms.
        # 1. Swapping: Swap out the blocks of the preempted sequences to CPU
        # memory and swap them back in when the sequences are resumed.
        # 2. Recomputation: Discard the blocks of the preempted sequences and
        # recompute them when the sequences are resumed.
        # We originally used swapping, but it turned out that recomputation
        # is more efficient. We keep the swapping code for future reference.
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)

    def _free_seq(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.FINISHED
        self.block_manager.free(seq)

    def _return(self, seq_group: SequenceGroup) -> None:
        group_id = seq_group.group_id
        del self.num_steps[group_id]
        del self.sampling_params[group_id]
        self.frontend.print_response(seq_group)

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
