import enum
import time
import copy
from typing import Union, Dict, List, Optional, Tuple

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.block_manager import BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceOutputs,
                           SequenceStatus)

logger = init_logger(__name__)


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


class SchedulerOutputs:

    def __init__(
        self,
        scheduled_seq_groups: List[SequenceGroup],
        prompt_run: bool,
        num_batched_tokens: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.prompt_run = prompt_run
        self.num_batched_tokens = num_batched_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
        )

        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: str) -> None:
        for state_queue in [self.waiting, self.running, self.swapped]:
            for seq_group in state_queue:
                if seq_group.request_id == request_id:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.seqs:
                        if seq.is_finished():
                            continue
                        self.free_seq(seq, SequenceStatus.FINISHED_ABORTED)
                    return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def _schedule(self) -> SchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.time()

        # Join waiting sequences if possible.
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            num_batched_tokens = 0
            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting:
                seq_group = self.waiting[0]

                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                prompt_limit = min(
                    self.scheduler_config.max_model_len,
                    self.scheduler_config.max_num_batched_tokens)
                if num_prompt_tokens > prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    break

                # If the sequence group cannot be allocated, stop.
                if not self.block_manager.can_allocate(seq_group):
                    break

                # If the number of batched tokens exceeds the limit, stop.
                if (num_batched_tokens + num_prompt_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.num_seqs(
                    status=SequenceStatus.WAITING)
                num_curr_seqs = sum(
                    seq_group.num_seqs(status=SequenceStatus.RUNNING)
                    for seq_group in self.running)
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_batched_tokens += num_prompt_tokens
                scheduled.append(seq_group)

            if scheduled:
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=num_batched_tokens,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.pop(0)
            while not self.block_manager.can_append_slot(seq_group):
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
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
        self.running = running

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        while self.swapped and not blocks_to_swap_out:
            seq_group = self.swapped[0]
            # If the sequence group has been preempted in this step, stop.
            if seq_group in preempted:
                break
            # If the sequence group cannot be swapped in, stop.
            if not self.block_manager.can_swap_in(seq_group):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
            num_curr_seqs = sum(
                seq_group.num_seqs(status=SequenceStatus.RUNNING)
                for seq_group in self.running)
            if (num_curr_seqs + num_new_seqs >
                    self.scheduler_config.max_num_seqs):
                break

            seq_group = self.swapped.pop(0)
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slot(seq_group, blocks_to_copy)
            self.running.append(seq_group)

        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_data: Dict[int, List[SequenceData]] = {}
            block_tables: Dict[int, List[int]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def update(
        self,
        seq_outputs: Dict[int, SequenceOutputs],
        decode_func,
    ) -> List[SequenceGroup]:
        # Update the running sequences and free blocks.
        seq_outputs_data = self._decode_seq_outputs(seq_outputs, decode_func)
        for i, seq_group in enumerate(self.running):
            try:
                group_outputs_data = seq_outputs_data[i]
            except KeyError:
                continue
            sampling_params = seq_group.sampling_params
            finished = []
            pending = []
            for _, data in group_outputs_data.items():
                stopped, reason = self._stopping_criteria(
                    data[1], sampling_params)
                if stopped:
                    finished.append([data, reason])
                else:
                    pending.append(data)

            if sampling_params.use_beam_search:
                # update finished sequences
                force_stop = False
                length_penalty = sampling_params.length_penalty
                if finished:
                    highest_attainable_score = max([
                        data[0][1].get_score(length_penalty)
                        for data in finished
                    ])
                for j, ((finished_seq_id, finished_seq_data),
                        reason) in enumerate(finished):
                    finished_seqs = sorted(
                        seq_group.get_seqs(
                            status=SequenceStatus.FINISHED_STOPPED) +
                        seq_group.get_seqs(
                            status=SequenceStatus.FINISHED_LENGTH_CAPPED),
                        key=lambda x: x.data.get_score(length_penalty))
                    assert len(finished_seqs) <= sampling_params.n

                    if len(finished_seqs) < sampling_params.n:
                        finished_seq = copy.deepcopy(
                            seq_group.find(finished_seq_id,
                                           status=SequenceStatus.RUNNING))
                        finished_seq.data = copy.deepcopy(finished_seq_data)
                        finished_seq.status = reason
                        seq_group.append_seq(finished_seq)
                    else:
                        worst_seq = finished_seqs[0]
                        worst_score = worst_seq.data.get_score(length_penalty)
                        curr_score = finished_seq_data.get_score(
                            length_penalty)

                        if j == 0 and worst_score >= highest_attainable_score:
                            force_stop = True
                            break

                        if curr_score > worst_score:
                            worst_seq.data = copy.deepcopy(finished_seq_data)
                            worst_seq.status = reason

                if force_stop:
                    for seq in seq_group.get_seqs(
                            status=SequenceStatus.RUNNING):
                        self.block_manager.free(seq)
                        seq_group.seqs.pop(seq_group.seqs.index(seq))
                    continue

                # schedule next-beam tasks
                pending = pending[:sampling_params.n]
                running_ids = [
                    seq.seq_id for seq in seq_group.get_seqs(
                        status=SequenceStatus.RUNNING)
                ]
                all_ids = [j + min(running_ids) for j in range(len(pending))]
                pending_ids = [p[0] for p in pending]
                new_ids = list(set(all_ids) - set(pending_ids))

                used_seq_ids = []
                for (parent_id, seq_data) in pending:
                    parent_seq = seq_group.find(parent_id,
                                                status=SequenceStatus.RUNNING)
                    if parent_id not in used_seq_ids:
                        parent_seq.append_token_id(
                            seq_data.get_last_token_id())
                        parent_seq.data = copy.deepcopy(seq_data)
                        used_seq_ids.append(parent_id)
                    else:
                        new_seq_id = new_ids.pop()
                        used_seq_ids.append(new_seq_id)
                        if new_seq_id in running_ids:
                            new_seq = seq_group.find(
                                new_seq_id, status=SequenceStatus.RUNNING)
                            self.block_manager.free(new_seq)
                            parent_seq.fork(new_seq)
                        else:
                            new_seq = copy.deepcopy(parent_seq)
                            new_seq.seq_id = new_seq_id
                            seq_group.append_seq(new_seq)
                            new_seq = seq_group.find(
                                new_seq_id, status=SequenceStatus.RUNNING)

                        self.block_manager.fork(parent_seq, new_seq)
                        new_seq.append_token_id(seq_data.get_last_token_id())
                        new_seq.data = copy.deepcopy(seq_data)

                for unused_id in list(set(running_ids) - set(used_seq_ids)):
                    try:
                        unused_seq = seq_group.find(
                            unused_id, status=SequenceStatus.RUNNING)
                        self.block_manager.free(unused_seq)
                        seq_group.seqs.pop(seq_group.seqs.index(unused_seq))
                    except ValueError:
                        continue
            else:
                for parent_id, seq_data in pending:
                    parent_seq = seq_group.find(parent_id)
                    parent_seq.append_token_id(seq_data.get_last_token_id())
                    parent_seq.data = copy.deepcopy(seq_data)

                for (parent_id, seq_data), reason in finished:
                    parent_seq = seq_group.find(parent_id)
                    parent_seq.data = copy.deepcopy(seq_data)
                    self.free_seq(parent_seq, reason)

        # Return a shallow copy of the running queue to prevent the queue
        # from being modified by the caller.
        return self.running.copy()

    def _decode_seq_outputs(
        self,
        seq_outputs: List[Dict[int, SequenceOutputs]],
        decode_func,
    ) -> Dict[int, Dict[int, List[Union[int, SequenceData]]]]:
        seq_outputs_data = {}
        seq_outputs_ = {}
        for seq_output in seq_outputs:
            seq_output_ids = [v.parent_seq_id for _, v in seq_output.items()]
            for i, seq_group in enumerate(self.running):
                seq_group_ids = [
                    seq.seq_id for seq in seq_group.get_seqs(
                        status=SequenceStatus.RUNNING)
                ]
                if set(seq_output_ids).issubset(set(seq_group_ids)):
                    seq_outputs_[i] = seq_output
                    break
        for i, seq_group in enumerate(self.running):
            try:
                group_outputs = seq_outputs_[i]
            except KeyError:
                continue
            group_outputs_data = {}
            for pseudo_seq_id, output in group_outputs.items():
                try:
                    parent_seq = seq_group.find(output.parent_seq_id,
                                                status=SequenceStatus.RUNNING)
                except ValueError:
                    continue
                seq_data = copy.deepcopy(parent_seq.data)
                seq_data.append_token_id(output.output_token, output.logprobs)
                new_token, new_output_text = decode_func(
                    prev_output_tokens=seq_data.output_tokens,
                    new_token_id=seq_data.get_last_token_id(),
                )
                if new_token is not None:
                    seq_data.output_tokens.append(new_token)
                    seq_data.output_text = new_output_text
                group_outputs_data[pseudo_seq_id] = [
                    output.parent_seq_id, seq_data
                ]
            seq_outputs_data[i] = group_outputs_data

        return seq_outputs_data

    def _stopping_criteria(
            self, seq_data: SequenceData,
            sampling_params) -> List[Union[bool, SequenceStatus | None]]:
        """Check if the given sequence stopped."""
        for stop_str in sampling_params.stop:
            if seq_data.output_text.endswith(stop_str):
                # Truncate the output text so that the stop string is
                # not included in the output.
                seq_data.output_text = seq_data.output_text[:-len(stop_str)]
                return [True, SequenceStatus.FINISHED_STOPPED]

        # Check if the sequence has reached max_model_len or max_tokens.
        if (seq_data.get_len() >= self.scheduler_config.max_model_len) or \
            (seq_data.get_output_len() >= sampling_params.max_tokens):
            return [True, SequenceStatus.FINISHED_LENGTH_CAPPED]
        # Check if the sequence has generated the EOS token.
        if not sampling_params.ignore_eos:
            if seq_data.get_last_token_id() == sampling_params.eos_token_id:
                return [True, SequenceStatus.FINISHED_STOPPED]

        return [False, None]

    def free_seq(self, seq: Sequence, finish_status: SequenceStatus) -> None:
        seq.status = finish_status
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING

    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
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
            assert False, "Invalid preemption mode."

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.insert(0, seq_group)

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
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
