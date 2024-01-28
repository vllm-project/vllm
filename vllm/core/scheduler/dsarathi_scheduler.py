import time
from typing import List, Dict, Type

from vllm.logger import init_logger
from vllm.sequence import SequenceGroup
from vllm.sequence_status import SequenceStatus
from vllm.core.scheduler.base_scheduler import SchedulerOutputs
from vllm.core.scheduler.sarathi_scheduler import SarathiScheduler
from vllm.core.block_space_manager.base_block_space_manager import BaseBlockSpaceManager
from vllm.core.block_space_manager.dsarathi_block_space_manager import DSarathiBlockSpaceManager

logger = init_logger(__name__)


class DSarathiScheduler(SarathiScheduler):

    def _get_block_space_manager_class(self) -> Type[BaseBlockSpaceManager]:
        return DSarathiBlockSpaceManager

    def _schedule(self) -> SchedulerOutputs:
        # Blocks that need to be swapped or copied before model execution
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time
        now = time.monotonic()

        running: List[SequenceGroup] = []
        ignored_seq_groups: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        prefill_seq_groups: List[SequenceGroup] = []
        decode_seq_groups: List[SequenceGroup] = []
        prefill_prompt_chunk_lens: List[int] = []
        decode_prompt_chunk_lens: List[int] = []  # just a list of zeros
        num_batched_tokens = 0
        batch_contains_prefill = False
        num_batched_output_tokens = 0

        ######################################################################
        # Phase 1: Add existing running sequence groups to the batch.
        # There are two cases:
        # 1. The sequence group has incomplete prefill. The routine
        # remains identical to the one in sarathi scheduler for such sequences.
        # 2. The sequence group has completed prefill. In this case, we need to
        # check for memory availability for the next chunk of decode tokens, and preempt
        # some sequence groups if necessary. Note that, the preempted sequence groups
        # might belong to either of the two categories.
        ######################################################################

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # in first pass process all the requests with prefill completed
        # this allows us to accurately account for the number of decode tokens
        running_prefills: List[SequenceGroup] = []

        while self.running:
            seq_group = self.running.pop(0)
            seq = seq_group.get_seqs()[0]

            if seq.get_status() != SequenceStatus.PAUSED:
                running.append(seq_group)
                continue

            if not seq.is_prompt_processing_finished():
                running_prefills.append(seq_group)
                continue

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
                num_seqs = seq_group.num_seqs(status=SequenceStatus.PAUSED)
                num_batched_tokens += num_seqs
                num_batched_output_tokens += num_seqs
                decode_seq_groups.append(seq_group)
                decode_prompt_chunk_lens.append(0)
                running.append(seq_group)

        # now add the requests with prefill incomplete
        # the memory for all these prefills has already been allocated
        # so we should be able to run all of them
        for seq_group in running_prefills:
            seq = seq_group.get_seqs()[0]

            assert not seq.is_prompt_processing_finished()

            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, batch_contains_prefill, num_batched_tokens)

            # as long as the request could fit in the batch previously
            # it should be able to fit in the batch now
            # so in non-pipeline case this condition should always be false
            # however, in pipeline case, the grouping of requests can change
            # between different microbatches, so this is not guaranteed to be always true
            if next_num_prefill_tokens == 0:
                running.append(seq_group)
                continue

            batch_contains_prefill = True
            num_batched_tokens += next_num_prefill_tokens
            prefill_prompt_chunk_lens.append(next_num_prefill_tokens)
            prefill_seq_groups.append(seq_group)
            running.append(seq_group)

        if preempted:
            # make sure that prefills are at the start of the batch, so that we don't violate assumptions
            # made in the original vllm codebase
            self.running = running
            scheduled = prefill_seq_groups + decode_seq_groups
            prompt_chunk_lens = prefill_prompt_chunk_lens + decode_prompt_chunk_lens

            scheduler_outputs = SchedulerOutputs(
                id=self._iteration_id,
                scheduled_seq_groups=scheduled,
                prompt_chunk_lens=prompt_chunk_lens,
                num_batched_prompt_tokens=sum(prompt_chunk_lens),
                num_batched_output_tokens=num_batched_output_tokens,
                num_batched_tokens=num_batched_tokens,
                blocks_to_swap_in=blocks_to_swap_in,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
                ignored_seq_groups=ignored_seq_groups,
            )
            return scheduler_outputs

        ######################################################################
        # Phase 2: Add swapped out sequence groups to the batch.
        ######################################################################
        self.swapped = self.policy.sort_by_priority(now, self.swapped)

        while self.swapped:
            seq_group = self.swapped[0]
            seq = seq_group.get_seqs()[0]

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            if len(running) >= self.scheduler_config.max_num_seqs:
                break

            # If the sequence group cannot be swapped in, stop.
            if not self.block_manager.can_swap_in(seq_group):
                break

            # we don't know if the prefill is complete or not
            next_num_prefill_tokens = 0
            if not seq.is_prompt_processing_finished():
                next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                    seq, batch_contains_prefill, num_batched_tokens)
                assert next_num_prefill_tokens > 0
                batch_contains_prefill = True

            seq_group = self.swapped.pop(0)
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slot(seq_group, blocks_to_copy)
            if seq.is_prompt_processing_finished():
                decode_seq_groups.append(seq_group)
                decode_prompt_chunk_lens.append(0)
                num_new_seqs = seq_group.get_max_num_running_seqs()
                num_batched_tokens += num_new_seqs
            else:
                prefill_seq_groups.append(seq_group)
                prefill_prompt_chunk_lens.append(next_num_prefill_tokens)
                num_batched_tokens += next_num_prefill_tokens
                running_prefills.append(seq_group)

            running.append(seq_group)

        ######################################################################
        # Phase 3: Add waiting (new) sequence groups to the batch.
        # This routine is nearly-identical to the one in sarathi scheduler
        ######################################################################
        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.
        while self.waiting:
            seq_group = self.waiting[0]

            assert seq_group.num_seqs() == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_prompt_tokens = seq_group.get_seqs()[0].get_len()
            if num_prompt_tokens > self.prompt_limit:
                logger.warning(
                    f"Input prompt ({num_prompt_tokens} tokens) is too long"
                    f" and exceeds limit of {self.prompt_limit}")
                for seq in seq_group.get_seqs():
                    seq.set_status(SequenceStatus.FINISHED_IGNORED)
                ignored_seq_groups.append(seq_group)
                self.waiting.pop(0)
                continue

            # If the sequence group cannot be allocated, stop.
            if not self.block_manager.can_allocate(seq_group):
                # this is different from vllm scheduler
                # even if we cannot allocate this sequence group
                # there might be other sequence groups that can be allocated
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            if len(running) >= self.scheduler_config.max_num_seqs:
                break

            # check if we can fit the prefill in the batch
            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq_group.get_seqs()[0], batch_contains_prefill,
                num_batched_tokens)

            if next_num_prefill_tokens == 0:
                break

            seq_group = self.waiting.pop(0)
            self._allocate(seq_group)
            batch_contains_prefill = True
            num_batched_tokens += next_num_prefill_tokens
            prefill_seq_groups.append(seq_group)
            prefill_prompt_chunk_lens.append(next_num_prefill_tokens)
            running.append(seq_group)
            running_prefills.append(seq_group)

        # make sure that prefills are at the start of the batch, so that we don't violate assumptions
        # made in the original vllm codebase
        self.running = running
        scheduled = prefill_seq_groups + decode_seq_groups
        prompt_chunk_lens = prefill_prompt_chunk_lens + decode_prompt_chunk_lens

        scheduler_outputs = SchedulerOutputs(
            id=self._iteration_id,
            scheduled_seq_groups=scheduled,
            prompt_chunk_lens=prompt_chunk_lens,
            num_batched_prompt_tokens=sum(prompt_chunk_lens),
            num_batched_output_tokens=num_batched_output_tokens,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
        )
        return scheduler_outputs
