import enum
import time
from typing import Dict, List, Optional, Tuple, Union, Iterable, Set

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.block_manager import AllocStatus, BlockSpaceManager
from vllm.anyscale.lora.utils import LoRARequest
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta,
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
        num_chunked_prefill_groups: int,
        num_prompt_groups: int,
        num_batched_tokens: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
        done_seq_group_ids: Set[str],
        num_preallocated_slots: int = 0,
        num_preempted_seqs: int = 0,
        lora_enabled: bool = False,
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.num_chunked_prefill_groups = num_chunked_prefill_groups
        self.num_prompt_groups = num_prompt_groups
        self.num_batched_tokens = num_batched_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        self.done_seq_group_ids = done_seq_group_ids
        self.num_preempted_seqs = num_preempted_seqs

        # The number of preallocated slots per sequence in the KV cache.
        # This is normally zero, but is greater than zero when multiple
        # tokens are generated per scheduling iteration
        self.num_preallocated_slots = num_preallocated_slots
        assert self.num_preallocated_slots >= 0

        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups
        if lora_enabled:
            self.num_loras = len(set(self.lora_requests))
            self._sort_by_lora_ids()

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy
                and not self.done_seq_group_ids)

    def _sort_by_lora_ids(self) -> bool:
        self.scheduled_seq_groups.sort(key=lambda g: (
            g.lora_request.lora_int_id if g.lora_request else 0, g.request_id))

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {g.lora_request for g in self.scheduled_seq_groups}


class SchedulerDecodeOutputs:
    """Outputs of the decoding phase of the scheduler.

    Attributes:
        token_budget: The number of available token slots after scheduling.
        decoding_seq_groups: Selected sequence groups for decoding.
        num_preempted_seqs: The number of preempted sequences.
        blocks_to_swap_in: The blocks to swap in.
        blocks_to_swap_out: The blocks to swap out.
        blocks_to_copy: The blocks to copy.
    """

    def __init__(self, token_budget: int,
                 decoding_seq_groups: List[SequenceGroup],
                 num_preempted_seqs: int, blocks_to_swap_in: Dict[int, int],
                 blocks_to_swap_out: Dict[int, int],
                 blocks_to_copy: Dict[int, List[int]]) -> None:
        self.token_budget = token_budget
        self.decoding_seq_groups = decoding_seq_groups
        self.num_preempted_seqs = num_preempted_seqs
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy

    @staticmethod
    def create_empty() -> "SchedulerDecodeOutputs":
        return SchedulerDecodeOutputs(0, [], 0, {}, {}, {})

    def num_decoding_seqs(self):
        return sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.decoding_seq_groups)

    def curr_loras(self):
        return set(seq_group.lora_int_id
                   for seq_group in self.decoding_seq_groups)


class SchedulePrefillOutputs:
    """Outputs of the prefilling phase of the scheduler.

    Attributes:
        token_budget: The number of available token slots after scheduling.
        num_batched_tokens: The number of batched tokens.
        chunk_prefilling_seq_groups: Selected sequence groups for chunked
            prefilling.
        prompting_seq_groups: Selected sequence groups for prompting.
        ignored_seq_groups: Ignored sequence groups.
    """

    def __init__(
        self,
        token_budget: int,
        num_batched_tokens: int,
        chunk_prefilling_seq_groups: List[SequenceGroup],
        prompting_seq_groups: List[SequenceGroup],
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        self.token_budget = token_budget
        self.num_batched_tokens = num_batched_tokens
        self.chunk_prefilling_seq_groups = chunk_prefilling_seq_groups
        self.prompting_seq_groups = prompting_seq_groups
        self.ignored_seq_groups = ignored_seq_groups

    def num_prompting_groups(self):
        return len(self.prompting_seq_groups)

    def num_chunk_prefilling_groups(self):
        return len(self.chunk_prefilling_seq_groups)

    def num_selected_groups(self):
        return len(self.chunk_prefilling_seq_groups) + len(
            self.prompting_seq_groups)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config
        self.prompt_limit = self.scheduler_config.max_model_len
        self.chunked_prefill_enabled = \
            self.scheduler_config.max_chunked_prefill_len >= 0
        if self.chunked_prefill_enabled:
            self.max_chunked_prefill_len = \
                scheduler_config.max_chunked_prefill_len
            logger.info(
                f"chunked prefill enabled, {self.max_chunked_prefill_len=}"
                f", {self.scheduler_config.max_num_prompt_seqs=}"
                f", { self.scheduler_config.max_num_batched_tokens=}")
            assert not self.lora_enabled, \
                "chunked prefilling is not supported with LoRA"
        else:
            self.max_chunked_prefill_len = 1000_000_000

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window)

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the CHUNKED PREFILLING state.
        self.chunked_prefilling: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []

        # IDs of aborted & finished seq groups before
        # the current scheduling iteration.
        self.done_ids: Set[str] = set()

    @property
    def lora_enabled(self):
        return bool(self.lora_config)

    @property
    def _use_deltas(self):
        return self.scheduler_config.use_deltas

    @property
    def _num_preallocated_slots(self) -> int:
        """The number of slots to preallocate per decode step.

        This is greater than zero when the worker runs more than one step per
        scheduler invocation.
        """
        return self.scheduler_config.num_preallocated_slots_per_step

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens will be generated."""
        return self._num_preallocated_slots + 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        # logger.debug(f"add_seq_group {seq_group.request_id}")
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> int:
        """Returns the number of actually aborted seq groups."""
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        self.done_ids.update(request_ids)
        aborted = 0
        for state_queue in [
                self.waiting, self.running, self.swapped,
                self.chunked_prefilling
        ]:
            # We need to reverse the list as we are removing elements
            # from it as we iterate over it. If we don't do it,
            # indices will get messed up and we will skip over elements.
            for seq_group in reversed(state_queue):
                if seq_group.request_id in request_ids:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    aborted += 1
                    for seq in seq_group.get_seqs():
                        if seq.is_finished():
                            continue
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.free_seq(seq)
                    request_ids.remove(seq_group.request_id)
                    if not request_ids:
                        return aborted
        return aborted

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped \
            or self.chunked_prefilling

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def _schedule_decoding(self, token_budget: int) -> SchedulerDecodeOutputs:
        """Schedule sequence groups for decoding.
        First schedule the sequence groups in the RUNNING state.
        Then schedule the sequence groups in the SWAPPED state.

        Args:
            token_budget: The number of available token slots.
        """
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}
        decoding_seq_groups: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []

        # Fix the current time.
        now = time.monotonic()

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Step 1: Schedule as many decoding requests as possible.
        # If we run out of token budget, stop.
        # If we run out of available slots, try to preempt
        # the lowest-priority sequence groups.
        while self.running:
            if token_budget < self.running[0].num_unfinished_seqs(
            ) * self.num_decoding_tokens_per_seq:
                break
            seq_group = self.running.pop(0)
            while not self._can_append_slots(seq_group):
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
                # logger.debug(f"append slot for {seq_group}")
                # Append new slots to the sequence group.
                self._append_slots(seq_group, blocks_to_copy)
                # logger.debug(f"scheduled r -> r {seq_group.request_id}")
                decoding_seq_groups.append(seq_group)
                token_budget -= seq_group.num_seqs(
                    status=SequenceStatus.RUNNING
                ) * self.num_decoding_tokens_per_seq

        # If any sequence group is preempted, do not swap in any sequence group.
        if preempted:
            return SchedulerDecodeOutputs(token_budget, decoding_seq_groups,
                                          len(preempted), blocks_to_swap_in,
                                          blocks_to_swap_out, blocks_to_copy)

        # Step 2: Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        num_curr_seqs = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in decoding_seq_groups)
        curr_loras = set(
            seq_group.lora_int_id
            for seq_group in self.running) if self.lora_enabled else None

        swapped_indices_to_remove = []
        for i, seq_group in enumerate(self.swapped):
            if token_budget < self.swapped[0].num_unfinished_seqs(
            ) * self.num_decoding_tokens_per_seq:
                break

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                if lora_int_id > 0 and lora_int_id not in curr_loras and (
                        len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    continue

            # If the sequence group cannot be swapped in, stop.
            if not self._can_swap_in(seq_group):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
            if (num_curr_seqs + num_new_seqs >
                    self.scheduler_config.max_num_seqs):
                break

            swapped_indices_to_remove.append(i)
            if lora_int_id > 0:
                curr_loras.add(lora_int_id)
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy)
            num_curr_seqs += num_new_seqs
            # logger.debug(f"scheduled s -> r {seq_group.request_id}")
            decoding_seq_groups.append(seq_group)
            token_budget -= seq_group.num_seqs(
                status=SequenceStatus.RUNNING
            ) * self.num_decoding_tokens_per_seq

        for i in reversed(swapped_indices_to_remove):
            self.swapped.pop(i)

        return SchedulerDecodeOutputs(token_budget, decoding_seq_groups,
                                      len(preempted), blocks_to_swap_in,
                                      blocks_to_swap_out, blocks_to_copy)

    def _chunk_prefill_sequence_group(
            self, seq_group: SequenceGroup, token_budget: int,
            chunk_prefilling_seq_groups: List[SequenceGroup],
            prompting_seq_groups: List[SequenceGroup]) -> int:
        """Chunked prefilling one sequence_group

        Args:
            token_budget: The number of available token slots.
            seq_group: The sequence to be chunk prefilled.
            chunk_prefilling_seq_groups: (output) if the sequence group has more
                to prefill after this step, it will be added to this list.
            prompting_seq_groups: (output) The prompting sequence groups. If
                the sequence group finishes prefilling after this step, it will
                be added to this list.

        Returns:
            num_tokens: The number of tokens to be prefilled from
                the sequence group.
        """
        num_unprefilled_tokens = seq_group.get_num_unprefilled()
        to_advance = min(num_unprefilled_tokens, token_budget,
                         self.max_chunked_prefill_len)

        seq_group.advance_prefill_range(to_advance)

        # If the sequence group is not fully prefilled, put it into the
        # chunked prefilling queue.
        if seq_group.get_num_unprefilled() > 0:
            # logger.debug(f"scheduled p -> p {seq_group.request_id}")
            chunk_prefilling_seq_groups.append(seq_group)
        else:
            # logger.debug(f"scheduled p -> r {seq_group.request_id}")
            prompting_seq_groups.append(seq_group)

        return to_advance

    def _schedule_prefilling(
            self,
            token_budget: int,
            num_curr_seqs: int,
            curr_loras: Optional[Set[int]] = None) -> SchedulePrefillOutputs:
        """Schedule sequence groups for (chunked) prefilling.

        Args:
            token_budget: The number of available token slots.
            num_curr_seqs: The number of sequences already scheduled.
            curr_loras: The set of LoRA IDs already scheduled.

        Returns:
            SchedulePrefillOutputs: The outputs of the prefilling phase.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        num_batched_tokens: int = 0
        prompting_seq_groups: List[SequenceGroup] = []
        chunk_prefilling_seq_groups: List[SequenceGroup] = []
        num_prompting_seqs: int = 0

        # If any request in swapped state, try not schedule any prefilling.
        if self.swapped:
            return SchedulePrefillOutputs(token_budget, num_batched_tokens,
                                          chunk_prefilling_seq_groups,
                                          prompting_seq_groups,
                                          ignored_seq_groups)

        # Step 1: Continue schedule those requests are in chunked prefilling.
        # This is called only if chunked prefilling is enabled.
        while self.chunked_prefilling and token_budget > 0 \
            and num_prompting_seqs < self.scheduler_config.max_num_prompt_seqs:

            if not self.chunked_prefill_enabled:
                assert False, "can't reach here since chunk prefill is disabled"

            seq_group = self.chunked_prefilling.pop(0)

            num_prefilled_tokens = self._chunk_prefill_sequence_group(
                seq_group, token_budget, chunk_prefilling_seq_groups,
                prompting_seq_groups)

            token_budget -= num_prefilled_tokens
            num_batched_tokens += num_prefilled_tokens
            num_curr_seqs += seq_group.get_max_num_running_seqs()
            num_prompting_seqs += 1

        # Step 2: Schedule the waiting requests for (chunked) prefilling.

        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.
        waiting_indices_to_remove = []
        for i, seq_group in enumerate(self.waiting):
            if not (token_budget > 0 and num_prompting_seqs <
                    self.scheduler_config.max_num_prompt_seqs):
                break

            assert seq_group.num_seqs() == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")

            # If the sequence group cannot be allocated, put into the ignored.
            num_prompt_tokens = seq_group.get_seqs()[0].get_len()
            if num_prompt_tokens > self.prompt_limit:
                logger.warning(
                    f"Input prompt ({num_prompt_tokens} tokens) is too long"
                    f" and exceeds limit of {self.prompt_limit}")
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_indices_to_remove.append(i)
                continue

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    f"Input prompt ({num_prompt_tokens} tokens) is too long"
                    f" and exceeds the capacity of block_manager")
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_indices_to_remove.append(i)
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                if lora_int_id > 0 and lora_int_id not in curr_loras and len(
                        curr_loras) >= self.lora_config.max_loras:
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    continue

            # If the number of batched tokens exceeds the limit and
            # chunked prefill is disabled, stop.
            if num_prompt_tokens > token_budget and \
                    not self.chunked_prefill_enabled:
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_curr_seqs + num_new_seqs >
                    self.scheduler_config.max_num_seqs):
                break

            waiting_indices_to_remove.append(i)
            if lora_int_id > 0:
                curr_loras.add(lora_int_id)
            self._allocate(seq_group)

            num_prefilled_tokens = self._chunk_prefill_sequence_group(
                seq_group, token_budget, chunk_prefilling_seq_groups,
                prompting_seq_groups)

            token_budget -= num_prefilled_tokens
            num_batched_tokens += num_prefilled_tokens
            num_curr_seqs += seq_group.get_max_num_running_seqs()
            num_prompting_seqs += 1

        for i in reversed(waiting_indices_to_remove):
            self.waiting.pop(i)

        return SchedulePrefillOutputs(token_budget, num_batched_tokens,
                                      chunk_prefilling_seq_groups,
                                      prompting_seq_groups, ignored_seq_groups)

    def _schedule(self) -> SchedulerOutputs:
        token_budget = self._round_down_by_padding(
            self.scheduler_config.max_num_batched_tokens)

        if self.chunked_prefill_enabled:
            # Chunked prefilling is enabled.
            # We first schedule as many decoding requests as possible,
            # and then schedule chunked prefilling requests.
            decoding_outputs = self._schedule_decoding(token_budget)

            token_budget = self._round_down_by_padding(
                decoding_outputs.token_budget)

            prefilling_outputs = self._schedule_prefilling(
                token_budget, decoding_outputs.num_decoding_seqs(),
                decoding_outputs.curr_loras() if self.lora_enabled else None)
        else:
            # Default behavior
            # First schedule as many prefilling requests as possible,
            # then schedule decoding requests.

            num_curr_seqs = sum(
                seq_group.num_seqs(status=SequenceStatus.RUNNING)
                for seq_group in self.running)
            curr_loras = set(
                seq_group.lora_int_id
                for seq_group in self.running) if self.lora_enabled else None

            prefilling_outputs = self._schedule_prefilling(
                token_budget, num_curr_seqs, curr_loras)

            assert len(prefilling_outputs.chunk_prefilling_seq_groups
                       ) == 0, "Chunked prefill is disabled"

            if len(prefilling_outputs.prompting_seq_groups) > 0:
                decoding_outputs = SchedulerDecodeOutputs.create_empty()
            else:
                decoding_outputs = self._schedule_decoding(token_budget)

        num_batched_tokens = prefilling_outputs.num_batched_tokens + \
            decoding_outputs.num_decoding_seqs() * \
            self.num_decoding_tokens_per_seq

        is_decoding_only = prefilling_outputs.num_selected_groups() == 0
        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=prefilling_outputs.chunk_prefilling_seq_groups
            + prefilling_outputs.prompting_seq_groups +
            decoding_outputs.decoding_seq_groups,
            num_chunked_prefill_groups=prefilling_outputs.
            num_chunk_prefilling_groups(),
            num_prompt_groups=prefilling_outputs.num_selected_groups(),
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=decoding_outputs.blocks_to_swap_in,
            blocks_to_swap_out=decoding_outputs.blocks_to_swap_out,
            blocks_to_copy=decoding_outputs.blocks_to_copy,
            ignored_seq_groups=prefilling_outputs.ignored_seq_groups,
            num_preempted_seqs=decoding_outputs.num_preempted_seqs,
            done_seq_group_ids=self.done_ids.copy(),
            num_preallocated_slots=self._num_preallocated_slots
            if is_decoding_only else 0,
            lora_enabled=self.lora_enabled,
        )

        self.done_ids.clear()

        self.chunked_prefilling = \
            prefilling_outputs.chunk_prefilling_seq_groups + \
                self.chunked_prefilling
        self.running = self.running + \
            prefilling_outputs.prompting_seq_groups + \
            decoding_outputs.decoding_seq_groups
        return scheduler_outputs

    def schedule(
        self
    ) -> Tuple[List[Union[SequenceGroupMetadata, SequenceGroupMetadataDelta]],
               SchedulerOutputs]:
        now_perf_counter = time.perf_counter()
        now = time.time()
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()

        # Create input data structures.
        seq_group_metadata_list: List[Union[SequenceGroupMetadata,
                                            SequenceGroupMetadataDelta]] = []
        for i, seq_group in enumerate(scheduler_outputs.scheduled_seq_groups):
            if seq_group.first_scheduled_time is None:
                seq_group.first_scheduled_time = now
                seq_group.time_in_queue = (now_perf_counter -
                                           seq_group.arrival_time_perf_counter)
            seq_data: Dict[int, List[SequenceData]] = {}
            block_tables: Dict[int, List[int]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)

            is_prompt = i < scheduler_outputs.num_prompt_groups
            is_chunked_prefill = \
                i < scheduler_outputs.num_chunked_prefill_groups

            if not self._use_deltas or is_prompt:
                seq_group_metadata = SequenceGroupMetadata(
                    request_id=seq_group.request_id,
                    is_chunked_prefill=is_chunked_prefill,
                    is_prompt=is_prompt,
                    seq_data=seq_data,
                    sampling_params=seq_group.sampling_params,
                    block_tables=block_tables,
                    lora_request=seq_group.lora_request,
                )
            else:
                seq_group_metadata = SequenceGroupMetadataDelta(
                    request_id=seq_group.request_id,
                    block_tables=block_tables,
                )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        new_running = []
        for seq_group in self.running:
            if seq_group.is_finished():
                self.done_ids.add(seq_group.request_id)
            else:
                new_running.append(seq_group)
        self.running = new_running

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING

    def _append_slots(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            additional_blocks_to_copy = self.block_manager.append_slots(
                seq, self._num_preallocated_slots)

            for src_block, dst_blocks in additional_blocks_to_copy.items():
                if src_block not in blocks_to_copy:
                    blocks_to_copy[src_block] = []
                blocks_to_copy[src_block].extend(dst_blocks)

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
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

    def _can_swap_in(self, seq_group: SequenceGroup) -> bool:
        return self.block_manager.can_swap_in(seq_group,
                                              self._num_preallocated_slots)

    def _can_append_slots(self, seq_group: SequenceGroup) -> bool:
        return self.block_manager.can_append_slots(
            seq_group, self._num_preallocated_slots)

    def _round_down_by_padding(self, x: int) -> int:
        return x // self.scheduler_config.input_padding_size \
            * self.scheduler_config.input_padding_size
