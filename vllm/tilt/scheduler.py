import os
import random
import time
from collections import deque
from collections.abc import Iterable
from collections.abc import Sequence as GenericSequence
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Optional, Union

from vllm.config import CacheConfig, LoRAConfig, ModelConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)
from vllm.tilt.block_manager import TiltBlockSpaceManager
from vllm.utils import Device, PyObjectCache

logger = init_logger(__name__)

# Test-only. If configured, decode is preempted with
# ARTIFICIAL_PREEMPTION_PROB% probability.
ENABLE_ARTIFICIAL_PREEMPT = bool(
    os.getenv("VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT", False))  # noqa
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500


@dataclass
class TiltSchedulerConfig(SchedulerConfig):
    # Maximum number of encoder tokens to be processed in a single iteration.
    max_num_batched_encoder_tokens: int = 4096

    # Maximum number of chunks in a single batch.
    # Protects from OOM in image embedder due adding too many small chunks.
    # Each chunk can have at least one image.
    # Default: 2 * number of full chunks that fit in the batch
    max_chunks_in_batch: int = 0

    # Size of the encoder chunk in TILT Long. The value should be taken from
    # the model config.
    encoder_chunk_size: int = 1024

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_num_batched_encoder_tokens < self.encoder_chunk_size:
            raise ValueError(
                "max_num_batched_encoder_tokens "
                f"({self.max_num_batched_encoder_tokens}) must be greater than "
                "or equal to encoder_chunk_size "
                f"({self.encoder_chunk_size}).")
        if self.max_chunks_in_batch == 0:
            full_chunk_count = (self.max_num_batched_encoder_tokens //
                                self.encoder_chunk_size)
            self.max_chunks_in_batch = 2 * full_chunk_count

    @staticmethod
    def extend_scheduler_config(
        scheduler_config: SchedulerConfig,
        model_config: ModelConfig,
        additional_config: dict[str, Any] | None,
    ) -> "TiltSchedulerConfig":
        init_kwargs = {f.name for f in fields(TiltSchedulerConfig) if f.init}
        additional_kwargs = {}
        if additional_config is not None:
            for k, v in additional_config.items():
                if k in [
                        "max_num_batched_encoder_tokens", "max_chunks_in_batch"
                ]:
                    additional_kwargs[k] = v
        return TiltSchedulerConfig(
            **{
                k: v
                for k, v in asdict(scheduler_config).items()
                if k in init_kwargs
            },
            encoder_chunk_size=model_config.hf_config.chunk_length,
            **additional_kwargs,
        )


@dataclass
class SchedulingBudget:
    """Budget for scheduling TILT.

    Attributes:
        token_budget: Maximum number of decoder tokens in the batch.
        encoder_token_budget: Maximum number of encoder tokens in the batch.
        chunk_budget: Maximum number of chunks in the batch.
        max_num_seqs: Maximum number of sequences in the batch.
    """

    token_budget: int
    encoder_token_budget: int
    chunk_budget: int
    max_num_seqs: int
    _request_ids_num_batched_tokens: set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: set[str] = field(default_factory=set)
    # Number of cached tokens in the batch.
    # Number of actual non-cached tokens in the batch.
    _num_batched_tokens: int = 0
    _num_batched_encoder_tokens: int = 0
    _num_chunks: int = 0
    _num_curr_seqs: int = 0

    def can_schedule(
        self,
        *,
        num_new_tokens: int,
        num_new_encoder_tokens: int,
        num_chunks: int,
        num_new_seqs: int,
    ):
        # We allow num_new_tokens to be 0 when the entire sequence has
        # been cached.
        assert num_new_tokens >= 0
        assert num_new_seqs != 0
        assert num_chunks >= 0
        assert num_new_encoder_tokens >= 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_batched_encoder_tokens + num_new_encoder_tokens
                <= self.encoder_token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs
                and self.num_chunks + num_chunks <= self.chunk_budget)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def remaining_encoder_token_budget(self):
        return self.encoder_token_budget - self.num_batched_encoder_tokens

    def remaining_chunk_budget(self):
        return self.chunk_budget - self.num_chunks

    def remaining_num_seqs_budget(self):
        return self.max_num_seqs - self.num_curr_seqs

    def add_num_batched_tokens(
        self,
        req_id: str,
        num_batched_tokens: int,
        num_batched_encoder_tokens: int,
        num_chunks,
    ):
        if req_id in self._request_ids_num_batched_tokens:
            return
        assert num_batched_tokens >= 0
        assert num_batched_encoder_tokens >= 0
        assert num_chunks >= 0

        self._request_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens
        self._num_batched_encoder_tokens += num_batched_encoder_tokens
        self._num_chunks += num_chunks

    def subtract_num_batched_tokens(
        self,
        req_id: str,
        num_batched_tokens: int,
        num_batched_encoder_tokens: int,
        num_chunks: int,
    ):
        if req_id in self._request_ids_num_batched_tokens:
            self._request_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens
            self._num_batched_encoder_tokens -= num_batched_encoder_tokens
            self._num_chunks -= num_chunks

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            return

        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_batched_encoder_tokens(self):
        return self._num_batched_encoder_tokens

    @property
    def num_chunks(self):
        return self._num_chunks

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs


@dataclass
class ScheduledSequenceGroup:
    # A sequence group that's scheduled.
    seq_group: SequenceGroup
    # The total chunk size (number of tokens) to process for next iteration.
    # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    # chunked, it can be smaller than that.

    # token_chunk_size can be 0 zero if encoder uses chunked prefill and has
    # not completed encoding the input
    token_chunk_size: int

    # encoder_token_chunk_size is the number of encoder_prompt tokens that
    # are available to the decoder cross attention.
    # It is different from budgeted_encoder_token_chunk_size when some tokens
    # are discarded after decoding (e.g. repeated question in TILT Long chunks).
    # > 0 - number of tokens to encode
    # 0 - decoding phase or decoder prefill
    encoder_token_chunk_size: int
    # The amount of tokens provided to the encoder.
    budgeted_encoder_token_chunk_size: int


@dataclass
class SchedulerOutputs:
    """The scheduling decision made from a scheduler."""

    # Scheduled sequence groups.
    scheduled_seq_groups: GenericSequence[ScheduledSequenceGroup]
    # Number of prefill groups scheduled.
    num_prefill_groups: int
    # Total number of batched tokens.
    num_batched_tokens: int
    num_encoder_batched_tokens: int
    # Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: list[tuple[int, int]]
    # Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: list[tuple[int, int]]
    # Blocks to copy. Source to dest block.
    blocks_to_copy: list[tuple[int, int]]
    # Sequence groups that are going to be ignored.
    ignored_seq_groups: list[SequenceGroup]
    # The number of slots for lookahead decoding.
    # num_lookahead_slots: int
    # The number of requests in the running queue
    running_queue_size: int
    preempted: int
    num_lookahead_slots: int = 0

    def __post_init__(self):
        # Swap in and swap out should never happen at the same time.
        assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)


def scheduled_seq_group_builder():
    return ScheduledSequenceGroup(
        SequenceGroup.__new__(SequenceGroup),
        token_chunk_size=0,
        encoder_token_chunk_size=0,
        budgeted_encoder_token_chunk_size=0,
    )


class Scheduler:
    """Scheduler for TILT.

    Scheduling algorithm:
        1. First schedule all RUNNING sequence groups.
        2. If KV cache is full, preempt the newest RUNNING sequence group.
        3. If there is budget left, schedule WAITING sequence groups.

    Budget is allocated in following order:
        1. If the limit on the number of sequences is reached, stop.
        2. If there are uncomputed encoder tokens, try to schedule as many
        encoder chunks as possible. If either chunk or encoder token limit is
        reached, stop.
        3. If all encoder tokens have been computed or will be computed in this
        batch, schedule decoder tokens up to the limit.

    There is no preemption due to priority.
    """

    def __init__(
        self,
        scheduler_config: TiltSchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        assert (not scheduler_config.chunked_prefill_enabled
                ), "Chunked prefill not supported yet"
        assert (scheduler_config.num_lookahead_slots == 0
                ), "Speculative decoding not supported yet"
        assert (not cache_config.enable_prefix_caching
                ), "Prefix caching not supported yet"
        assert lora_config is None, "LoRA not supported yet"
        assert output_proc_callback is None, "Async output processing not supported yet"

        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.lora_config = lora_config

        num_gpu_blocks = cache_config.num_gpu_blocks
        if num_gpu_blocks:
            num_gpu_blocks //= pipeline_parallel_size

        num_cpu_blocks = cache_config.num_cpu_blocks
        if num_cpu_blocks:
            num_cpu_blocks //= pipeline_parallel_size

        # Create the block space manager.
        self.block_manager = TiltBlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=None,
            enable_caching=False,
        )

        # Sequence groups in the WAITING state.
        # Contain new prefill or preempted requests.
        self.waiting: deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        self.running: deque[SequenceGroup] = deque()
        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0

        # Sequence groups in the SWAPPED state.
        # Not supported. Attribute kept for API compat.
        self.swapped: deque[SequenceGroup] = deque()

        # The following field is test-only. It is used to inject artificial
        # preemption.
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                       if self.enable_artificial_preemption
                                       else 0)
        self.num_cumulative_preemption: int = 0

        # Used to cache python objects
        self._scheduled_seq_group_cache = PyObjectCache(
            scheduled_seq_group_builder)

    @property
    def lora_enabled(self) -> bool:
        return False

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def _add_seq_group_to_running(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the running queue.
        # Only for testing purposes.
        self.running.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running]:
            aborted_groups: list[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity.
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

                if aborted_group.is_encoder_decoder():
                    self.block_manager.free_cross(aborted_group)

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) != 0 or len(self.running) != 0

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_manager.get_prefix_cache_hit_rate(device)

    def reset_prefix_cache(self) -> bool:
        return self.block_manager.reset_prefix_cache()

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running)

    def get_and_reset_finished_requests_ids(self) -> list[str]:
        return []

    def _can_append_slots(self, seq_group: SequenceGroup) -> bool:
        # It is True only for testing case to trigger artificial preemption.
        if (self.enable_artificial_preemption
                and random.uniform(0, 1) < ARTIFICIAL_PREEMPTION_PROB
                and self.artificial_preempt_cnt > 0):
            self.artificial_preempt_cnt -= 1
            return False

        return self.block_manager.can_append_slots(seq_group=seq_group,
                                                   num_lookahead_slots=0)

    def schedule(
            self
    ) -> tuple[list[SequenceGroupMetadata], SchedulerOutputs, bool]:
        scheduler_start_time = time.perf_counter()

        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            encoder_token_budget=self.scheduler_config.
            max_num_batched_encoder_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
            chunk_budget=self.scheduler_config.max_chunks_in_batch,
        )

        running_seq_groups = []
        preempted_seq_groups = []
        ignored_seq_groups = []
        sched_running_seq_groups = []
        blocks_to_copy = []

        running_queue = self.running
        while running_queue:
            seq_group = running_queue[0]
            if (budget.remaining_num_seqs_budget() == 0
                    and seq_group.request_id
                    not in budget._request_ids_num_curr_seqs):
                continue

            (
                num_running_tokens,
                num_running_encoder_tokens,
                num_running_budgeted_encoder_tokens,
                num_chunks,
            ) = self._get_num_new_tokens(seq_group, SequenceStatus.RUNNING,
                                         budget)
            if num_running_tokens == 0 and num_running_encoder_tokens == 0:
                # No budget => Stop
                break

            running_queue.popleft()

            # NOTE(woosuk): Preemption happens only when there is no available
            # slot to keep all the sequence groups in the RUNNING state.
            while not seq_group.is_prefill() and not self._can_append_slots(
                    seq_group):
                budget.subtract_num_batched_tokens(
                    req_id=seq_group.request_id,
                    num_batched_tokens=num_running_tokens,
                    num_batched_encoder_tokens=
                    num_running_budgeted_encoder_tokens,
                    num_chunks=num_chunks,
                )
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                # Determine victim sequence
                cont_loop = True
                if running_queue:
                    # Preempt the lowest-priority sequence group.
                    victim_seq_group = running_queue.pop()
                else:
                    # No other sequence group can be preempted.
                    # Preempt the current sequence group.
                    # Note: This is also where we stop this loop
                    # (since there is nothing else to preempt)
                    victim_seq_group = seq_group
                    cont_loop = False

                # Do preemption
                self._preempt_by_recompute(victim_seq_group)
                preempted_seq_groups.append(victim_seq_group)

                if not cont_loop:
                    break
            else:
                scheduled_seq_group: ScheduledSequenceGroup = (
                    self._scheduled_seq_group_cache.get_object())
                scheduled_seq_group.seq_group = seq_group
                scheduled_seq_group.token_chunk_size = num_running_tokens
                scheduled_seq_group.encoder_token_chunk_size = (
                    num_running_encoder_tokens)
                scheduled_seq_group.budgeted_encoder_token_chunk_size = (
                    num_running_budgeted_encoder_tokens)
                sched_running_seq_groups.append(scheduled_seq_group)
                running_seq_groups.append(seq_group)

                if scheduled_seq_group.token_chunk_size > 0:
                    self._append_slots(seq_group, blocks_to_copy)

                budget.add_num_batched_tokens(
                    req_id=seq_group.request_id,
                    num_batched_tokens=num_running_tokens,
                    num_batched_encoder_tokens=
                    num_running_budgeted_encoder_tokens,
                    num_chunks=num_chunks,
                )
                budget.add_num_seqs(seq_group.request_id, 1)

        new_prompts = 0
        waiting_queue = self.waiting
        leftover_waiting_sequences: deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            if budget.remaining_num_seqs_budget() == 0:
                break

            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            (
                num_new_tokens,
                num_new_encoder_tokens,
                num_new_budgeted_encoder_tokens,
                num_chunks,
            ) = self._get_num_new_tokens(seq_group, SequenceStatus.WAITING,
                                         budget)

            num_prompt_tokens = waiting_seqs[0].get_len()
            assert num_new_tokens == num_prompt_tokens or num_new_tokens == 0

            prompt_limit = min(
                self.scheduler_config.max_model_len,
                self.scheduler_config.max_num_batched_tokens,
            )
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d",
                    num_new_tokens,
                    prompt_limit,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots: int = 0
            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens,
                    num_lookahead_slots,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0 and num_new_encoder_tokens
                    == 0) or not budget.can_schedule(
                        num_new_tokens=num_new_tokens,
                        num_new_seqs=num_new_seqs,
                        num_new_encoder_tokens=num_new_budgeted_encoder_tokens,
                        num_chunks=num_chunks,
                    ):
                break

            # Can schedule this request.
            waiting_queue.popleft()
            self.block_manager.allocate(seq_group)
            for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
                seq.status = SequenceStatus.RUNNING

            seq_group.init_multi_step_from_lookahead_slots(
                num_lookahead_slots,
                num_scheduler_steps=self.scheduler_config.num_scheduler_steps,
                is_multi_step=self.scheduler_config.is_multi_step,
                enable_chunking=False,
            )

            new_prompts += 1
            sched_running_seq_groups.append(
                ScheduledSequenceGroup(
                    seq_group=seq_group,
                    token_chunk_size=num_new_tokens,
                    encoder_token_chunk_size=num_new_encoder_tokens,
                    budgeted_encoder_token_chunk_size=
                    num_new_budgeted_encoder_tokens,
                ))
            running_seq_groups.append(seq_group)
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens,
                num_batched_encoder_tokens=num_new_budgeted_encoder_tokens,
                num_chunks=num_chunks,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if new_prompts > 0:
            self.prev_prompt = True

        self._scheduled_seq_group_cache.reset()

        assert budget.num_batched_tokens <= self.scheduler_config.max_num_batched_tokens
        assert (budget.num_batched_encoder_tokens
                <= self.scheduler_config.max_num_batched_encoder_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        self.waiting.extendleft(preempted_seq_groups)
        self.running.extend(running_seq_groups)

        sched_running_seq_groups = list(
            sorted(
                sched_running_seq_groups,
                key=lambda x: x.seq_group.is_prefill(),
                reverse=True,
            ))

        num_prefill_groups = 0
        for sched_seq_group in sched_running_seq_groups:
            if sched_seq_group.seq_group.is_prefill():
                num_prefill_groups += 1

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=sched_running_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens,
            num_encoder_batched_tokens=budget.num_batched_encoder_tokens,
            # NOTE: We are not implementing swapping, as V1 does not support it.
            #       We would lose it anyways when migrating to V1.
            blocks_to_swap_in=[],
            blocks_to_swap_out=[],
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
            running_queue_size=len(self.running),
            preempted=len(preempted_seq_groups),
        )
        now = time.time()

        if not self.cache_config.enable_prefix_caching:
            common_computed_block_nums = []

        # Create input data structures.
        seq_group_metadata_list: list[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            # seq_id -> SequenceData
            seq_data: dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: dict[int, list[int]] = {}

            encoder_seq_data = None
            cross_block_table = None
            encoder_prefix_seq_data = None
            encoder_prefix_multi_modal_data = None
            if seq_group.is_encoder_decoder():
                # Encoder associated with SequenceGroup
                encoder_seq = seq_group.get_encoder_seq()
                assert encoder_seq is not None
                encoder_seq_data = encoder_seq.data

                if seq_group.encoder_prefix_seq is not None:
                    encoder_prefix_seq_data = seq_group.encoder_prefix_seq.data
                    encoder_prefix_multi_modal_data = (
                        seq_group.encoder_prefix_seq.multi_modal_data)

                # Block table for cross-attention
                # Also managed at SequenceGroup level
                cross_block_table = self.block_manager.get_cross_block_table(
                    seq_group)

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            if self.cache_config.enable_prefix_caching:
                common_computed_block_nums = (
                    self.block_manager.get_common_computed_block_ids(
                        seq_group.get_seqs(status=SequenceStatus.RUNNING)))

            do_sample = True
            is_prompt = seq_group.is_prefill()
            # We should send the metadata to workers when the first prefill
            # is sent. Subsequent requests could be chunked prefill or decode.
            if is_prompt:
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                num_computed_tokens = seqs[0].data.get_num_computed_tokens()
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if token_chunk_size + num_computed_tokens < seqs[
                        0].data.get_len():
                    do_sample = False

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=is_prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                do_sample=do_sample,
                pooling_params=seq_group.pooling_params,
                token_chunk_size=token_chunk_size,
                encoder_token_chunk_size=scheduled_seq_group.
                encoder_token_chunk_size,
                # lora_request=seq_group.lora_request,
                computed_block_nums=common_computed_block_nums,
                encoder_seq_data=encoder_seq_data,
                encoder_prefix_seq_data=encoder_prefix_seq_data,
                encoder_prefix_multi_modal_data=encoder_prefix_multi_modal_data,
                cross_block_table=cross_block_table,
                state=seq_group.state,
                token_type_ids=seq_group.token_type_ids,
                # `multi_modal_data` will only be present for the 1st comm
                # between engine and worker.
                # the subsequent comms can still use delta, but
                # `multi_modal_data` will be None.
                multi_modal_data=(seq_group.multi_modal_data
                                  if scheduler_outputs.num_prefill_groups > 0
                                  else None),
                multi_modal_placeholders=(seq_group.multi_modal_placeholders if
                                          scheduler_outputs.num_prefill_groups
                                          > 0 else None),
                mm_processor_kwargs=seq_group.mm_processor_kwargs,
                prompt_adapter_request=seq_group.prompt_adapter_request,
            )
            seq_group_metadata_list.append(seq_group_metadata)

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group,
                scheduled_seq_group.token_chunk_size)

        scheduler_time = time.perf_counter() - scheduler_start_time
        # Add this to scheduler time to all the sequences that are currently
        # running. This will help estimate if the scheduler is a significant
        # component in the e2e latency.
        for seq_group in self.running:
            if seq_group is not None and seq_group.metrics is not None:
                if seq_group.metrics.scheduler_time is not None:
                    seq_group.metrics.scheduler_time += scheduler_time
                else:
                    seq_group.metrics.scheduler_time = scheduler_time

        # Return results
        return (seq_group_metadata_list, scheduler_outputs, False)

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        self.block_manager.free(seq)

    def _free_finished_seqs(self, seq_group: SequenceGroup) -> None:
        """Free finished seqs in a sequence group."""
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                self.free_seq(seq)

    def _free_finished_seq_group(self, seq_group: SequenceGroup) -> None:
        if seq_group.is_finished():
            # Free cross-attention block table, if it exists
            if seq_group.is_encoder_decoder():
                self.block_manager.free_cross(seq_group)

        # Free finished seqs
        self._free_finished_seqs(seq_group)

    def free_finished_seq_groups(self) -> None:
        remaining: deque[SequenceGroup] = deque()
        for seq_group in self.running:
            self._free_finished_seq_group(seq_group)
            if not seq_group.is_finished():
                remaining.append(seq_group)

        self.running = remaining

    def _append_slots(self, seq_group: SequenceGroup,
                      blocks_to_copy: list[tuple[int, int]]) -> None:
        """Appends new slots to the sequences in the given sequence group.

        Args:
            seq_group (SequenceGroup): The sequence group containing the
                sequences to append slots to.
            blocks_to_copy (List[Tuple[int, int]]): A list of tuple of two
                ints, the first int is the source block index, and the second
                int is the destination block index. This list is updated with
                the new source and destination block indices for the appended
                slots.
        """
        seq_group.init_multi_step_from_lookahead_slots(
            num_lookahead_slots=0,
            num_scheduler_steps=self.scheduler_config.num_scheduler_steps,
            is_multi_step=self.scheduler_config.is_multi_step,
            enable_chunking=False,
        )

        seq_status: Optional[SequenceStatus] = SequenceStatus.RUNNING
        for seq in seq_group.get_seqs(status=seq_status):
            cows = self.block_manager.append_slots(seq, num_lookahead_slots=0)
            if len(cows) > 0:
                blocks_to_copy.extend(cows)

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()
        if seq_group.is_encoder_decoder():
            self.block_manager.free_cross(seq_group)
            seq_group.encoder_seq.reset_state_for_recompute()
            if seq_group.encoder_prefix_seq is not None:
                seq_group.encoder_prefix_seq.reset_state_for_recompute()

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.waiting])
            passed_delay = (now - earliest_arrival_time) > (
                self.scheduler_config.delay_factor *
                self.last_prompt_latency) or not self.running
        else:
            passed_delay = True
        return passed_delay

    def _get_num_new_tokens(
        self,
        seq_group: SequenceGroup,
        status: SequenceStatus,
        budget: SchedulingBudget,
    ) -> tuple[int, int, int, int]:
        num_new_tokens = 0

        # Decoder tokens
        seqs = seq_group.get_seqs(status=status)
        for seq in seqs:
            if not seq.is_prefill():
                num_new_tokens += 1
                continue

            num_computed_tokens_seq = seq.get_num_computed_tokens()
            all_num_new_tokens_seq = seq.get_len() - num_computed_tokens_seq
            num_new_tokens += all_num_new_tokens_seq

        # Encoder tokens
        encoder_seq = seq_group.encoder_seq
        encoder_prefix_seq = seq_group.encoder_prefix_seq

        chunk_size = self.scheduler_config.encoder_chunk_size
        num_encoder_prefix_tokens = (encoder_prefix_seq.get_len()
                                     if encoder_prefix_seq is not None else 0)
        usable_chunk_size = chunk_size - num_encoder_prefix_tokens
        if encoder_seq is not None:
            assert usable_chunk_size > 0
            encoder_token_budget = budget.remaining_encoder_token_budget()
            encoder_chunk_budget = encoder_token_budget // chunk_size

            remaining_num_new_encoder_tokens = (
                encoder_seq.get_len() - encoder_seq.get_num_computed_tokens())
            remaining_num_encoder_chunks = (remaining_num_new_encoder_tokens //
                                            usable_chunk_size)

            # Calculate the number of new encoder chunks to process
            num_chunks = min(
                encoder_chunk_budget,
                remaining_num_encoder_chunks,
                budget.remaining_chunk_budget(),
            )
            num_new_encoder_tokens = num_chunks * usable_chunk_size
            num_new_budgeted_encoder_tokens = num_chunks * chunk_size

            # Update remaining tokens and budget
            remaining_num_new_encoder_tokens -= num_new_encoder_tokens
            encoder_token_budget -= num_new_budgeted_encoder_tokens

            if (0 < remaining_num_new_encoder_tokens < usable_chunk_size
                    and remaining_num_new_encoder_tokens +
                    num_encoder_prefix_tokens < encoder_token_budget
                    and budget.remaining_chunk_budget() > num_chunks):
                # There is a final incomplete chunk remaining and it is small
                # enough to fit in the remaining budget
                num_new_encoder_tokens += remaining_num_new_encoder_tokens
                num_new_budgeted_encoder_tokens += (
                    remaining_num_new_encoder_tokens +
                    num_encoder_prefix_tokens)
                remaining_num_new_encoder_tokens = 0
                num_chunks += 1

            if remaining_num_new_encoder_tokens > 0:
                # Encoding has not finished. Decoder has to wait.
                num_new_tokens = 0
        else:
            num_new_encoder_tokens = 0
            num_new_budgeted_encoder_tokens = 0
            num_chunks = 0

        return (
            num_new_tokens,
            num_new_encoder_tokens,
            num_new_budgeted_encoder_tokens,
            num_chunks,
        )
