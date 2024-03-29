import time
from typing import List
from unittest.mock import MagicMock

import pytest  # noqa

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus
from vllm.core.scheduler import Scheduler
from vllm.lora.request import LoRARequest
from vllm.sequence import Logprob, SequenceGroup

from .utils import create_dummy_prompt


def get_sequence_groups(scheduler_output):
    return [s.seq_group for s in scheduler_output.scheduled_seq_groups]


def test_scheduler_add_seq_group():
    block_size = 4
    scheduler_config = SchedulerConfig(100, 64, 1)
    cache_config = CacheConfig(block_size, 1.0, 1, cache_dtype="auto")
    cache_config.num_cpu_blocks = 4
    cache_config.num_gpu_blocks = 4
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add seq group to scheduler.
    num_seq_group = 4
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), block_size)
        scheduler.add_seq_group(seq_group)
        assert scheduler.get_num_unfinished_seq_groups() == i + 1


def test_scheduler_abort_seq_group():
    block_size = 4
    scheduler_config = SchedulerConfig(100, 64, 1)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 4
    cache_config.num_gpu_blocks = 4
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add multiple seq groups to scheduler.
    num_seq_group = 4
    request_ids = set()
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), block_size)
        scheduler.add_seq_group(seq_group)
        request_ids.add(str(i))

    # Abort all added seq groups.
    assert scheduler.get_num_unfinished_seq_groups() == num_seq_group
    scheduler.abort_seq_group(request_ids)
    assert scheduler.get_num_unfinished_seq_groups() == 0


def test_scheduler_schedule_simple():
    block_size = 4
    num_seq_group = 4
    max_model_len = 16
    scheduler_config = SchedulerConfig(64, num_seq_group, max_model_len)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: List[SequenceGroup] = []

    # Add seq groups to scheduler.
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

    # Schedule seq groups prompts.
    num_tokens = block_size * num_seq_group
    seq_group_meta, out = scheduler.schedule()
    assert set(get_sequence_groups(out)) == set(running)
    assert out.num_batched_tokens == num_tokens
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == num_seq_group

    # Schedule seq groups generation.
    seq_group_meta, out = scheduler.schedule()
    assert set(get_sequence_groups(out)) == set(running)
    assert out.num_batched_tokens == num_seq_group
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == num_seq_group


def test_scheduler_schedule_preempt_abort():
    block_size = 4
    max_model_len = 16
    scheduler_config = SchedulerConfig(64, 2, max_model_len)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 2
    cache_config.num_gpu_blocks = 2
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add seq groups to scheduler.
    seq_a, seq_group_a = create_dummy_prompt("1", block_size)
    seq_b, seq_group_b = create_dummy_prompt("2", block_size)
    scheduler.add_seq_group(seq_group_a)
    scheduler.add_seq_group(seq_group_b)

    # Schedule seq groups prompts.
    seq_group_meta, out = scheduler.schedule()
    assert get_sequence_groups(out) == [seq_group_a, seq_group_b]
    assert out.num_batched_tokens == block_size * 2  # seq_a and seq_b
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 2
    assert scheduler.get_num_unfinished_seq_groups() == 2

    # Append "generated" tokens, allowing the sequence to mark prompt tokens as
    # processed.
    token_id = 0
    seq_a.append_token_id(token_id, {token_id: Logprob(0.0)})
    seq_b.append_token_id(token_id, {token_id: Logprob(0.0)})

    # Schedule seq groups generation and preempt seq group b.
    seq_group_meta, out = scheduler.schedule()
    assert get_sequence_groups(out) == [seq_group_a]
    assert out.num_batched_tokens == 1
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 1
    assert scheduler.get_num_unfinished_seq_groups() == 2

    # Abort seq group a. Re-schedule seq group b prompt with recomputation.
    scheduler.abort_seq_group("1")
    seq_group_meta, out = scheduler.schedule()
    assert get_sequence_groups(out) == [seq_group_b]
    assert out.num_batched_tokens == 5  # 4 prompt + 1 generation.
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 1
    assert scheduler.get_num_unfinished_seq_groups() == 1


def test_scheduler_max_seqs():
    block_size = 4
    num_seq_group = 4
    max_seq_group = 2
    max_model_len = 16
    scheduler_config = SchedulerConfig(64, max_seq_group, max_model_len)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)

    all_seq_groups: List[SequenceGroup] = []
    # Add seq groups to scheduler.
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=block_size)
        all_seq_groups.append(seq_group)

    # Append 1 seq group
    scheduler.add_seq_group(all_seq_groups[0])

    # Schedule seq groups prompts.
    _, out = scheduler.schedule()
    assert set(get_sequence_groups(out)) == set([all_seq_groups[0]])

    # Schedule seq groups generation.
    _, out = scheduler.schedule()
    assert set(get_sequence_groups(out)) == set([all_seq_groups[0]])

    # Append 2 more seq group
    scheduler.add_seq_group(all_seq_groups[1])
    scheduler.add_seq_group(all_seq_groups[2])

    # Schedule seq groups prompts.
    # Only 1 seq group should be scheduled since max_seq_group is 2
    # and one is prompting.
    _, out = scheduler.schedule()
    assert set(get_sequence_groups(out)) == set([all_seq_groups[1]])


def test_scheduler_delay_factor():
    block_size = 4
    scheduler_config = SchedulerConfig(100, 64, 16, delay_factor=0.5)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # schedule first prompt
    _, seq_group = create_dummy_prompt("0", prompt_length=block_size)
    scheduler.add_seq_group(seq_group)
    seq_group_meta, out = scheduler.schedule()
    assert out.num_prefill_groups > 0
    assert seq_group_meta[0].request_id == '0'

    # wait for a second before scheduling next prompt
    time.sleep(1)
    _, seq_group = create_dummy_prompt("1", prompt_length=block_size)
    scheduler.add_seq_group(seq_group)

    # second prompt should *not* be scheduled
    seq_group_meta, out = scheduler.schedule()
    assert out.num_prefill_groups == 0
    assert seq_group_meta[0].request_id == '0'

    # wait for more than 0.5 second and try again
    time.sleep(0.6)
    seq_group_meta, out = scheduler.schedule()
    assert out.num_prefill_groups > 0
    assert seq_group_meta[0].request_id == '1'


def initialize_scheduler(*,
                         max_num_seq=1000,
                         max_token_budget=1000,
                         max_model_len=1000,
                         lora_config=None):
    block_size = 4
    scheduler_config = SchedulerConfig(max_token_budget, max_num_seq,
                                       max_model_len)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, lora_config)
    return scheduler


def test_prefill_schedule():
    """
    Test prompt longer than max_prompt_len is aborted.
    """
    scheduler = initialize_scheduler(max_model_len=30)
    _, seq_group = create_dummy_prompt(0, prompt_length=60)
    scheduler.add_seq_group(seq_group)
    output = scheduler._schedule_prefills(100)
    assert len(output.ignored_seq_groups) == 1
    assert len(output.seq_groups) == 0
    assert output.num_batched_tokens == 0
    """
    Test token budget respected.
    """
    scheduler = initialize_scheduler()
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)

    # 0 token budget == nothing is scheduled.
    output = scheduler._schedule_prefills(0)
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 0
    assert output.num_batched_tokens == 0

    # 60 token budget == 1 request scheduled.
    output = scheduler._schedule_prefills(60)
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 1
    assert output.num_batched_tokens == 60
    """
    Test max seq respected.
    """
    scheduler = initialize_scheduler(max_num_seq=2)
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)
    output = scheduler._schedule_prefills(1000)
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 2
    assert output.num_batched_tokens == 120
    """
    Test max lora is respected and prioritized.
    """
    lora_config = LoRAConfig(max_lora_rank=8, max_loras=1)
    scheduler = initialize_scheduler(lora_config=lora_config)
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           lora_request=LoRARequest(
                                               lora_name=str(i),
                                               lora_int_id=i + 1,
                                               lora_local_path="abc"))
        scheduler.add_seq_group(seq_group)
    # Add two more requests to verify lora is prioritized.
    # 0: Lora, 1: Lora, 2: regular, 3: regular
    # In the first iteration, index 0, 2 is scheduled.
    # If a request is not scheduled because it hits max lora, it is
    # prioritized. Verify that.
    for i in range(2, 4):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)
    # Schedule 2 requests (0 and 2)
    output = scheduler._schedule_prefills(120)
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 2
    assert output.num_batched_tokens == 120
    # The second lora request should be scheduled first.
    output = scheduler._schedule_prefills(60)
    assert len(output.seq_groups) == 1
    assert output.seq_groups[0].seq_group.request_id == "1"
    """
    Test sequence cannot be scheduled due to block manager has no capacity.
    """
    scheduler = initialize_scheduler()
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)
    scheduler.block_manager.can_allocate = MagicMock()
    scheduler.block_manager.can_allocate.return_value = AllocStatus.LATER
    output = scheduler._schedule_prefills(1000)
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 0
    assert output.num_batched_tokens == 0

    scheduler = initialize_scheduler()
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)
    scheduler.block_manager.can_allocate = MagicMock()
    scheduler.block_manager.can_allocate.return_value = AllocStatus.NEVER
    output = scheduler._schedule_prefills(1000)
    assert len(output.ignored_seq_groups) == 3
    assert len(output.seq_groups) == 0
    assert output.num_batched_tokens == 0


def test_decode_schedule():
    """
    Test token budget respected.
    """
    scheduler = initialize_scheduler()
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)
    _, out = scheduler.schedule()
    # prefill scheduled now.
    assert len(out.scheduled_seq_groups) == 3

    # 0 token budget == nothing is scheduled.
    output = scheduler._schedule_decodes(0)
    assert len(output.seq_groups) == 0
    assert output.num_preempted_seqs == 0
    assert output.num_batched_tokens == 0

    # 2 token budget == 2 decodes scheduled
    output = scheduler._schedule_decodes(2)
    assert len(output.seq_groups) == 2
    assert output.num_preempted_seqs == 0
    assert output.num_batched_tokens == 2
    assert output.blocks_to_swap_out == {}
    assert output.blocks_to_copy == {}

    # NOTE: max_num_seqs not respected with decodes.
    """
    Test decodes cannot be scheduled and preempted.
    """
    scheduler = initialize_scheduler()
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)
    _, out = scheduler.schedule()
    # prefill scheduled now.
    assert len(out.scheduled_seq_groups) == 3

    scheduler.block_manager.can_append_slot = MagicMock()

    def cannot_append_second_group(seq_group):
        return seq_group.request_id != "1"

    scheduler.block_manager.can_append_slot.side_effect = (
        cannot_append_second_group)

    # 1 cannot be scheduled, and the lowest priority (request 2)
    # should be preempted. 1 will also be preempted.
    output = scheduler._schedule_decodes(100)
    assert len(output.seq_groups) == 1
    assert output.seq_groups[0].seq_group.request_id == "0"
    assert output.num_preempted_seqs == 2
    assert output.num_batched_tokens == 1
    # Both should be preempted, not swapped.
    assert output.blocks_to_swap_out == {}
    # Nothing is copied.
    assert output.blocks_to_copy == {}

    # Since both are preempted, prefill scheduling should schedule them.
    output = scheduler._schedule_prefills(1000)
    assert len(output.seq_groups) == 2
    """
    Test best_of > 1 swap out blocks
    """
    scheduler = initialize_scheduler()
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60, best_of=2)
        scheduler.add_seq_group(seq_group)
    _, out = scheduler.schedule()
    # prefill scheduled now.
    assert len(out.scheduled_seq_groups) == 3

    # The last request should be swapped out.
    scheduler.block_manager.can_append_slot = MagicMock()

    def cannot_append_second_group(seq_group):
        return seq_group.request_id != "2"

    scheduler.block_manager.can_append_slot.side_effect = (
        cannot_append_second_group)
    scheduler.block_manager.swap_out = MagicMock()
    expected_swap_mapping = {"5": "7"}
    scheduler.block_manager.swap_out.return_value = expected_swap_mapping

    output = scheduler._schedule_decodes(100)
    assert len(output.seq_groups) == 2
    assert output.seq_groups[0].seq_group.request_id == "0"
    assert output.seq_groups[1].seq_group.request_id == "1"
    assert output.num_preempted_seqs == 1
    assert output.num_batched_tokens == 2
    # Both should be preempted, not swapped.
    assert output.blocks_to_swap_out == expected_swap_mapping
    # Nothing is copied.
    assert output.blocks_to_copy == {}
    """
    Verify blocks_to_copy is updated.
    """
    scheduler = initialize_scheduler()
    _, seq_group = create_dummy_prompt(str(i), prompt_length=60, best_of=2)
    scheduler.add_seq_group(seq_group)
    _, out = scheduler.schedule()
    # prefill scheduled now.
    assert len(out.scheduled_seq_groups) == 1

    # The last request should be swapped out.
    scheduler.block_manager.append_slot = MagicMock()
    scheduler.block_manager.append_slot.return_value = (
        2,
        3,
    )

    output = scheduler._schedule_decodes(100)
    assert len(output.seq_groups) == 1
    assert output.num_preempted_seqs == 0
    assert output.num_batched_tokens == 1
    # Nothing is preempted.
    assert output.blocks_to_swap_out == {}
    # Since append_slot returns the source -> dist mapping, it should
    # applied.
    assert output.blocks_to_copy == {2: [3]}


def test_swapped_schedule():
    scheduler = initialize_scheduler(max_num_seq=6)
    # best_of=2 * 3 == 6 sequences.
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60, best_of=2)
        scheduler.add_seq_group(seq_group)
    _, out = scheduler.schedule()
    # prefill scheduled now.
    assert len(out.scheduled_seq_groups) == 3

    # The last request should be swapped out.
    scheduler.block_manager.can_append_slot = MagicMock()

    def cannot_append_second_group(seq_group):
        return seq_group.request_id != "2"

    scheduler.block_manager.can_append_slot.side_effect = (
        cannot_append_second_group)

    _, out = scheduler.schedule()
    assert len(out.scheduled_seq_groups) == 2
    assert out.num_batched_tokens == 2
    assert out.blocks_to_swap_out != {}
    assert out.blocks_to_swap_in == {}

    # Add 1 more task. Swap should be prioritized over prefill.
    _, seq_group = create_dummy_prompt(str(i), prompt_length=60, best_of=2)
    scheduler.add_seq_group(seq_group)
    _, out = scheduler.schedule()
    assert len(out.scheduled_seq_groups) == 3
    # 3 decodes. It is swapped in.
    assert out.num_batched_tokens == 3
    assert out.blocks_to_swap_in != {}
    assert out.blocks_to_swap_out == {}
