import time
from typing import List

import pytest  # noqa

from vllm import SamplingParams
from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.sequence import Logprob, Sequence, SequenceGroup

from .utils import create_dummy_prompt


def test_scheduler_add_seq_group():
    block_size = 4
    scheduler_config = SchedulerConfig(100, 64, 1)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
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

    # Add seq groups to scheduler.
    running: List[SequenceGroup] = []
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

    # Schedule seq groups prompts.
    num_tokens = block_size * num_seq_group
    seq_group_meta, out = scheduler.schedule()
    assert set(out.scheduled_seq_groups) == set(running)
    assert out.num_batched_tokens == num_tokens
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == num_seq_group

    # Schedule seq groups generation.
    seq_group_meta, out = scheduler.schedule()
    assert set(out.scheduled_seq_groups) == set(running)
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
    assert out.scheduled_seq_groups == [seq_group_a, seq_group_b]
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
    assert out.scheduled_seq_groups == [seq_group_a]
    assert out.num_batched_tokens == 1
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 1
    assert scheduler.get_num_unfinished_seq_groups() == 2

    # Abort seq group a. Re-schedule seq group b prompt with recomputation.
    scheduler.abort_seq_group("1")
    seq_group_meta, out = scheduler.schedule()
    assert out.scheduled_seq_groups == [seq_group_b]
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
    assert set(out.scheduled_seq_groups) == set([all_seq_groups[0]])

    # Schedule seq groups generation.
    _, out = scheduler.schedule()
    assert set(out.scheduled_seq_groups) == set([all_seq_groups[0]])

    # Append 2 more seq group
    scheduler.add_seq_group(all_seq_groups[1])
    scheduler.add_seq_group(all_seq_groups[2])

    # Schedule seq groups prompts.
    # Only 1 seq group should be scheduled since max_seq_group is 2
    # and one is prompting.
    _, out = scheduler.schedule()
    assert set(out.scheduled_seq_groups) == set([all_seq_groups[1]])


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
    assert out.prompt_run
    assert seq_group_meta[0].request_id == '0'

    # wait for a second before scheduling next prompt
    time.sleep(1)
    _, seq_group = create_dummy_prompt("1", prompt_length=block_size)
    scheduler.add_seq_group(seq_group)

    # second prompt should *not* be scheduled
    seq_group_meta, out = scheduler.schedule()
    assert not out.prompt_run
    assert seq_group_meta[0].request_id == '0'

    # wait for more than 0.5 second and try again
    time.sleep(0.6)
    seq_group_meta, out = scheduler.schedule()
    assert out.prompt_run
    assert seq_group_meta[0].request_id == '1'


def test_scheduler_with_cache():
    # Initialize the scheduler
    max_batched_tokens = 80
    max_seq_group = 8
    max_model_length = 80
    max_paddings = 256
    scheduler_config = SchedulerConfig(max_batched_tokens, max_seq_group,
                                       max_model_length, max_paddings)

    block_size = 16
    cache_config = CacheConfig(block_size,
                               1.0,
                               1,
                               "auto",
                               enable_prefix_caching=True)
    cache_config.num_gpu_blocks = 8
    cache_config.num_cpu_blocks = 8

    scheduler = Scheduler(scheduler_config, cache_config, None)

    seq0_prompt_length = 64
    seq0 = Sequence(seq_id=0,
                    prompt="zero to sixty three",
                    block_size=block_size,
                    prompt_token_ids=list(range(seq0_prompt_length)))
    seq0_group = SequenceGroup(request_id=0,
                               seqs=[seq0],
                               sampling_params=SamplingParams(),
                               arrival_time=time.time())
    # Allocate 4 blocks for caching
    scheduler.block_manager.allocate(seq0_group)
    # Mark the 4 blocks as computed
    scheduler.block_manager.mark_blocks_as_computed(seq0_group)
    # Requires 0 extra blocks, 16 batched tokens
    scheduler.add_seq_group(seq0_group)
    assert len(seq0.logical_token_blocks) -\
        scheduler.block_manager.get_num_cached_blocks(seq0) == 0
    assert seq0.get_len() -\
        scheduler.block_manager.get_num_computed_tokens(seq0) == 16

    seq1_prompt_length = 48
    seq1 = Sequence(seq_id=1,
                    prompt="zero to forty seven",
                    block_size=block_size,
                    prompt_token_ids=list(range(seq1_prompt_length)))
    seq1_group = SequenceGroup(request_id=1,
                               seqs=[seq1],
                               sampling_params=SamplingParams(),
                               arrival_time=time.time())
    # Requires 0 extra block, 16 batched tokens
    scheduler.add_seq_group(seq1_group)
    assert len(seq1.logical_token_blocks) -\
        scheduler.block_manager.get_num_cached_blocks(seq1) == 0
    assert seq1.get_len() -\
        scheduler.block_manager.get_num_computed_tokens(seq1) == 16

    seq2_prompt_length = 56
    seq2 = Sequence(seq_id=2,
                    prompt="zero to fifty four",
                    block_size=block_size,
                    prompt_token_ids=list(range(seq2_prompt_length)))
    seq2_group = SequenceGroup(request_id=2,
                               seqs=[seq2],
                               sampling_params=SamplingParams(),
                               arrival_time=time.time())
    # Requires 1 extra block, 8 batched tokens
    scheduler.add_seq_group(seq2_group)
    assert len(seq2.logical_token_blocks) -\
        scheduler.block_manager.get_num_cached_blocks(seq2) == 1
    assert seq2.get_len() -\
        scheduler.block_manager.get_num_computed_tokens(seq2) == 8

    seq3_prompt_length = 80
    seq3 = Sequence(seq_id=3,
                    prompt="zero to seventy nine",
                    block_size=block_size,
                    prompt_token_ids=list(range(seq3_prompt_length)))
    seq3_group = SequenceGroup(request_id=3,
                               seqs=[seq3],
                               sampling_params=SamplingParams(),
                               arrival_time=time.time())
    # Requires 1 extra blocks, 16 batched tokens
    scheduler.add_seq_group(seq3_group)
    assert len(seq3.logical_token_blocks) -\
        scheduler.block_manager.get_num_cached_blocks(seq3) == 1
    assert seq3.get_len() -\
        scheduler.block_manager.get_num_computed_tokens(seq3) == 16

    seq4_prompt_length = 96
    seq4 = Sequence(seq_id=4,
                    prompt="zero to ninety five",
                    block_size=block_size,
                    prompt_token_ids=list(range(seq4_prompt_length)))
    seq4_group = SequenceGroup(request_id=4,
                               seqs=[seq4],
                               sampling_params=SamplingParams(),
                               arrival_time=time.time())
    # Requires 2 extra block, 32 batched tokens
    scheduler.add_seq_group(seq4_group)
    assert len(seq4.logical_token_blocks) -\
        scheduler.block_manager.get_num_cached_blocks(seq4) == 2
    assert seq4.get_len() -\
        scheduler.block_manager.get_num_computed_tokens(seq4) == 32

    scheduler_outputs = scheduler._schedule()
    scheduled_seq_groups_ids = []
    for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
        scheduled_seq_groups_ids.append(scheduled_seq_group.request_id)
    scheduled_seq_groups_ids.sort()
    # The seq4 cannot be scheduled because if it is added, then the
    # batched tokens num will exceed the limitation
    assert scheduled_seq_groups_ids == [0, 1, 2, 3]
