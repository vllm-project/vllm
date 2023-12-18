
from typing import List
import pytest

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.sequence import SequenceGroup
from tests.utils import round_up_to_next_block

from .utils import create_dummy_prompt


def test_scheduler_add_seq_group():
    block_size = 4
    scheduler_config = SchedulerConfig(100, 64, 1)
    cache_config = CacheConfig(block_size, 1.0, 1)
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
    cache_config = CacheConfig(block_size, 1.0, 1)
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
    cache_config = CacheConfig(block_size, 1.0, 1)
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add seq groups to scheduler.
    running: List[SequenceGroup] = []
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=block_size,
                                           num_processed_token_ids=block_size -
                                           1)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

    # Schedule seq groups prompts.
    seq_group_meta, out = scheduler.schedule()
    assert set(out.scheduled_seq_groups) == set(running)
    assert out.num_batched_tokens == num_seq_group * seq_group.get_seqs(
    )[0].get_len()
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
    cache_config = CacheConfig(block_size, 1.0, 1)
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
    assert out.num_batched_tokens == seq_group_a.get_seqs()[0].get_len() * 2
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 2
    assert scheduler.get_num_unfinished_seq_groups() == 2

    # Append "generated" tokens, allowing the sequence to mark prompt tokens as
    # processed.
    seq_a.append_token_id(0, {0: 0.0})
    seq_b.append_token_id(0, {0: 0.0})

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
    assert out.num_batched_tokens == seq_group_b.get_seqs()[0].get_len()
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 1
    assert scheduler.get_num_unfinished_seq_groups() == 1


@pytest.mark.parametrize("block_size", [1, 8, 16, 32])
@pytest.mark.parametrize("prompt_len", [1, 128])
@pytest.mark.parametrize("num_unprocessed_tokens", [2, 17, 128])
@pytest.mark.parametrize("num_seq_group", [1, 4, 16])
def test_can_schedule_seqs_with_multiple_unprocessed_tokens(
        block_size: int, prompt_len: int, num_unprocessed_tokens: int,
        num_seq_group: int):
    """Verify scheduler can schedule sequences with more than one unprocessed
    tokens. This occurs when the worker emits more than one token.
    """
    max_model_len = 2048
    scheduler_config = SchedulerConfig(max_num_batched_tokens=max_model_len,
                                       max_num_seqs=num_seq_group,
                                       max_model_len=max_model_len)
    cache_config = CacheConfig(block_size=block_size,
                               gpu_memory_utilization=1.0,
                               swap_space=0)
    cache_config.num_cpu_blocks = 0
    cache_config.num_gpu_blocks = 8192 // block_size
    scheduler = Scheduler(scheduler_config, cache_config, None)

    prompt_lens = [prompt_len for _ in range(num_seq_group)]

    token_ids_to_append = [
        list(range(num_unprocessed_tokens)) for _ in range(num_seq_group)
    ]

    # Add seq groups to scheduler.
    running: List[SequenceGroup] = []
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(request_id=str(i),
                                           prompt_length=prompt_lens[i],
                                           block_size=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

    # Schedule seq groups prompts.
    _, out = scheduler.schedule()
    assert set(out.scheduled_seq_groups) == set(running)

    # Add tokens to sequences
    for seq_group in out.scheduled_seq_groups:
        for i, seq in enumerate(seq_group.get_seqs()):
            seq.append_token_ids(token_ids_to_append[i],
                                 logprobs=[{
                                     token_id: 0.0
                                 } for token_id in token_ids_to_append[i]])

    # Schedule seq groups generation.
    seq_group_meta, out = scheduler.schedule()
    assert set(out.scheduled_seq_groups) == set(running)

    for seq_group_metadata in seq_group_meta:
        # Only one seq per group in this test.
        seq_id = next(iter(seq_group_metadata.seq_data.keys()))

        block_table = seq_group_metadata.block_tables[seq_id]
        blocks_required = (seq_group_metadata.seq_data[seq_id].get_len() - 1 +
                           block_size) // block_size
        assert len(block_table) == blocks_required


@pytest.mark.parametrize("block_size", [1, 8, 16, 32])
@pytest.mark.parametrize("prompt_len", [1, 128])
@pytest.mark.parametrize("num_unprocessed_tokens", [1, 9])
@pytest.mark.parametrize("num_preallocated_slots_per_step", [1, 9])
@pytest.mark.parametrize("num_seq_group", [1, 4])
def test_can_schedule_multiple_steps(block_size: int, prompt_len: int,
                                     num_preallocated_slots_per_step: int,
                                     num_unprocessed_tokens: int,
                                     num_seq_group: int):
    """Verify correct scheduling when the model runs more than one step per
    scheduler iteration.
    """
    max_model_len = 2048
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=max_model_len,
        max_num_seqs=num_seq_group,
        max_model_len=max_model_len,
        num_preallocated_slots_per_step=num_preallocated_slots_per_step)
    cache_config = CacheConfig(block_size=block_size,
                               gpu_memory_utilization=1.0,
                               swap_space=0)
    cache_config.num_cpu_blocks = 0
    cache_config.num_gpu_blocks = 8192 // block_size
    scheduler = Scheduler(scheduler_config, cache_config, None)

    prompt_lens = [prompt_len for _ in range(num_seq_group)]

    token_ids_to_append = [
        list(range(num_unprocessed_tokens)) for _ in range(num_seq_group)
    ]

    # Add seq groups to scheduler.
    running: List[SequenceGroup] = []
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(request_id=str(i),
                                           prompt_length=prompt_lens[i],
                                           block_size=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

    # Schedule seq groups prompts.
    _, out = scheduler.schedule()
    assert set(out.scheduled_seq_groups) == set(running)

    # Add tokens to sequences
    for seq_group in out.scheduled_seq_groups:
        for i, seq in enumerate(seq_group.get_seqs()):
            seq.append_token_ids(token_ids_to_append[i],
                                 logprobs=[{
                                     token_id: 0.0
                                 } for token_id in token_ids_to_append[i]])

    # Schedule seq groups generation.
    seq_group_meta, out = scheduler.schedule()
    assert set(out.scheduled_seq_groups) == set(running)

    for seq_group_metadata in seq_group_meta:
        # Only one seq per group in this test.
        seq_id = next(iter(seq_group_metadata.seq_data.keys()))

        # The last slot is not required because it is for the last generated
        # token, and will be stored in the next iteration.
        slots_required = (seq_group_metadata.seq_data[seq_id].get_len() +
                          num_preallocated_slots_per_step)
        blocks_required = round_up_to_next_block(slots_required, block_size)

        block_table = seq_group_metadata.block_tables[seq_id]
        assert len(block_table) == blocks_required


def test_scheduler_schedule_chunked_prefill():
    block_size = 4
    num_seq_group = 2
    max_model_len = 16
    max_chunked_prefill_len = 2
    max_num_prompt_seqs = 1
    scheduler_config = SchedulerConfig(
        64,
        num_seq_group,
        max_model_len,
        flash_style=True,
        max_chunked_prefill_len=max_chunked_prefill_len,
        max_num_prompt_seqs=max_num_prompt_seqs)
    cache_config = CacheConfig(block_size, 1.0, 1)
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add seq groups to scheduler.
    seq_groups: List[SequenceGroup] = []
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=block_size,
                                           num_processed_token_ids=block_size -
                                           1)
        scheduler.add_seq_group(seq_group)
        seq_groups.append(seq_group)

    # Schedule chunk prefill. Only the first seq_group should be scheduled.
    seq_group_meta, out = scheduler.schedule()
    assert set(out.scheduled_seq_groups) == set(seq_groups[:1])
    seq_groups[0].get_num_unprefilled() == 2
    seq_groups[1].get_num_unprefilled() == 4
    assert out.num_batched_tokens == 2
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 1
    assert seq_group_meta[0].request_id == "0"
    assert seq_group_meta[0].is_chunked_prefill
    assert seq_group_meta[0].is_prompt

    # Schedule chunk prefill. Still Only the first seq_group should be scheduled.
    seq_group_meta, out = scheduler.schedule()
    assert set(out.scheduled_seq_groups) == set(seq_groups[:1])
    seq_groups[0].get_num_unprefilled() == 0
    seq_groups[1].get_num_unprefilled() == 4
    assert out.num_batched_tokens == 2
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 1
    assert seq_group_meta[0].request_id == "0"
    assert not seq_group_meta[0].is_chunked_prefill
    assert seq_group_meta[0].is_prompt

    # Schedule chunk prefill. This time the second seq_group should be selected
    # for chunk prefill, and the first seq_group should be select for decoding.
    seq_group_meta, out = scheduler.schedule()
    assert set(out.scheduled_seq_groups) == set(seq_groups)
    seq_groups[0].get_num_unprefilled() == 0
    seq_groups[1].get_num_unprefilled() == 2
    assert out.num_batched_tokens == 3
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 2
    assert seq_group_meta[0].request_id == "1"
    assert seq_group_meta[0].is_chunked_prefill
    assert seq_group_meta[0].is_prompt
    assert seq_group_meta[1].request_id == "0"
    assert not seq_group_meta[1].is_chunked_prefill
    assert not seq_group_meta[1].is_prompt


def test_scheduler_max_seqs():
    block_size = 4
    num_seq_group = 4
    max_seq_group = 2
    max_model_len = 16
    scheduler_config = SchedulerConfig(64, max_seq_group, max_model_len)
    cache_config = CacheConfig(block_size, 1.0, 1)
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)

    all_seq_groups: List[SequenceGroup] = []
    # Add seq groups to scheduler.
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=block_size,
                                           num_processed_token_ids=block_size -
                                           1)
        all_seq_groups.append(seq_group)

    # Append 1 seq group
    scheduler.add_seq_group(all_seq_groups[0])

    # Schedule seq groups prompts.
    seq_group_meta, out = scheduler.schedule()
    assert set(out.scheduled_seq_groups) == set([all_seq_groups[0]])

    # Schedule seq groups generation.
    seq_group_meta, out = scheduler.schedule()
    assert set(out.scheduled_seq_groups) == set([all_seq_groups[0]])

    # Append 2 more seq group
    running: List[SequenceGroup] = []
    scheduler.add_seq_group(all_seq_groups[1])
    scheduler.add_seq_group(all_seq_groups[2])

    # Schedule seq groups prompts.
    # Only 1 seq group should be scheduled since max_seq_group is 2
    # and one is prompting.
    seq_group_meta, out = scheduler.schedule()
    assert set(out.scheduled_seq_groups) == set([all_seq_groups[1]])
