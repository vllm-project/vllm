from typing import List

import pytest  # noqa

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.sequence import SequenceGroup, Logprob

from .utils import create_dummy_prompt


def get_sequence_groups(scheduler_output):
    return [s.seq_group for s in scheduler_output.scheduled_seq_groups]


def append_new_token(seq_group, token_id: int):
    for seq in seq_group.get_seqs():
        seq.append_token_id(token_id, {token_id: Logprob(token_id)})


def test_chunked_prefill_schedule_simple():
    """Verify basic scheduling works."""
    block_size = 4
    num_seq_group = 4
    max_model_len = 16
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig(max_num_batched_tokens, num_seq_group, max_model_len, enable_chunked_prefill=True)
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


def test_chunked_prefill_chunked():
    """Verify prefills are chunked properly."""
    block_size = 4
    max_seqs = 60
    max_model_len = 80
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig(max_num_batched_tokens, max_seqs, max_model_len, enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: List[SequenceGroup] = []

    # Add seq groups to scheduler.
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)
    
    # Verify the second request is chunked.
    seq_group_meta, out = scheduler.schedule()

    assert set(get_sequence_groups(out)) == set(running)
    assert seq_group_meta[0].token_chunk_size == 60
    # Verify it is chunked.
    assert seq_group_meta[1].token_chunk_size == 4
    assert out.num_prefill_groups == 2
    assert out.num_batched_tokens == 64
    # Only the first seq group has a new token appended.
    append_new_token(running[0], 1)

    # One chunked prefill, and one decoding.
    breakpoint()
    seq_group_meta, out = scheduler.schedule()
    assert set(get_sequence_groups(out)) == set(running)
    # The first one is decoding.
    assert seq_group_meta[0].token_chunk_size == 1
    # The second one is a chunked prefill.
    assert seq_group_meta[1].token_chunk_size == 60
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 61



def test_chunked_prefill_maximal_decoding():
    """Verify decoding requests are prirotized.
    
    The priority should be
    1. running decode.
    2. runnig prefill (chunked prefill).
    3. swapped.
    4. new prefill.
    """
    pass


def test_chunked_prefill_schedule_prompt_limit():
    """Verify prompt limit is correctly set."""
    pass


def test_chunked_prefill_swap():
    """Verify swapping works with chunked prefill requests"""
    pass


def test_chunked_prefill_preempt():
    """Verify preempt works with chunked prefill requests"""
    pass


def test_chunked_prefill_max_tokens():
    """Verify scheduler doesn't schedule """
