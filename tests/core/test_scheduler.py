from typing import List
import pytest

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.sequence import SequenceGroup

from .utils import create_dummy_prompt


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
