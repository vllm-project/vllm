from typing import List
import pytest

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.scheduler import Scheduler

from .utils import create_dummy_prompt


@pytest.mark.parametrize("min_decodes_per_prefill",
                         [1, 2, 4, 8, 16, 32, 64, 128])
def test_scheduler_delayed_prefill_scheduling(min_decodes_per_prefill):
    block_size = 4
    num_seq_group = 64
    max_model_len = 16
    num_pattern_repeat = 4
    scheduler_config = SchedulerConfig(
        64,
        num_seq_group,
        max_model_len,
        min_decodes_per_prefill=min_decodes_per_prefill)
    cache_config = CacheConfig(block_size, 1.0, 1)
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add seq groups to scheduler.
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=block_size,
                                           num_processed_token_ids=block_size -
                                           1)
        scheduler.add_seq_group(seq_group)

    scheduled_is_prefilling = []
    # Schedule seq groups prompts.
    for i in range((min_decodes_per_prefill + 1) * num_pattern_repeat):
        seq_group_meta, out = scheduler.schedule()
        # num_prompt_groups only come from prefilling_outputs
        # use this to determine if a prefilling step is scheduled
        scheduled_is_prefilling.append(out.num_prompt_groups > 0)

        # abort most of the seq groups to free slots for the next iteration
        # note we leave one in the scheduler to avoid always schedule prefilling
        for j in range(len(seq_group_meta) - 1):
            scheduler.abort_seq_group(seq_group_meta[j].request_id)

    # the scheduled sequence with delayed prefilling should look like
    # [True, False * min_decodes_per_prefill,
    # True, False * min_decodes_per_prefill, ...]
    expected_pattern = [True] + [False] * min_decodes_per_prefill
    assert scheduled_is_prefilling == expected_pattern * num_pattern_repeat,\
    f"Delayed refill scheduling is not correct, expected pattern \
          {expected_pattern} (to be repeated {num_pattern_repeat} times), \
            got {scheduled_is_prefilling}"
