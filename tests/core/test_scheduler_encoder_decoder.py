from typing import List

import pytest  # noqa

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.sequence import SequenceGroup

from .utils import (append_new_token, create_dummy_prompt_encoder_decoder,
                    get_sequence_groups, schedule_and_update_computed_tokens)

def test_scheduler_schedule_simple_encoder_decoder():
    block_size = 4
    num_seq_group = 4
    max_model_len = 16
    scheduler_config = SchedulerConfig(64, num_seq_group, max_model_len)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 16  # enc and dec prompts per seq_group
    cache_config.num_gpu_blocks = 16  # enc and dec prompts per seq_group
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: List[SequenceGroup] = []

    # Add seq groups to scheduler.
    req_id_list = []
    for i in range(num_seq_group):
        req_id = str(i)
        req_id_list.append(req_id)
        _, _, seq_group = create_dummy_prompt_encoder_decoder(
            req_id, block_size, block_size, block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

    # Schedule seq groups prompts.
    num_tokens = block_size * num_seq_group
    seq_group_meta_list, out = schedule_and_update_computed_tokens(scheduler)
    # - Verify that sequence group cross-attention block tables are
    #   registered with the block manager
    assert all([(req_id in scheduler.block_manager.cross_block_tables)
                for req_id in req_id_list])
    assert set(get_sequence_groups(out)) == set(running)
    assert out.num_batched_tokens == num_tokens
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta_list) == num_seq_group
    append_new_token(out, 1)

    # Schedule seq groups generation.
    seq_group_meta_list, out = schedule_and_update_computed_tokens(scheduler)
    # - Verify that sequence group metadata includes encoder attention
    #   and cross-attention metadata
    assert all([
        not ((seq_group_meta.encoder_seq_data is None) or
             (seq_group_meta.cross_block_table is None))
        for seq_group_meta in seq_group_meta_list
    ])
    assert set(get_sequence_groups(out)) == set(running)
    assert out.num_batched_tokens == num_seq_group
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta_list) == num_seq_group
    append_new_token(out, 1)

    # Abort sequences
    for req_id in req_id_list:
        scheduler.abort_seq_group(req_id)
        # - Verify that sequence group cross-attention block tables are
        #   NO LONGER registered with the block manager
        assert req_id not in scheduler.block_manager.cross_block_tables
