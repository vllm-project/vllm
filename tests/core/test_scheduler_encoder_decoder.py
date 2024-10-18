from typing import List

import pytest  # noqa

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.sequence import SequenceGroup

from .utils import (append_new_token, create_dummy_prompt_encoder_decoder,
                    get_sequence_groups, schedule_and_update_computed_tokens)


def test_scheduler_schedule_simple_encoder_decoder():
    '''
    Test basic scheduler functionality in the context
    of an encoder/decoder model. Focus on testing
    enc/dec-specific functionality sense tests already
    exist for decoder-only functionality

    Test behavior:
    * Construct Scheduler
    * Construct dummy encoder/decoder sequence groups
    * Add dummy seq groups to scheduler backlog
    * Schedule the next seq group & validate:
        * Cross-attn block tables
        * Updated states of seq groups
        * Number of batched tokens
        * Number of blocks to copy/swap-in/swap-out
        * Number of scheduled seq groups
    * Repeat for both prefill- and decode-phase
    * Abort scheduled seq groups
    * Assert that aborted seq groups no longer appear in
      cross-attention block table
    '''

    block_size = 4
    num_seq_group = 4
    max_model_len = 16
    scheduler_config = SchedulerConfig(
        task="generate",
        max_num_batched_tokens=64,
        max_num_seqs=num_seq_group,
        max_model_len=max_model_len,
    )
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

    # Schedule seq groups prefill.
    num_tokens = block_size * num_seq_group
    seq_group_meta_list, out = schedule_and_update_computed_tokens(scheduler)
    # - Verify that sequence group cross-attention block tables are
    #   registered with the block manager
    assert all([(req_id in scheduler.block_manager.cross_block_tables)
                for req_id in req_id_list])
    # - Validate sequence-group status
    assert set(get_sequence_groups(out)) == set(running)
    # - Validate number of batched tokens
    assert out.num_batched_tokens == num_tokens
    # - Validate there are no remaining blocks to swap
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    # - Validate all seq groups were scheduled
    assert len(seq_group_meta_list) == num_seq_group
    append_new_token(out, 1)

    # Schedule seq groups decode.
    seq_group_meta_list, out = schedule_and_update_computed_tokens(scheduler)
    # - Verify that sequence group metadata includes encoder attention
    #   and cross-attention metadata
    assert all([
        not ((seq_group_meta.encoder_seq_data is None) or
             (seq_group_meta.cross_block_table is None))
        for seq_group_meta in seq_group_meta_list
    ])
    # - Validate sequence-group status
    assert set(get_sequence_groups(out)) == set(running)
    # - Validate there is one batched token per seq group
    assert out.num_batched_tokens == num_seq_group
    # - Validate there are no remaining blocks to swap
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    # - Validate that all seq groups were scheduled
    assert len(seq_group_meta_list) == num_seq_group
    append_new_token(out, 1)

    # Abort sequences
    for req_id in req_id_list:
        scheduler.abort_seq_group(req_id)
        # - Verify that sequence group cross-attention block tables are
        #   NO LONGER registered with the block manager
        assert req_id not in scheduler.block_manager.cross_block_tables
