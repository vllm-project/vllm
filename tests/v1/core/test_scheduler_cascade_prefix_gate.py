# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""num_common_prefix_blocks is computed only when cascade attention is enabled.

The value has exactly one consumer, _compute_cascade_attn_prefix_lens in the
model runner. Producing it costs a python walk over the first running
request's block list, which for a single long-context request is one iteration
per allocated block, every scheduler step. disable_cascade_attn defaults to
True, so by default that walk buys nothing.
"""

from unittest.mock import patch

from tests.v1.core.utils import create_requests, create_scheduler


def _drain_to_running(scheduler, num_requests=2, num_tokens=4):
    for request in create_requests(num_requests=num_requests, num_tokens=num_tokens):
        scheduler.add_request(request)
    scheduler.schedule()
    assert scheduler.running
    return scheduler


def test_skipped_when_cascade_attention_disabled():
    scheduler = _drain_to_running(create_scheduler())
    assert scheduler.disable_cascade_attn, "expected the vLLM default"

    with patch.object(
        scheduler.kv_cache_manager, "get_num_common_prefix_blocks"
    ) as get_blocks:
        output = scheduler.schedule()

    get_blocks.assert_not_called()
    num_groups = len(scheduler.kv_cache_config.kv_cache_groups)
    assert output.num_common_prefix_blocks == [0] * num_groups


def test_computed_when_cascade_attention_enabled():
    scheduler = _drain_to_running(create_scheduler())
    scheduler.disable_cascade_attn = False
    num_groups = len(scheduler.kv_cache_config.kv_cache_groups)

    with patch.object(
        scheduler.kv_cache_manager,
        "get_num_common_prefix_blocks",
        return_value=[7] * num_groups,
    ) as get_blocks:
        output = scheduler.schedule()

    get_blocks.assert_called_once_with(scheduler.running[0].request_id)
    assert output.num_common_prefix_blocks == [7] * num_groups


def test_zeros_when_nothing_is_running():
    scheduler = create_scheduler()
    scheduler.disable_cascade_attn = False
    assert not scheduler.running

    with patch.object(
        scheduler.kv_cache_manager, "get_num_common_prefix_blocks"
    ) as get_blocks:
        output = scheduler.schedule()

    get_blocks.assert_not_called()
    num_groups = len(scheduler.kv_cache_config.kv_cache_groups)
    assert output.num_common_prefix_blocks == [0] * num_groups
