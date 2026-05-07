# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import torch

from vllm.model_executor.layers.mamba.mamba_utils import get_temporal_copy_spec
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.worker.mamba_utils import preprocess_mamba


def _make_scheduler_output(
    finished_req_ids: set[str],
    preempted_req_ids: set[str] | None,
    resumed_req_ids: set[str],
) -> SchedulerOutput:
    cached = CachedRequestData.make_empty()
    cached.resumed_req_ids = resumed_req_ids
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=finished_req_ids,
        free_encoder_mm_hashes=[],
        preempted_req_ids=preempted_req_ids,
    )


def test_resumed_req_ids_cleared_from_mamba_state_idx():
    """When a request is force-preempted (e.g. reset_prefix_cache),
    it appears in resumed_req_ids but NOT in preempted_req_ids.
    preprocess_mamba must still clear its mamba_state_idx entry,
    otherwise stale indices can point beyond the new block allocation.
    """
    spec = MagicMock(block_size=64, num_speculative_blocks=0)
    cache_config = MagicMock(enable_prefix_caching=True)
    input_batch = MagicMock(req_ids=[])
    copy_bufs = MagicMock(mamba_group_ids=[0], mamba_spec=spec)

    mamba_state_idx = {
        "finished": 1,
        "preempted": 2,
        "resumed": 3,  # only in resumed_req_ids, NOT in preempted
        "keep": 99,
    }
    sched = _make_scheduler_output(
        finished_req_ids={"finished"},
        preempted_req_ids={"preempted"},
        resumed_req_ids={"resumed"},
    )

    with patch(
        "vllm.v1.worker.mamba_utils.get_mamba_groups",
        return_value=([0], spec),
    ):
        preprocess_mamba(
            sched,
            MagicMock(),
            cache_config,
            mamba_state_idx,
            input_batch,
            {},
            {},
            (),
            copy_bufs,
        )

    assert mamba_state_idx == {"keep": 99}


def test_get_temporal_copy_spec_clamps_out_of_range_block_index():
    state = torch.arange(12, dtype=torch.float32).view(3, 4)
    block_ids = [0, 2]

    copy_spec = get_temporal_copy_spec(
        state,
        block_ids,
        cur_block_idx=1,
        num_accepted_tokens=2,
    )

    assert copy_spec.start_addr == state[block_ids[-1]].data_ptr()
    assert copy_spec.num_elements == state[block_ids[-1]].numel()
