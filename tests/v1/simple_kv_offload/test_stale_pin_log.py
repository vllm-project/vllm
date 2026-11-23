# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The unused pending CPU hit release is benign and logged at DEBUG only."""

import logging

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request
from vllm.v1.simple_kv_offload.manager import logger as manager_logger

from .test_scheduler import (
    BLOCK_SIZE,
    _alloc_and_register,
    make_request,
    make_scheduler,
    make_scheduler_output,
    simulate_store_completion,
)


def test_unused_pending_cpu_hit_released_at_debug(caplog_vllm) -> None:
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    num_blocks = 2
    req = make_request(num_blocks=num_blocks)
    kv_blocks = _alloc_and_register(fix, req, num_blocks)
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)
    meta = sched.build_connector_meta(
        make_scheduler_output(
            {req.request_id: num_blocks * BLOCK_SIZE},
            new_reqs={req.request_id: kv_blocks.get_block_ids()},
        )
    )
    simulate_store_completion(sched, meta.store_event)

    req2 = Request(
        request_id="req-unused-hit",
        prompt_token_ids=req.prompt_token_ids,
        sampling_params=req.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=req._block_hasher,
    )
    hit_tokens, _ = sched.get_num_new_matched_tokens(req2, num_computed_tokens=0)
    assert hit_tokens > 0
    assert req2.request_id in sched._pending_cpu_hits

    gpu_blocks = fix.gpu_block_pool.get_new_blocks(num_blocks)
    # Set DEBUG on the manager's own logger: the "vllm" parent logger may have
    # been configured at INFO by an earlier test in the same session.
    with caplog_vllm.at_level(logging.DEBUG, logger=manager_logger.name):
        sched.update_state_after_alloc(
            req2, KVCacheBlocks(blocks=(gpu_blocks,)), num_external_tokens=0
        )

    assert req2.request_id not in sched._pending_cpu_hits
    records = [r for r in caplog_vllm.records if "req-unused-hit" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    message = records[0].getMessage()
    assert "unexpectedly" not in message
    assert "stale" not in message
