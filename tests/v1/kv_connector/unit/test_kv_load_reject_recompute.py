# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for correct recompute when a KV connector rejects a
synchronous load (kv_load_failure_policy="recompute").

Two bugs are covered:
  * Worker (V2 GPU model runner): a KV-load-failure recompute moves
    num_computed_tokens backward, but it was written only to the optimistic CPU
    mirror and not the GPU tensor, so the recompute gathered tokens/positions from
    a stale offset. Covered end-to-end by test_reject_recompute_matches_baseline
    (GPU).
  * Scheduler (async scheduling): the failed request's in-flight
    num_output_placeholders were not rolled back, inflating the recompute's
    scheduled token count. Covered by test_recompute_rolls_back_async_placeholders
    (CPU).
"""

from collections.abc import Callable
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.request import Request

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)


def _make_get_num_new_matched_tokens(
    req_num_new_matched_tokens: dict[str, int],
    async_load: bool,
) -> Callable[[Request, int], tuple[int, bool]]:
    def get_num_new_matched_tokens(request: Request, _: int) -> tuple[int, bool]:
        return req_num_new_matched_tokens.get(request.request_id, 0), async_load

    return get_num_new_matched_tokens


@pytest.mark.cpu_test
def test_recompute_rolls_back_async_placeholders():
    """A rejected sync KV load must roll back the failed request's in-flight
    async output placeholders during recompute recovery.

    Otherwise the recompute is scheduled with num_tokens + num_output_placeholders
    tokens and appends garbage past the prompt. Exactly one in-flight frame is the
    one consumed by the recovery skip in the same step, so the number of stale
    future frames to discard is num_output_placeholders - 1 (discarding one more
    drops the recompute's first output token -> off-by-one shift).
    """
    vllm_config = create_vllm_config(kv_load_failure_policy="recompute")
    scheduler = create_scheduler(vllm_config)

    num_external = 3 * scheduler.block_size
    num_prompt_tokens = 4 * scheduler.block_size
    request = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request)

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {request.request_id: num_external}, async_load=False
        )
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1

    # num_output_placeholders is normally set by AsyncScheduler. Set it directly to
    # simulate two async output frames in flight (a completed prefill's sampled
    # token plus one decode) reserved before the load failure surfaced.
    request.num_output_placeholders = 2
    request.async_tokens_to_discard = 0

    # Fail all of the request's blocks -> num_computed_tokens truncates to 0.
    req_block_ids = scheduler_output.scheduled_new_reqs[0].block_ids[0]
    model_runner_output = create_model_runner_output(
        [request], invalid_block_ids=set(req_block_ids), use_eos=True
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # The request is rescheduled to recompute from a truncated prefix...
    assert request.request_id in scheduler.requests
    assert request.num_computed_tokens == 0
    # ...with its in-flight placeholders rolled back and num_output_placeholders - 1
    # stale frames marked for discard.
    assert request.num_output_placeholders == 0
    assert request.async_tokens_to_discard == 1


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="end-to-end reject->recompute correctness test requires a GPU",
)
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_reject_recompute_matches_baseline(async_scheduling: bool):
    """End-to-end: a connector that promises a block-aligned prefix then rejects
    the load must recompute to the SAME greedy output as no connector at all,
    under both synchronous and asynchronous scheduling.

    Run from the repo root so the worker can import the connector module path.
    """
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    model = "facebook/opt-125m"  # non-MoE -> V2 GPU model runner (the affected path)
    prompt = (
        "The quick brown fox jumps over the lazy dog near the river while the "
        "sun sets slowly behind the distant mountains at the end of a long day"
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=8)

    def run(kv_transfer_config: "KVTransferConfig | None") -> list[int]:
        llm = LLM(
            model=model,
            enforce_eager=True,
            enable_prefix_caching=False,
            async_scheduling=async_scheduling,
            gpu_memory_utilization=0.3,
            kv_transfer_config=kv_transfer_config,
        )
        out = list(llm.generate([prompt], sampling)[0].outputs[0].token_ids)
        del llm
        return out

    baseline = run(None)
    recovered = run(
        KVTransferConfig(
            kv_connector="RejectRecomputeConnector",
            kv_connector_module_path=(
                "tests.v1.kv_connector.unit.reject_recompute_connector"
            ),
            kv_role="kv_both",
            kv_load_failure_policy="recompute",
        )
    )

    assert recovered == baseline, (
        f"reject->recompute {recovered} != baseline {baseline} "
        f"(async_scheduling={async_scheduling})"
    )
