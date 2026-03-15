# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.v1 import utils as v1_utils
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    # These tests do not initialize distributed state.
    # Skip the global teardown fixture to stay hermetic on CPU-only hosts.
    return False


def _make_scheduler_output(
    num_scheduled_tokens: dict[str, int],
    new_req_ids: list[str] | None = None,
    cached_context_req_ids: list[str] | None = None,
    cached_generation_req_ids: list[str] | None = None,
) -> SchedulerOutput:
    new_req_ids = new_req_ids or []
    cached_context_req_ids = cached_context_req_ids or []
    cached_generation_req_ids = cached_generation_req_ids or []

    cached_req_ids = cached_context_req_ids + cached_generation_req_ids
    cached_num_output_tokens = (
        [0] * len(cached_context_req_ids) + [1] * len(cached_generation_req_ids)
    )
    cached_reqs = CachedRequestData(
        req_ids=cached_req_ids,
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in cached_req_ids],
        all_token_ids={},
        new_block_ids=[None for _ in cached_req_ids],
        num_computed_tokens=[0 for _ in cached_req_ids],
        num_output_tokens=cached_num_output_tokens,
    )

    return SchedulerOutput(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=req_id) for req_id in new_req_ids
        ],
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


@pytest.mark.parametrize(
    "scheduler_output,expected",
    [
        (
            _make_scheduler_output(
                num_scheduled_tokens={"new_req": 4},
                new_req_ids=["new_req"],
            ),
            "prefill_batch",
        ),
        (
            _make_scheduler_output(
                num_scheduled_tokens={"decode_req": 2},
                cached_generation_req_ids=["decode_req"],
            ),
            "decode_batch",
        ),
        (
            _make_scheduler_output(
                num_scheduled_tokens={"new_req": 3, "decode_req": 2},
                new_req_ids=["new_req"],
                cached_generation_req_ids=["decode_req"],
            ),
            "mixed_batch",
        ),
        (
            _make_scheduler_output(num_scheduled_tokens={}),
            "empty_batch",
        ),
    ],
)
def test_classify_batch_stage(scheduler_output: SchedulerOutput, expected: str):
    assert v1_utils.classify_batch_stage(scheduler_output) == expected


def test_get_batch_stage_scope_name_disabled(monkeypatch):
    scheduler_output = _make_scheduler_output(
        num_scheduled_tokens={"new_req": 1},
        new_req_ids=["new_req"],
    )
    monkeypatch.setattr(v1_utils.envs, "VLLM_CUSTOM_SCOPES_FOR_PROFILING", False)
    monkeypatch.setattr(v1_utils.envs, "VLLM_NVTX_SCOPES_FOR_PROFILING", False)

    assert v1_utils.get_batch_stage_scope_name(scheduler_output) is None


def test_get_batch_stage_scope_name_enabled(monkeypatch):
    scheduler_output = _make_scheduler_output(
        num_scheduled_tokens={"decode_req": 1},
        cached_generation_req_ids=["decode_req"],
    )
    monkeypatch.setattr(v1_utils.envs, "VLLM_CUSTOM_SCOPES_FOR_PROFILING", False)
    monkeypatch.setattr(v1_utils.envs, "VLLM_NVTX_SCOPES_FOR_PROFILING", True)

    assert v1_utils.get_batch_stage_scope_name(scheduler_output) == "decode_batch"
