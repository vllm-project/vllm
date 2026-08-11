# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.v1.executor import ray_utils
from vllm.v1.executor.ray_utils import detach_zero_copy_from_model_runner_output
from vllm.v1.outputs import (
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    RoutedExpertsLists,
)


def _make_readonly(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


def test_detach_zero_copy_from_model_runner_output_copies_only_numpy_views():
    cu_num_generated_tokens = [0, 2]
    prompt_logprobs = LogprobsTensors.empty_cpu(1, 2)
    output = ModelRunnerOutput(
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        logprobs=LogprobsLists(
            logprob_token_ids=_make_readonly(
                np.array([[1, 2], [3, 4]], dtype=np.int32)
            ),
            logprobs=_make_readonly(
                np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
            ),
            sampled_token_ranks=_make_readonly(np.array([1, 2], dtype=np.int32)),
            cu_num_generated_tokens=cu_num_generated_tokens,
        ),
        prompt_logprobs_dict={"req-0": prompt_logprobs},
    )

    original_logprobs = output.logprobs
    assert original_logprobs is not None

    detach_zero_copy_from_model_runner_output(output)

    detached_logprobs = output.logprobs
    assert detached_logprobs is not None
    assert detached_logprobs is not original_logprobs
    assert (
        detached_logprobs.logprob_token_ids is not original_logprobs.logprob_token_ids
    )
    assert detached_logprobs.logprobs is not original_logprobs.logprobs
    assert (
        detached_logprobs.sampled_token_ranks
        is not original_logprobs.sampled_token_ranks
    )
    assert detached_logprobs.logprob_token_ids.flags.writeable
    assert detached_logprobs.logprobs.flags.writeable
    assert detached_logprobs.sampled_token_ranks.flags.writeable
    assert detached_logprobs.cu_num_generated_tokens is cu_num_generated_tokens
    assert output.prompt_logprobs_dict["req-0"] is prompt_logprobs


def test_detach_zero_copy_routed_experts_without_logprobs():
    output = ModelRunnerOutput(
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        routed_experts=RoutedExpertsLists(
            routing_data=_make_readonly(np.arange(12, dtype=np.int32).reshape(2, 3, 2)),
            slot_mapping=_make_readonly(np.array([7, 8], dtype=np.int64)),
        ),
    )
    original = output.routed_experts
    assert output.logprobs is None

    detach_zero_copy_from_model_runner_output(output)

    detached = output.routed_experts
    assert detached is not None
    assert detached is not original
    assert detached.routing_data is not original.routing_data
    assert detached.slot_mapping is not original.slot_mapping
    assert detached.routing_data.flags.writeable
    assert detached.slot_mapping.flags.writeable
    np.testing.assert_array_equal(detached.routing_data, original.routing_data)
    np.testing.assert_array_equal(detached.slot_mapping, original.slot_mapping)


class _FakeClock:
    """Deterministic stand-in for ``time.monotonic``."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakePlacementGroup:
    bundle_specs = [{"GPU": 1}]

    def ready(self):
        return "pg-ready-ref"


def test_wait_until_pg_ready_does_not_wait_past_the_deadline(monkeypatch):
    """Before the cap, the final wait started at 1270s and blocked another
    1280s, so the loop returned at ~2550s for a nominal 1800s timeout."""
    clock = _FakeClock()
    waits: list[float] = []

    def fake_wait(refs, timeout):
        # The placement group never becomes ready.
        waits.append(timeout)
        clock.advance(timeout)
        return [], refs

    def fake_get(ref, timeout):
        raise ray_utils.ray.exceptions.GetTimeoutError()

    monkeypatch.setattr(ray_utils.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(ray_utils.ray, "wait", fake_wait)
    monkeypatch.setattr(ray_utils.ray, "get", fake_get)

    start = clock.now
    with pytest.raises(ValueError) as exc_info:
        ray_utils._wait_until_pg_ready(_FakePlacementGroup())

    elapsed = clock.now - start
    assert elapsed == ray_utils.PG_WAIT_TIMEOUT
    assert all(w > 0 for w in waits)
    # The final wait is the one that used to overshoot.
    assert waits[-1] == ray_utils.PG_WAIT_TIMEOUT - sum(waits[:-1])
    # The error reports what was actually waited, not the nominal constant.
    assert f"within {int(elapsed)} seconds" in str(exc_info.value)


def test_wait_until_pg_removed_does_not_sleep_past_the_deadline(monkeypatch):
    """The removal loop must also stop at the deadline instead of overshooting."""
    clock = _FakeClock()
    sleeps: list[float] = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)

    monkeypatch.setattr(ray_utils.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(ray_utils.time, "sleep", fake_sleep)
    monkeypatch.setattr(ray_utils.ray.util, "remove_placement_group", lambda pg: None)
    # The placement group is never actually removed.
    monkeypatch.setattr(
        ray_utils.ray.util, "get_current_placement_group", lambda: object()
    )

    start = clock.now
    ray_utils._wait_until_pg_removed(_FakePlacementGroup())

    elapsed = clock.now - start
    assert elapsed == ray_utils.PG_WAIT_TIMEOUT
    assert all(s >= 0 for s in sleeps)
