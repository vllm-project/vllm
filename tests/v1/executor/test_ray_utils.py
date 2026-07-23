# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.v1.executor.ray_utils import detach_zero_copy_from_model_runner_output
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, ModelRunnerOutput


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


def test_pg_timeout_default(monkeypatch):
    """VLLM_RAY_PG_TIMEOUT_S defaults to 1800."""
    monkeypatch.delenv("VLLM_RAY_PG_TIMEOUT_S", raising=False)
    from vllm.envs import environment_variables

    assert environment_variables["VLLM_RAY_PG_TIMEOUT_S"]() == 1800


def test_pg_timeout_non_default(monkeypatch):
    """VLLM_RAY_PG_TIMEOUT_S reads non-default integer values."""
    monkeypatch.setenv("VLLM_RAY_PG_TIMEOUT_S", "60")
    from vllm.envs import environment_variables

    assert environment_variables["VLLM_RAY_PG_TIMEOUT_S"]() == 60


@pytest.mark.parametrize("value", ["abc", "3.14"])
def test_pg_timeout_invalid_value_raises(monkeypatch, value):
    """Non-integer VLLM_RAY_PG_TIMEOUT_S raises ValueError."""
    monkeypatch.setenv("VLLM_RAY_PG_TIMEOUT_S", value)
    from vllm.envs import environment_variables

    with pytest.raises(ValueError, match="VLLM_RAY_PG_TIMEOUT_S"):
        environment_variables["VLLM_RAY_PG_TIMEOUT_S"]()


@pytest.mark.parametrize("value", ["0", "-1"])
def test_pg_timeout_zero_or_negative_raises(monkeypatch, value):
    """Zero or negative VLLM_RAY_PG_TIMEOUT_S raises ValueError."""
    monkeypatch.setenv("VLLM_RAY_PG_TIMEOUT_S", value)
    from vllm.envs import environment_variables

    with pytest.raises(ValueError, match="must be > 0"):
        environment_variables["VLLM_RAY_PG_TIMEOUT_S"]()


class TestPlacementGroupTimeout:
    """Behavioral tests for placement-group timeout enforcement."""

    def test_ray_wait_timeout_capped_at_deadline(self, monkeypatch, mocker):
        """ray.wait timeout must not exceed the configured deadline."""
        import time as _time_module

        from vllm.v1.executor import ray_utils

        monkeypatch.setenv("VLLM_RAY_PG_TIMEOUT_S", "1")
        monkeypatch.setattr(ray_utils.envs, "VLLM_RAY_PG_TIMEOUT_S", 1)

        t0 = _time_module.monotonic()
        monkeypatch.setattr(
            ray_utils.time,
            "monotonic",
            mocker.MagicMock(side_effect=[t0, t0, t0 + 0.5, t0 + 10]),
        )

        pg = mocker.MagicMock()
        pg.bundle_specs = [{"GPU": 1}]
        pg_ready_ref = mocker.MagicMock()
        pg.ready.return_value = pg_ready_ref

        mock_ray = mocker.patch.object(ray_utils, "ray", autospec=False)
        mock_ray.wait.return_value = ([], None)

        class _GetTimeoutError(Exception):
            pass

        mock_ray.exceptions.GetTimeoutError = _GetTimeoutError
        mock_ray.get.side_effect = _GetTimeoutError

        with pytest.raises(ValueError, match="1 seconds"):
            ray_utils._wait_until_pg_ready(pg)

        call_timeout = mock_ray.wait.call_args.kwargs["timeout"]
        assert call_timeout <= 1.0, (
            f"ray.wait timeout {call_timeout} exceeded configured deadline of 1 second"
        )

    def test_removal_sleep_capped_at_deadline(self, monkeypatch, mocker):
        """time.sleep in removal path must not exceed remaining time.

        When get_current_placement_group + logging consume enough time
        to exhaust the deadline, the subsequent sleep must be 0 — not
        the stale remaining value captured at the top of the loop.
        """
        import time as _time_module

        from vllm.v1.executor import ray_utils

        monkeypatch.setenv("VLLM_RAY_PG_TIMEOUT_S", "2")
        monkeypatch.setattr(ray_utils.envs, "VLLM_RAY_PG_TIMEOUT_S", 2)

        t0 = _time_module.monotonic()
        monkeypatch.setattr(
            ray_utils.time,
            "monotonic",
            mocker.MagicMock(
                side_effect=[t0, t0, t0 + 1.5, t0 + 3, t0 + 5],
            ),
        )

        pg = mocker.MagicMock()
        mock_ray_util = mocker.patch.object(ray_utils, "ray")
        mock_ray_util.util.get_current_placement_group.return_value = pg

        mock_sleep = mocker.patch.object(ray_utils.time, "sleep")

        ray_utils._wait_until_pg_removed(pg)

        assert len(mock_sleep.call_args_list) == 1, (
            f"Expected exactly 1 sleep call, got {len(mock_sleep.call_args_list)}"
        )
        sleep_time = mock_sleep.call_args_list[0][0][0]
        assert sleep_time == 0, (
            f"Expected sleep(0) when deadline already elapsed, got sleep({sleep_time})"
        )
