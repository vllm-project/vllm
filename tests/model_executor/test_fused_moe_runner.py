# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner_module
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


def _make_runner(
    *,
    dp_size: int,
    use_ep: bool,
) -> MoERunner:
    runner = MoERunner.__new__(MoERunner)
    runner.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(dp_size=dp_size, use_ep=use_ep)
    )
    runner._shared_experts = None
    runner.enable_dbo = False
    runner.layer_name = "test_layer"
    return runner


@pytest.fixture(autouse=True)
def _cuda_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        moe_runner_module,
        "current_platform",
        SimpleNamespace(is_cuda_alike=lambda: True),
    )
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()


def test_unwrapped_rejects_native_dp_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_FUSED_MOE_WRAP_MODE", "unwrapped")

    runner = _make_runner(dp_size=2, use_ep=True)

    with pytest.raises(NotImplementedError, match=r"native DP\+EP is enabled"):
        runner._determine_forward_mode()


def test_unwrapped_allows_no_dp_ep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_FUSED_MOE_WRAP_MODE", "unwrapped")

    runner = _make_runner(dp_size=1, use_ep=False)

    assert runner._determine_forward_mode() == "unwrapped"
