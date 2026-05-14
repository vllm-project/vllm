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
    supports_unwrapped_forward: bool = True,
    shared_experts: object | None = None,
    enable_dbo: bool = False,
) -> MoERunner:
    runner = MoERunner.__new__(MoERunner)
    runner.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(dp_size=dp_size, use_ep=use_ep)
    )
    runner._quant_method = SimpleNamespace(
        method_name="TestMoEMethod",
        supports_unwrapped_forward=supports_unwrapped_forward,
    )
    runner._shared_experts = shared_experts
    runner.enable_dbo = enable_dbo
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


def test_unwrapped_rejects_unsupported_quant_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_FUSED_MOE_WRAP_MODE", "unwrapped")

    runner = _make_runner(
        dp_size=1,
        use_ep=False,
        supports_unwrapped_forward=False,
    )

    with pytest.raises(NotImplementedError, match="TestMoEMethod is not supported"):
        runner._determine_forward_mode()


def test_unwrapped_reports_all_blockers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_FUSED_MOE_WRAP_MODE", "unwrapped")

    runner = _make_runner(
        dp_size=2,
        use_ep=True,
        supports_unwrapped_forward=False,
        shared_experts=object(),
        enable_dbo=True,
    )

    with pytest.raises(NotImplementedError) as exc_info:
        runner._determine_forward_mode()

    message = str(exc_info.value)
    assert "shared experts are enabled" in message
    assert "DBO is enabled" in message
    assert "TestMoEMethod is not supported" in message
    assert "native DP+EP is enabled" in message


def test_wrapped_allows_otherwise_unsupported_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_FUSED_MOE_WRAP_MODE", "wrapped")

    runner = _make_runner(
        dp_size=2,
        use_ep=True,
        supports_unwrapped_forward=False,
        shared_experts=object(),
        enable_dbo=True,
    )

    assert runner._determine_forward_mode() == "wrapped"


def test_unwrapped_allows_no_dp_ep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_FUSED_MOE_WRAP_MODE", "unwrapped")

    runner = _make_runner(dp_size=1, use_ep=False)

    assert runner._determine_forward_mode() == "unwrapped"
