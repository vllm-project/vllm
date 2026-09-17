# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The modular kernel must reserve workspace for the configured worst-case
number of received tokens, independent of how the profile run routed tokens.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
import vllm.v1.worker.workspace as workspace
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig


class _FakeMoEConfig:
    max_num_global_tokens = FusedMoEConfig.max_num_global_tokens
    max_num_recv_tokens_per_rank = FusedMoEConfig.max_num_recv_tokens_per_rank

    def __init__(
        self, fraction: float, dp_size: int, sp_size: int, max_num_tokens: int
    ):
        self.moe_parallel_config = SimpleNamespace(dp_size=dp_size, use_ep=True)
        self.ep_max_recv_tokens_fraction = fraction
        self.dp_size = dp_size
        self.sp_size = sp_size
        self.max_num_tokens = max_num_tokens


class _FakeExperts:
    def __init__(self, moe_config: _FakeMoEConfig):
        self.moe_config = moe_config

    def workspace_dtype(self, dtype: torch.dtype) -> torch.dtype:
        return dtype

    def workspace_shapes(self, M, N, K, *args):
        return (M, K), (M, N), (M, K)


class _FakePrepareFinalize:
    def __init__(self, routing_dependent: bool):
        self.routing_dependent = routing_dependent

    def recv_tokens_depend_on_routing(self) -> bool:
        return self.routing_dependent


def _make_kernel(monkeypatch, routing_dependent: bool):
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(mk, "current_workspace_manager", lambda: manager)
    kernel = mk.FusedMoEKernelModularImpl(
        _FakePrepareFinalize(routing_dependent),  # type: ignore[arg-type]
        _FakeExperts(_FakeMoEConfig(0.5, dp_size=4, sp_size=1, max_num_tokens=100)),  # type: ignore[arg-type]
    )
    return kernel, manager


def _reserve(kernel, M: int) -> None:
    kernel._maybe_reserve_worst_case_buffers(
        torch.float32,
        torch.device("cpu"),
        M,
        N=4,
        K=8,
        top_k=2,
        global_num_experts=8,
        local_num_experts=2,
        activation=MoEActivation.SILU,
    )


@pytest.mark.parametrize(
    ("fraction", "dp_size", "sp_size", "max_num_tokens", "expected"),
    [
        (1.0, 2, 1, 8192, 16384),
        (0.5, 4, 1, 100, 200),
        # SP pads each DP rank's 8191 tokens to 8192 before sharding.
        (1.0, 2, 4, 8191, 16384),
        (0.25, 2, 4, 8191, 4096),
        (0.001, 1, 1, 10, 1),
    ],
)
def test_worst_case_recv_tokens(fraction, dp_size, sp_size, max_num_tokens, expected):
    config = _FakeMoEConfig(fraction, dp_size, sp_size, max_num_tokens)
    assert config.max_num_recv_tokens_per_rank == expected


def test_reserves_worst_case_even_when_no_tokens_received(monkeypatch):
    kernel, manager = _make_kernel(monkeypatch, routing_dependent=True)
    # ceil(0.5 * dp_size(4) * max_num_tokens(100))
    assert kernel.max_num_recv_tokens == 200

    _reserve(kernel, M=0)

    ws = manager._current_workspaces[0]
    assert ws is not None
    # workspace13/output: 200 * 8 float32, workspace2: 200 * 4 float32.
    assert ws.numel() >= 200 * 8 * 4 + 200 * 4 * 4


def test_exceeding_reservation_after_lock_raises(monkeypatch):
    kernel, manager = _make_kernel(monkeypatch, routing_dependent=True)
    _reserve(kernel, M=0)
    manager.lock()

    _reserve(kernel, M=200)
    with pytest.raises(RuntimeError, match="ep-max-recv-tokens-fraction"):
        _reserve(kernel, M=201)


def test_fixed_size_dispatchers_do_not_reserve(monkeypatch):
    kernel, manager = _make_kernel(monkeypatch, routing_dependent=False)
    assert kernel.max_num_recv_tokens is None

    _reserve(kernel, M=0)

    assert manager._current_workspaces[0] is None
