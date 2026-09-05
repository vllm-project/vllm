# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collective-safety tests for the Moondream3 MoE layer.

The MoE forward may fall back from the fused kernel to a Python expert loop.
That decision is made per rank from a rank-local exception, so every path has
to issue exactly one all-reduce. A path that issues two (or none) desyncs the
tensor-parallel group and wedges it for the rest of the run.
"""

import pytest
import torch

from tests.utils import ensure_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.models import moondream3
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

pytestmark = pytest.mark.cpu_test

HIDDEN_SIZE = 8
EXPERT_INNER_DIM = 16
NUM_EXPERTS = 4
EXPERTS_PER_TOKEN = 2
NUM_TOKENS = 3


class _StubGate(torch.nn.Module):
    """Router stand-in, so the test does not depend on platform-specific
    ``ReplicatedLinear`` weight post-processing."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, num_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.linear(x), None


class _CountingAllReduce:
    """Stand-in for tensor_model_parallel_all_reduce that counts calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x


@pytest.fixture
def tp1_cpu():
    """A single-rank CPU tensor-parallel group, enough to build the layer."""
    update_environment_variables(
        {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(get_open_port()),
        }
    )
    init_distributed_environment(backend="gloo")
    with ensure_current_vllm_config():
        initialize_model_parallel(tensor_model_parallel_size=1)
    yield
    destroy_model_parallel()
    destroy_distributed_environment()


@pytest.fixture
def moe(tp1_cpu) -> moondream3.Moondream3TextMoE:
    """A single-rank MoE layer with deterministic weights."""
    layer = moondream3.Moondream3TextMoE(
        hidden_size=HIDDEN_SIZE,
        expert_inner_dim=EXPERT_INNER_DIM,
        num_experts=NUM_EXPERTS,
        experts_per_token=EXPERTS_PER_TOKEN,
    )
    layer.gate = _StubGate(HIDDEN_SIZE, NUM_EXPERTS)
    torch.nn.init.normal_(layer.fc1_weight, std=0.02)
    torch.nn.init.normal_(layer.fc2_weight, std=0.02)
    return layer


@pytest.fixture
def all_reduce(monkeypatch: pytest.MonkeyPatch) -> _CountingAllReduce:
    counter = _CountingAllReduce()
    monkeypatch.setattr(moondream3, "tensor_model_parallel_all_reduce", counter)
    return counter


def test_expert_loop_all_reduces_once(moe, all_reduce):
    """The Python expert loop issues exactly one all-reduce."""
    moe._use_fused_moe = False
    x = torch.randn(NUM_TOKENS, HIDDEN_SIZE)

    out = moe(x)

    assert out.shape == x.shape
    assert all_reduce.calls == 1


def test_fused_path_all_reduces_once(moe, all_reduce, monkeypatch):
    """The fused kernel path issues exactly one all-reduce."""
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(
        moondream3,
        "fused_experts",
        lambda **kwargs: torch.zeros_like(kwargs["hidden_states"]),
    )
    x = torch.randn(NUM_TOKENS, HIDDEN_SIZE)

    out = moe(x)

    assert out.shape == x.shape
    assert all_reduce.calls == 1


def test_all_reduce_failure_is_not_retried(moe, monkeypatch):
    """A failing all-reduce must propagate, not trigger a second collective.

    Regression test. The all-reduce used to sit inside the ``try`` guarding
    the fused kernel, so a ``RuntimeError`` raised by the collective itself
    was caught, and this rank went on to run the fallback loop and all-reduce
    a *second* time -- two collectives on one rank against one on each of its
    peers, which desyncs the tensor-parallel group and wedges it for good.
    """
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(
        moondream3,
        "fused_experts",
        lambda **kwargs: torch.zeros_like(kwargs["hidden_states"]),
    )
    calls = []

    def _failing_all_reduce(x: torch.Tensor) -> torch.Tensor:
        calls.append(x)
        raise RuntimeError("collective failed")

    monkeypatch.setattr(
        moondream3, "tensor_model_parallel_all_reduce", _failing_all_reduce
    )
    x = torch.randn(NUM_TOKENS, HIDDEN_SIZE)

    with pytest.raises(RuntimeError, match="collective failed"):
        moe(x)

    assert len(calls) == 1, (
        "the failed all-reduce was retried on the fallback path; peers issued "
        "one collective and this rank issued two"
    )


@pytest.mark.parametrize("exc", [RuntimeError("boom"), NotImplementedError()])
def test_fallback_after_fused_failure_all_reduces_once(
    moe, all_reduce, monkeypatch, exc
):
    """A rank whose fused kernel fails still all-reduces exactly once."""
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    def _raise(**kwargs):
        raise exc

    monkeypatch.setattr(moondream3, "fused_experts", _raise)
    x = torch.randn(NUM_TOKENS, HIDDEN_SIZE)

    out = moe(x)

    assert out.shape == x.shape
    assert all_reduce.calls == 1
    # The fused path is disabled for subsequent forwards...
    assert moe._use_fused_moe is False
    # ...and those still all-reduce exactly once each.
    moe(x)
    assert all_reduce.calls == 2
