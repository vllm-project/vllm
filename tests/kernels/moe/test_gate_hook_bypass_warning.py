# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate fusion must not silently bypass forward hooks without saying so.

Run `pytest tests/kernels/moe/test_gate_hook_bypass_warning.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner import moe_runner as moe_runner_mod
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


@pytest.fixture
def warnings(monkeypatch):
    """Capture warning_once calls directly.

    The vllm logger is configured with propagate=False, so caplog cannot see
    it, and warning_once is lru_cached, so a second identical call anywhere in
    the process is dropped. Recording the call is both simpler and immune to
    ordering between tests.
    """
    recorded: list[str] = []

    def record(msg, *args, **kwargs):
        recorded.append(msg % args if args else msg)

    monkeypatch.setattr(moe_runner_mod.logger, "warning_once", record)
    return recorded


def _runner_with_gate(gate: torch.nn.Module | None) -> MoERunner:
    """A MoERunner stub carrying only what the warning helper reads.

    nn.Module.__init__ must run before assigning a submodule, otherwise
    nn.Module.__setattr__ raises "cannot assign module before
    Module.__init__() call".
    """
    runner = MoERunner.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.gate = gate
    runner.layer_name = "model.layers.0.mlp"
    return runner


def test_warns_when_a_forward_hook_would_be_bypassed(warnings):
    gate = torch.nn.Linear(8, 4)
    gate.register_forward_hook(lambda mod, inp, out: None)

    _runner_with_gate(gate)._warn_if_gate_hooks_are_bypassed()

    assert len(warnings) == 1
    assert "never fire" in warnings[0]
    assert "model.layers.0.mlp" in warnings[0]


def test_warns_for_forward_pre_hooks_too(warnings):
    gate = torch.nn.Linear(8, 4)
    gate.register_forward_pre_hook(lambda mod, inp: None)

    _runner_with_gate(gate)._warn_if_gate_hooks_are_bypassed()

    assert len(warnings) == 1
    assert "never fire" in warnings[0]


def test_silent_when_no_hooks_are_registered(warnings):
    _runner_with_gate(torch.nn.Linear(8, 4))._warn_if_gate_hooks_are_bypassed()
    assert warnings == []


def test_silent_when_there_is_no_gate(warnings):
    _runner_with_gate(None)._warn_if_gate_hooks_are_bypassed()
    assert warnings == []
