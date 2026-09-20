# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate fusion must not silently bypass forward hooks without saying so.

Run `pytest tests/kernels/moe/test_gate_hook_bypass_warning.py`.
"""

import torch

from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


def _runner_with_gate(gate: torch.nn.Module | None) -> MoERunner:
    """A MoERunner stub carrying only what the warning helper reads."""
    runner = MoERunner.__new__(MoERunner)
    runner.gate = gate
    runner.layer_name = "model.layers.0.mlp"
    return runner


def test_warns_when_a_forward_hook_would_be_bypassed(caplog):
    gate = torch.nn.Linear(8, 4)
    gate.register_forward_hook(lambda mod, inp, out: None)

    _runner_with_gate(gate)._warn_if_gate_hooks_are_bypassed()

    assert "never fire" in caplog.text
    assert "model.layers.0.mlp" in caplog.text


def test_warns_for_forward_pre_hooks_too(caplog):
    gate = torch.nn.Linear(8, 4)
    gate.register_forward_pre_hook(lambda mod, inp: None)

    _runner_with_gate(gate)._warn_if_gate_hooks_are_bypassed()

    assert "never fire" in caplog.text


def test_silent_when_no_hooks_are_registered(caplog):
    _runner_with_gate(torch.nn.Linear(8, 4))._warn_if_gate_hooks_are_bypassed()
    assert "never fire" not in caplog.text


def test_silent_when_there_is_no_gate(caplog):
    _runner_with_gate(None)._warn_if_gate_hooks_are_bypassed()
    assert "never fire" not in caplog.text
