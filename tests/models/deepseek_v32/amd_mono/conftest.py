# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures of the GLM-5.2 MonoKernel (ROCm, mono/) CPU tests: MonoLive /
Glm5MonoDecode are built bare (``__new__``) around fake kernel ops (no GPU, no FlyDSL)."""

import pytest
import torch


@pytest.fixture
def no_device_sync(monkeypatch):
    """The live-mode health check synchronizes the device; on CPU tensors that is a
    no-op."""
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def failstop_latch(monkeypatch):
    """Each test starts with no latched output-rank fail-stop, and a fail-stop does not
    arm the exit watchdog (it would end the test process): returns the arm calls."""
    from vllm.models.deepseek_v32.amd.mono import guards

    monkeypatch.setitem(guards._FAILED, "msg", None)
    armed: list = []
    monkeypatch.setattr(guards, "_arm_exit_watchdog", lambda *a: armed.append(a))
    return armed
