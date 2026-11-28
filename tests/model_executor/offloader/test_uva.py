# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

import vllm.envs as envs
from vllm.model_executor.offloader import UVAOffloader

pytestmark = pytest.mark.skip_global_cleanup


def _setup(monkeypatch, events: list[str]) -> None:
    monkeypatch.setenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", "1")
    monkeypatch.setenv("VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY", "1")
    envs.disable_envs_cache()
    monkeypatch.setattr(
        torch.accelerator, "empty_cache", lambda: events.append("release")
    )


def _make_module(name: str = "weight", device: str = "cuda") -> nn.Module:
    module = nn.Module()
    module.register_parameter(name, nn.Parameter(torch.empty(64, 64, device=device)))
    return module


def test_offload_releases_blocks_before_next_module(monkeypatch) -> None:
    """Blocks are released per module, before the next module is built."""
    events: list[str] = []
    _setup(monkeypatch, events)
    try:
        with FakeTensorMode():

            def layers(n: int):
                for i in range(n):
                    events.append(f"build{i}")
                    yield _make_module()

            offloader = UVAOffloader(cpu_offload_max_bytes=10 * 2**30)
            offloader.wrap_modules(layers(3))

        assert events == [
            "build0",
            "release",
            "build1",
            "release",
            "build2",
            "release",
        ]
        assert offloader.cpu_offload_bytes > 0
    finally:
        envs.disable_envs_cache()


def test_no_release_when_nothing_offloaded(monkeypatch) -> None:
    """A module that offloaded nothing must not flush the device cache."""
    events: list[str] = []
    _setup(monkeypatch, events)
    try:
        with FakeTensorMode():
            offloader = UVAOffloader(
                cpu_offload_max_bytes=10 * 2**30, cpu_offload_params={"visual"}
            )
            offloader.wrap_modules(_make_module() for _ in range(4))

        assert events == []
        assert offloader.cpu_offload_bytes == 0
    finally:
        envs.disable_envs_cache()


def test_no_release_for_meta_weights(monkeypatch) -> None:
    """Skipped meta weights (online quantization) must not flush the device cache."""
    events: list[str] = []
    _setup(monkeypatch, events)
    try:
        offloader = UVAOffloader(cpu_offload_max_bytes=10 * 2**30)
        offloader.wrap_modules(_make_module(device="meta") for _ in range(2))

        assert events == []
        assert offloader.cpu_offload_bytes == 0
    finally:
        envs.disable_envs_cache()
