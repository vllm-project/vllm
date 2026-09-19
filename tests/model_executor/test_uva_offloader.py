# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

import vllm.envs as envs
from vllm.model_executor.offloader import UVAOffloader


def _setup(monkeypatch, events):
    monkeypatch.setenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", "1")
    monkeypatch.setenv("VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY", "1")
    envs.disable_envs_cache()
    monkeypatch.setattr(
        torch.accelerator, "empty_cache", lambda: events.append("release")
    )


def _make_module(name: str = "weight") -> nn.Module:
    module = nn.Module()
    module.register_parameter(name, nn.Parameter(torch.empty(64, 64, device="cuda")))
    return module


def test_offload_releases_blocks_before_next_module(monkeypatch):
    """The freed accelerator blocks must be released as each module is done.

    `_maybe_offload_to_cpu` rebinds `p.data` to a host view, which drops the
    accelerator storage but leaves the block in the caching allocator. Because
    `make_layers` hands `wrap_modules` a lazy generator, module N+1 is built
    after N is offloaded -- so releasing once after the loop is too late to
    stop reserved memory growing across the stack.
    """
    events: list[str] = []
    _setup(monkeypatch, events)
    try:
        with FakeTensorMode():

            def layers(n):
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


def test_no_release_when_nothing_offloaded(monkeypatch):
    """A module that offloaded nothing freed nothing, so it must not flush.

    Guards the over-correction of calling `empty_cache` unconditionally, which
    would purge the device cache once per module during load for no gain.
    """
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
