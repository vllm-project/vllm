# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
import torch.nn as nn

import vllm.model_executor.offloader.prefetch_ops  # noqa: F401
from vllm.model_executor.offloader import base as offloader_base
from vllm.model_executor.offloader.base import BaseOffloader
from vllm.model_executor.offloader.prefetch import PrefetchOffloader
from vllm.platforms import current_platform


def _registered_device() -> str:
    """Match the dispatch_key the prefetch custom ops were registered for.

    `direct_register_custom_op` registers against `current_platform.dispatch_key`.
    When platform detection falls back (e.g. minimal test envs) that key may
    be "CPU" even on a CUDA host, so tests must pick the matching device or
    the op dispatch errors out.
    """
    return "cuda" if current_platform.dispatch_key == "CUDA" else "cpu"


class _TwoArgModule(nn.Module):
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        return hidden_states + 1


class _TupleOutputModule(nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        aux = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return aux, hidden_states + 1


class _SingleTensorModule(nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        return hidden_states * 2


class _KeywordHiddenStatesModule(nn.Module):
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        return hidden_states + positions.to(hidden_states.dtype).unsqueeze(-1)


class _RecordingOffloader(BaseOffloader):
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def wrap_modules(self, modules_generator):
        return list(modules_generator)

    def _wait_for_layer(self, layer_idx: int) -> None:
        self.calls.append(("wait", layer_idx))

    def _start_prefetch(
        self,
        layer_idx: int,
        is_tail_prefetch: bool = False,
    ) -> None:
        self.calls.append(("start", layer_idx))


class _NoNextPrefetchRuntime:
    def prefetch_after(self, index: int):
        return None


class _StaticNextPrefetchRuntime:
    def __init__(self, unit_idx: int):
        self.unit_idx = unit_idx

    def prefetch_after(self, index: int):
        return SimpleNamespace(unit_idx=self.unit_idx)


def test_prefetch_hook_uses_hidden_states_not_positions(monkeypatch):
    captured: list[torch.Tensor] = []
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch.torch.cuda.Stream",
        lambda: object(),
    )
    offloader = PrefetchOffloader(group_size=1, num_in_group=1, prefetch_step=1)
    offloader.runtime = SimpleNamespace(prefetch_after=lambda index: None)

    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch.torch.ops.vllm.wait_prefetch",
        lambda tensor, index: captured.append(tensor),
    )

    module = _TwoArgModule()
    offloader._hook_module_forward(0, module)

    positions = torch.arange(4, dtype=torch.long)
    hidden_states = torch.randn(4, 8)
    module(positions, hidden_states)

    assert captured == [hidden_states]


def test_prefetch_hook_uses_tensor_output_not_aux_metadata(monkeypatch):
    captured: list[torch.Tensor] = []
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch.torch.cuda.Stream",
        lambda: object(),
    )
    offloader = PrefetchOffloader(group_size=1, num_in_group=1, prefetch_step=1)
    offloader.runtime = SimpleNamespace(
        prefetch_after=lambda index: SimpleNamespace(unit_idx=1)
    )

    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch.torch.ops.vllm.wait_prefetch",
        lambda tensor, index: None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch.torch.ops.vllm.start_prefetch",
        lambda tensor, index, is_tail: captured.append(tensor),
    )

    module = _TupleOutputModule()
    offloader._hook_module_forward(0, module)

    hidden_states = torch.randn(4, 8)
    module(hidden_states)

    assert len(captured) == 1
    assert captured[0].shape == hidden_states.shape
    assert captured[0].dtype == hidden_states.dtype


def test_prefetch_custom_ops_are_torch_compile_traceable():
    previous = offloader_base.get_offloader()
    offloader = _RecordingOffloader()
    offloader_base.set_offloader(offloader)

    try:

        def fn(x: torch.Tensor) -> torch.Tensor:
            torch.ops.vllm.wait_prefetch(x, 3)
            y = x + 1
            torch.ops.vllm.start_prefetch(y, 5, False)
            return y

        device = _registered_device()
        x = torch.ones(4, device=device)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)

        output = compiled_fn(x)

        assert torch.equal(output, x + 1)
        assert offloader.calls == [("wait", 3), ("start", 5)]
    finally:
        offloader_base.set_offloader(previous)


def test_prefetch_hooked_forward_is_torch_compile_traceable():
    previous = offloader_base.get_offloader()
    offloader = _RecordingOffloader()
    offloader_base.set_offloader(offloader)

    try:
        prefetch_offloader = PrefetchOffloader.__new__(PrefetchOffloader)
        prefetch_offloader.runtime = _StaticNextPrefetchRuntime(unit_idx=7)

        device = _registered_device()
        module = _SingleTensorModule().to(device)
        prefetch_offloader._hook_module_forward(2, module)
        compiled_module = torch.compile(module, backend="eager", fullgraph=True)

        x = torch.ones(4, device=device)
        output = compiled_module(x)
        second_output = compiled_module(x + 1)

        assert torch.equal(output, x * 2)
        assert torch.equal(second_output, (x + 1) * 2)
        assert offloader.calls == [
            ("wait", 2),
            ("start", 7),
            ("wait", 2),
            ("start", 7),
        ]
    finally:
        offloader_base.set_offloader(previous)


def test_prefetch_hooked_forward_compile_prefers_keyword_hidden_states():
    previous = offloader_base.get_offloader()
    offloader = _RecordingOffloader()
    offloader_base.set_offloader(offloader)

    try:
        prefetch_offloader = PrefetchOffloader.__new__(PrefetchOffloader)
        prefetch_offloader.runtime = _NoNextPrefetchRuntime()

        device = _registered_device()
        module = _KeywordHiddenStatesModule().to(device)
        prefetch_offloader._hook_module_forward(4, module)
        compiled_module = torch.compile(module, backend="eager", fullgraph=True)

        positions = torch.arange(4, device=device)
        hidden_states = torch.ones(4, 3, device=device)
        output = compiled_module(positions, hidden_states=hidden_states)

        expected = hidden_states + positions.to(hidden_states.dtype).unsqueeze(-1)
        assert torch.equal(output, expected)
        assert offloader.calls == [("wait", 4)]
    finally:
        offloader_base.set_offloader(previous)
