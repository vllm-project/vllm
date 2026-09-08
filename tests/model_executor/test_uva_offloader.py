# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.model_loader.utils import device_loading_context
from vllm.model_executor.offloader.uva import UVAOffloader


class _CopyRecorder:
    def __init__(self):
        self.calls: list[tuple[torch.device, bool]] = []

    def to(self, device: torch.device, non_blocking: bool):
        self.calls.append((device, non_blocking))
        return self


class _StateDictRaises(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.register_buffer("persistent_buffer", torch.ones(1))
        self.register_buffer("temporary_buffer", torch.ones(1), persistent=False)

    def state_dict(self, *args, **kwargs):
        raise AssertionError("state_dict should not be called")


class _RaisesOnce(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.raise_next = True

    def forward(self, x):
        if self.raise_next:
            self.raise_next = False
            raise RuntimeError("expected failure")
        return x @ self.weight


class _BlockingLinear(nn.Linear):
    def __init__(self):
        super().__init__(2, 2, bias=False)
        self.first_entered = threading.Event()
        self.second_entered = threading.Event()
        self.release = threading.Event()
        self._calls_lock = threading.Lock()
        self._calls = 0

    def forward(self, x):
        with self._calls_lock:
            self._calls += 1
            if self._calls == 1:
                self.first_entered.set()
            elif self._calls == 2:
                self.second_entered.set()
        assert self.release.wait(timeout=5)
        return nn.functional.linear(x, self.weight)


class _TiedWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.tied_weight = self.weight

    def forward(self, x):
        return x @ self.weight + x @ self.tied_weight


class _MutableState(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = nn.Linear(2, 2, bias=False)
        self.register_buffer("scale", torch.ones(()))

    def forward(self, x):
        output = self.child(x) * self.scale
        if hasattr(self, "extra"):
            output = output + self.extra(x)
        return output


class _BufferState(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.register_buffer("persistent", torch.ones(()))
        self.register_buffer("temporary", torch.ones(()), persistent=False)

    def forward(self, x):
        return x @ self.weight * self.persistent * self.temporary


class _Recursive(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))

    def forward(self, x, recurse=True):
        if recurse:
            return self(x, recurse=False)
        return x @ self.weight

    def state_dict(self, *args, **kwargs):
        raise AssertionError("state_dict should not be called")


def test_named_state_tensors_does_not_call_state_dict():
    module = _StateDictRaises()

    state = dict(UVAOffloader._named_state_tensors(module))

    assert set(state) == {"weight", "persistent_buffer"}


def test_move_state_to_device_uses_requested_non_blocking():
    tensor = _CopyRecorder()
    device = torch.device("cuda")

    state = UVAOffloader._move_state_to_device(
        [("weight", tensor)],
        device,
        non_blocking=False,
    )

    assert state == {"weight": tensor}
    assert tensor.calls == [(device, False)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fallback_wrapper_is_restored_after_forward_error():
    module = _RaisesOnce().cuda()
    offloader = UVAOffloader(module.weight.nbytes)
    offloader.pin_memory = False
    offloader.uva_offloading = False
    offloader._maybe_offload_to_cpu(module)

    x = torch.ones(1, 2, device="cuda")
    with pytest.raises(RuntimeError, match="expected failure"):
        module(x)

    torch.testing.assert_close(module(x), torch.full((1, 2), 2.0, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fallback_wrapper_is_fullgraph_traceable():
    module = _Recursive().cuda()
    offloader = UVAOffloader(module.weight.nbytes)
    offloader.pin_memory = False
    offloader.uva_offloading = False
    offloader._maybe_offload_to_cpu(module)

    x = torch.ones(1, 2, device="cuda")
    compiled = torch.compile(module, backend="eager", fullgraph=True)

    torch.testing.assert_close(compiled(x), torch.full((1, 2), 2.0, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fallback_wrapper_keeps_replaced_parameter_offloaded():
    module = nn.Linear(2, 2, bias=False).cuda()
    offloader = UVAOffloader(module.weight.nbytes)
    offloader.pin_memory = False
    offloader.uva_offloading = False
    offloader._maybe_offload_to_cpu(module)

    with device_loading_context(module, torch.device("cuda")):
        module.weight = nn.Parameter(torch.full_like(module.weight, 3.0))

    assert module.weight.device.type == "cpu"
    x = torch.ones(1, 2, device="cuda")
    torch.testing.assert_close(module(x), torch.full((1, 2), 6.0, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fallback_wrapper_supports_tied_parameters():
    module = _TiedWeights().cuda()
    offloader = UVAOffloader(module.weight.nbytes)
    offloader.pin_memory = False
    offloader.uva_offloading = False
    offloader._maybe_offload_to_cpu(module)

    assert module.weight is module.tied_weight
    x = torch.ones(1, 2, device="cuda")
    torch.testing.assert_close(module(x), torch.full((1, 2), 4.0, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fallback_wrapper_partial_offload_runs_hooks_once_and_accepts_kwargs():
    module = nn.Linear(2, 2).cuda()
    weight = module.weight.detach().clone()
    bias = module.bias.detach().clone()
    hook_calls = []
    module.register_forward_pre_hook(
        lambda hooked_module, _args: hook_calls.append((hooked_module, "local pre"))
    )
    module.register_forward_hook(
        lambda hooked_module, _args, _output: hook_calls.append(
            (hooked_module, "local post")
        )
    )

    offloader = UVAOffloader(module.weight.nbytes)
    offloader.pin_memory = False
    offloader.uva_offloading = False
    offloader._maybe_offload_to_cpu(module)

    assert module.weight.device.type == "cpu"
    assert module.bias.device.type == "cuda"
    x = torch.ones(1, 2, device="cuda")
    torch.testing.assert_close(module(input=x), nn.functional.linear(x, weight, bias))
    assert hook_calls == [(module, "local pre"), (module, "local post")]
    hook_calls.clear()

    def pre_hook(hooked_module, args):
        hook_calls.append((hooked_module, "global pre"))
        return (args[0] * 2,)

    def post_hook(hooked_module, _args, output):
        hook_calls.append((hooked_module, "global post"))
        return output + 1

    pre_handle = nn.modules.module.register_module_forward_pre_hook(pre_hook)
    post_handle = nn.modules.module.register_module_forward_hook(post_hook)
    try:
        output = module(x)
    finally:
        pre_handle.remove()
        post_handle.remove()

    expected = nn.functional.linear(x * 2, weight, bias) + 1
    torch.testing.assert_close(output, expected)
    assert hook_calls == [
        (module, "global pre"),
        (module, "local pre"),
        (module, "global post"),
        (module, "local post"),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fallback_wrapper_uses_replaced_and_added_module_state():
    module = _MutableState().cuda()
    offloader = UVAOffloader(module.child.weight.nbytes)
    offloader.pin_memory = False
    offloader.uva_offloading = False
    offloader._maybe_offload_to_cpu(module)

    with device_loading_context(module, torch.device("cuda")):
        module.child = nn.Linear(2, 2, bias=False, device="cuda")
        module.child.weight.data.fill_(2)
        module.scale = torch.full((), 3.0, device="cuda")
        module.extra = nn.Linear(2, 2, bias=False, device="cuda")
        module.extra.weight.data.fill_(4)

    assert module.child.weight.device.type == "cpu"
    assert module.extra.weight.device.type == "cuda"
    x = torch.ones(1, 2, device="cuda")
    torch.testing.assert_close(module(x), torch.full((1, 2), 20.0, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fallback_wrapper_uses_live_non_persistent_buffer():
    module = _BufferState().cuda()
    offloader = UVAOffloader(module.weight.nbytes)
    offloader.pin_memory = False
    offloader.uva_offloading = False
    offloader._maybe_offload_to_cpu(module)

    module.persistent = torch.full((), 2.0, device="cpu")
    module.temporary = torch.full((), 3.0, device="cuda")

    x = torch.ones(1, 2, device="cuda")
    torch.testing.assert_close(module(x), torch.full((1, 2), 12.0, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fallback_wrapper_serializes_concurrent_calls():
    module = _BlockingLinear().cuda()
    module.weight.data.fill_(1)
    offloader = UVAOffloader(module.weight.nbytes)
    offloader.pin_memory = False
    offloader.uva_offloading = False
    offloader._maybe_offload_to_cpu(module)

    x = torch.ones(1, 2, device="cuda")
    outputs = []
    threads = [
        threading.Thread(target=lambda: outputs.append(module(x))) for _ in range(2)
    ]
    for thread in threads:
        thread.start()

    assert module.first_entered.wait(timeout=5)
    assert not module.second_entered.wait(timeout=0.1)
    module.release.set()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert module.second_entered.is_set()
    assert len(outputs) == 2
    for output in outputs:
        torch.testing.assert_close(output, torch.full((1, 2), 2.0, device="cuda"))
