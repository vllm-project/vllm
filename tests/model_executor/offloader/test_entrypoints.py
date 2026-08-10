# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from types import SimpleNamespace

import torch

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper


class _InlineThread:
    def __init__(self, target, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)

    def join(self):
        return None


class _DummyBarrier:
    def wait(self):
        return None


class _DummyEvent:
    def __init__(self):
        self.was_set = False

    def set(self):
        self.was_set = True


class _DummyContext:
    def __init__(self, context_id: int):
        self.id = context_id
        self.cpu_wait_event = _DummyEvent()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeOffloader:
    def __init__(self):
        self.sync_calls = 0
        self.join_calls = 0
        self.reset_calls = 0

    def sync_prev_onload(self):
        self.sync_calls += 1

    def join_after_forward(self):
        self.join_calls += 1

    def reset_runtime_state(self):
        self.reset_calls += 1


def test_ubatch_wrapper_syncs_offloader_before_running_ubatches(monkeypatch):
    offloader = _FakeOffloader()
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_ubatch_wrapper.get_offloader", lambda: offloader
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_ubatch_wrapper.threading.Thread", _InlineThread
    )

    ubatch_metadata = [
        SimpleNamespace(
            context=_DummyContext(0),
            input_ids=torch.tensor([1, 2], dtype=torch.int64),
            positions=torch.tensor([0, 1], dtype=torch.int64),
            intermediate_tensors=None,
            inputs_embeds=None,
        )
    ]

    def model(**kwargs):
        return kwargs["input_ids"].to(torch.float32).unsqueeze(-1)

    wrapper = SimpleNamespace(ready_barrier=_DummyBarrier())
    output = UBatchWrapper._run_ubatches(wrapper, ubatch_metadata, model)

    assert offloader.sync_calls == 1
    assert ubatch_metadata[0].context.cpu_wait_event.was_set
    assert torch.equal(output, torch.tensor([[1.0], [2.0]]))


def test_cudagraph_capture_syncs_offloader_before_warmup_and_capture(monkeypatch):
    offloader = _FakeOffloader()
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.get_offloader", lambda: offloader
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.graph_capture",
        lambda device=None: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.is_global_first_rank", lambda: False
    )
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: object())
    monkeypatch.setattr(
        torch.cuda,
        "graph",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )

    desc = cudagraph_utils.BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=1,
        num_reqs=1,
    )
    calls: list[CUDAGraphMode] = []

    def create_forward_fn(batch_desc, warmup: bool):
        assert batch_desc == desc

        def forward_fn(mode: CUDAGraphMode):
            calls.append(mode)

        return forward_fn

    manager = SimpleNamespace(
        device=torch.device("cpu"),
        _capture_descs={CUDAGraphMode.FULL: [desc]},
        pool=None,
        graphs={},
        _graphs_captured=False,
    )

    cudagraph_utils.CudaGraphManager.capture(manager, create_forward_fn)

    assert offloader.reset_calls == 1
    assert offloader.sync_calls == 1
    assert offloader.join_calls == 1
    assert calls == [CUDAGraphMode.NONE, CUDAGraphMode.NONE]
    assert desc in manager.graphs
