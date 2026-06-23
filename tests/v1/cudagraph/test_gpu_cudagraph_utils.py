# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)

pytestmark = pytest.mark.cpu_test


def test_full_cudagraph_capture_prewarms_fresh_capture_forward_fn(monkeypatch):
    manager = object.__new__(CudaGraphManager)
    manager.device = torch.device("cpu")
    manager._capture_descs = {
        CUDAGraphMode.FULL: [
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=8,
                num_reqs=8,
            )
        ]
    }
    manager.graphs = {}
    manager.pool = object()
    manager._graphs_captured = False
    manager.use_breakable_cg = False

    calls: list[tuple[str, CUDAGraphMode, bool]] = []
    in_graph = False
    capture_factory_calls = 0

    def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
        nonlocal capture_factory_calls
        if warmup:
            label = "warmup"
        else:
            capture_factory_calls += 1
            label = f"capture-{capture_factory_calls}"

        def forward_fn(cg_mode: CUDAGraphMode) -> None:
            calls.append((label, cg_mode, in_graph))

        return forward_fn

    @contextmanager
    def fake_cuda_graph(graph, pool):
        nonlocal in_graph
        assert not in_graph
        in_graph = True
        try:
            yield
        finally:
            in_graph = False

    monkeypatch.setattr(cudagraph_utils, "graph_capture", lambda device: nullcontext())
    monkeypatch.setattr(cudagraph_utils, "is_global_first_rank", lambda: False)
    fake_graph_pool_setter = MagicMock()
    fake_offloader = MagicMock()
    monkeypatch.setattr(torch.cuda, "CUDAGraph", MagicMock)
    monkeypatch.setattr(torch.cuda, "graph", fake_cuda_graph)
    monkeypatch.setattr(cudagraph_utils, "get_offloader", lambda: fake_offloader)
    monkeypatch.setattr(cudagraph_utils, "set_graph_pool_id", fake_graph_pool_setter)

    manager.capture(create_forward_fn)

    assert calls == [
        ("warmup", CUDAGraphMode.NONE, False),
        ("capture-1", CUDAGraphMode.NONE, False),
        ("capture-2", CUDAGraphMode.NONE, True),
    ]
    assert fake_offloader.sync_prev_onload.call_count == 2
    assert fake_offloader.join_after_forward.call_count == 2
    fake_graph_pool_setter.assert_called_once_with(manager.pool)
