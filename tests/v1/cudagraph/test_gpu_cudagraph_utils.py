# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionState,
    BatchExecutionDescriptor,
    CudaGraphManager,
)


def test_full_cudagraph_capture_prewarms_capture_forward_fn(monkeypatch):
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

    calls: list[tuple[str, CUDAGraphMode, bool]] = []
    in_graph = False

    def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
        label = "warmup" if warmup else "capture"

        def forward_fn(cg_mode: CUDAGraphMode) -> None:
            calls.append((label, cg_mode, in_graph))

        return forward_fn, AttentionState(attn_metadata=None, slot_mappings={})

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
    monkeypatch.setattr(torch.cuda, "CUDAGraph", MagicMock)
    monkeypatch.setattr(torch.cuda, "graph", fake_cuda_graph)
    monkeypatch.setattr(
        cudagraph_utils,
        "get_offloader",
        lambda: MagicMock(
            sync_prev_onload=MagicMock(),
            join_after_forward=MagicMock(),
        ),
    )

    manager.capture(create_forward_fn)

    assert calls == [
        ("warmup", CUDAGraphMode.NONE, False),
        ("capture", CUDAGraphMode.NONE, False),
        ("capture", CUDAGraphMode.NONE, True),
    ]
