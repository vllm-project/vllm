# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch._dynamo.testing import EagerAndRecordGraphs

from vllm.compilation.side_stream import get_side_stream
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only test")
def test_side_stream_uses_native_compile_context() -> None:
    stream = get_side_stream()
    assert stream is not None
    backend = EagerAndRecordGraphs()

    def run(x: torch.Tensor) -> torch.Tensor:
        stream.wait_stream(torch.accelerator.current_stream())
        with stream:
            return x + 1

    x = torch.zeros(4, device="cuda")
    actual = torch.compile(run, backend=backend, fullgraph=True)(x)
    assert torch.equal(actual, x + 1)
    assert len(backend.graphs) == 1

    annotated_nodes = [
        node
        for node in backend.graphs[0].graph.nodes
        if node.meta.get("custom", {}).get("stream") not in (None, 0)
    ]
    assert annotated_nodes
    assert all(
        "vllm.side_stream" not in str(node.target)
        for node in backend.graphs[0].graph.nodes
    )
