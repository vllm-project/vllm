# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest
import torch
from torch._dynamo.graph_bytecode_inputs import index_to_external_object_weakref
from torch._dynamo.testing import EagerAndRecordGraphs

import vllm.compilation.side_stream as side_stream
from vllm.compilation.backends import wrap_with_cudagraph_if_needed
from vllm.config import CompilationConfig, CUDAGraphMode, VllmConfig
from vllm.platforms import current_platform


@pytest.fixture(autouse=True)
def resolver_installed() -> None:
    """register_side_stream() installs the resolver; these tests exercise it
    without registering a real stream."""
    side_stream._install_stream_index_resolver()


def test_stream_index_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Index 0 resolves to the current stream dynamically, registry hits
    resolve normally, and a registry miss falls back to the side stream -
    the only external object vLLM registers - so artifacts stay executable
    without dynamo's registry-repopulating bytecode."""
    from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index

    class ExternalObject:
        pass

    current = ExternalObject()
    side = ExternalObject()
    unrelated = ExternalObject()
    monkeypatch.setattr(side_stream, "current_stream", lambda: current)
    monkeypatch.setattr(side_stream, "_side_stream", side)
    monkeypatch.setitem(index_to_external_object_weakref, 9, weakref.ref(unrelated))
    index_to_external_object_weakref.pop(7, None)

    resolver = get_external_object_by_index
    assert getattr(resolver, "_vllm_stream_resolver", False)

    assert resolver(0) is current
    assert resolver(9) is unrelated
    assert resolver(7) is side


def test_stream_index_resolver_reraises_without_side_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index

    monkeypatch.setattr(side_stream, "_side_stream", None)
    index_to_external_object_weakref.pop(7, None)
    with pytest.raises(AssertionError, match="not registered"):
        get_external_object_by_index(7)


def test_stream_piece_skips_independent_cudagraph_capture() -> None:
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        )
    )
    runnable = object()

    wrapped = wrap_with_cudagraph_if_needed(
        runnable,
        vllm_config,
        vllm_config.compilation_config,
        is_first_graph=False,
        is_last_graph=False,
        uses_side_stream=True,
    )

    assert wrapped is runnable


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only test")
def test_side_stream_uses_native_compile_context() -> None:
    _, high_priority = torch.cuda.Stream.priority_range()
    stream = torch.cuda.Stream(priority=high_priority)
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
