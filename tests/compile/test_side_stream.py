# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest
import torch
from torch._dynamo.graph_bytecode_inputs import index_to_external_object_weakref
from torch._dynamo.testing import EagerAndRecordGraphs

import vllm.compilation.side_stream as side_stream
from vllm.compilation.backends import wrap_with_cudagraph_if_needed
from vllm.compilation.side_stream import get_side_stream
from vllm.config import CompilationConfig, CUDAGraphMode, VllmConfig
from vllm.platforms import current_platform


def test_stream_mapping_restores_exact_registry_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExternalObject:
        pass

    prior_current = ExternalObject()
    prior_side = ExternalObject()
    unrelated = ExternalObject()
    current = ExternalObject()
    side = ExternalObject()
    original = index_to_external_object_weakref.copy()
    index_to_external_object_weakref[0] = weakref.ref(prior_current)
    index_to_external_object_weakref[7] = weakref.ref(prior_side)
    index_to_external_object_weakref[9] = weakref.ref(unrelated)
    monkeypatch.setattr(side_stream, "current_stream", lambda: current)
    monkeypatch.setattr(side_stream, "get_side_stream", lambda: side)

    try:
        with side_stream.use_stream_mapping(((0, "current"), (7, "side"))):
            assert index_to_external_object_weakref[0]() is current
            assert index_to_external_object_weakref[7]() is side
            assert index_to_external_object_weakref[9]() is unrelated
        assert index_to_external_object_weakref[0]() is prior_current
        assert index_to_external_object_weakref[7]() is prior_side
        assert index_to_external_object_weakref[9]() is unrelated
    finally:
        index_to_external_object_weakref.clear()
        index_to_external_object_weakref.update(original)


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
        stream_mapping=((0, "current"), (7, "side")),
    )

    assert wrapped is runnable


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
