# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.runner import shared_experts as module
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)


def make_shared(monkeypatch):
    # Exercise the CUDA scheduling decision without allocating CUDA objects.
    monkeypatch.setattr(module, "aux_stream", lambda: object())
    monkeypatch.setattr(module.envs, "VLLM_DISABLE_SHARED_EXPERTS_STREAM", False)
    monkeypatch.setattr(module.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())
    parallel = SimpleNamespace(enable_eplb=False, use_fi_nvl_two_sided_kernels=False)
    return SharedExperts(
        torch.nn.Identity(),
        cast(FusedMoEConfig, SimpleNamespace(moe_parallel_config=parallel)),
        enable_dbo=False,
        mk_can_overlap_shared_experts=lambda: False,
    )


def test_disable_aux_stream_runs_shared_expert_inline(monkeypatch):
    shared = make_shared(monkeypatch)
    shared.disable_aux_stream()
    x = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    assert not shared.maybe_forward_async(x)
    shared(x, SharedExpertsOrder.NO_OVERLAP)
    torch.testing.assert_close(shared.output, x)


def test_disable_aux_stream_rejects_unconsumed_output(monkeypatch):
    shared = make_shared(monkeypatch)
    shared.disable_aux_stream()
    x = torch.ones(1, 4)
    shared(x, SharedExpertsOrder.NO_OVERLAP)
    with pytest.raises(RuntimeError, match="unconsumed"):
        shared.disable_aux_stream()
    torch.testing.assert_close(shared.output, x)
    shared.disable_aux_stream()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_inline_shared_expert_replays_inside_conditional_graph():
    parallel = SimpleNamespace(enable_eplb=False, use_fi_nvl_two_sided_kernels=False)
    shared = SharedExperts(
        torch.nn.Identity(),
        cast(FusedMoEConfig, SimpleNamespace(moe_parallel_config=parallel)),
        enable_dbo=False,
        mk_can_overlap_shared_experts=lambda: False,
    )
    shared.disable_aux_stream()
    x = torch.arange(4, dtype=torch.float32, device="cuda").reshape(1, 4)
    output = torch.zeros_like(x)
    condition = torch.ones((), dtype=torch.bool, device="cuda")
    graph = torch.cuda.CUDAGraph()
    if not hasattr(graph, "begin_capture_to_if_node"):
        pytest.skip("requires conditional CUDA graph support")
    with torch.cuda.graph(graph):
        graph.begin_capture_to_if_node(condition)
        assert not shared.maybe_forward_async(x)
        shared(x, SharedExpertsOrder.NO_OVERLAP)
        output.copy_(shared.output)
        graph.end_capture_to_conditional_node()
    graph.replay()
    torch.testing.assert_close(output, x)
    condition.zero_()
    output.fill_(-1)
    graph.replay()
    torch.testing.assert_close(output, torch.full_like(x, -1))
