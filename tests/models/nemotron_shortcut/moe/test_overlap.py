# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The staged MoE overlaps the DP all-gather and reduce-scatter with the mixer
and the shared experts, without changing the results."""

from unittest.mock import patch

import pytest
import torch

from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.models.nemotron_shortcut.nvidia.moe import (
    NemotronShortcutMoE,
    NemotronShortcutMoERunner,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

from ..utils import build as _build
from ..utils import make_config as _config

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA")


class _FakeEPGroup:
    """Two-rank stand-in: gather duplicates rows, reduce-scatter folds them."""

    def __init__(self) -> None:
        self.streams: list[int] = []

    def dispatch(self, x, topk_weights, topk_ids, is_sequence_parallel, extra_tensors):
        assert extra_tensors is None
        self.streams.append(torch.cuda.current_stream().cuda_stream)
        # Slow enough that an unjoined consumer would observe stale data.
        torch.cuda._sleep(2_000_000)
        return (
            torch.cat((x, x * 2)),
            torch.cat((topk_weights, topk_weights * 2)),
            torch.cat((topk_ids, topk_ids)),
        )

    def combine_into_output(self, hidden_states, output, is_sequence_parallel):
        self.streams.append(torch.cuda.current_stream().cuda_stream)
        torch.cuda._sleep(2_000_000)
        rows = output.shape[0]
        torch.add(hidden_states[:rows], hidden_states[rows:], out=output)
        return output

    def allocate_combine_input(self, shape, dtype, device, is_sequence_parallel):
        return None


@pytest.fixture
def group(dist_init, workspace_init, monkeypatch):
    group = _FakeEPGroup()
    monkeypatch.setattr(naive_dp_ep, "get_ep_group", lambda: group)
    with torch.inference_mode(), set_forward_context(None, get_current_vllm_config()):
        yield group


def _moe(dtype: torch.dtype) -> NemotronShortcutMoE:
    """A layer on the all-gather/reduce-scatter transport, sized so that BF16
    may select FlashInfer's experts (the `FusedMoEKernel.apply` path)."""
    config = _config(
        hidden_size=256,
        moe_latent_size=128,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=128,
    )
    with set_default_torch_dtype(dtype):
        moe = _build(lambda: NemotronShortcutMoE(config))
    kernel = moe.experts.routed_experts.quant_method.moe_kernel
    kernel.impl.prepare_finalize = MoEPrepareAndFinalizeNaiveDPEPModular()
    return moe


def _forward(moe, routed, shared):
    """A shortcut block's stages, with a stand-in for the mixer."""
    dispatch = moe.begin_staged(routed)
    projection = shared * 0.5
    dispatch = moe.receive_staged(dispatch, (projection,))
    combine = moe.run_staged_experts(dispatch, projection)
    return moe.finish_staged(combine, shared)


def _synchronous(moe, routed, shared, forward=_forward):
    with patch.object(NemotronShortcutMoERunner, "overlaps_transport", False):
        return forward(moe, routed, shared).clone()


def _inputs(moe):
    torch.manual_seed(0)
    # More than 32 tokens: torch.compile cannot trace GateLinear's CuTe-DSL tier
    # below that, which vLLM never traces.
    routed = torch.randn(48, 256, device="cuda", dtype=moe.gate.weight.dtype)
    return routed, torch.randn_like(routed)


def _warm_up(moe, routed, shared):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _forward(moe, routed, shared)
    torch.cuda.current_stream().wait_stream(stream)


def test_overlaps_only_the_all_gather_transport_without_dbo(group):
    moe = _moe(torch.bfloat16)
    runner = moe.experts
    assert runner.overlaps_transport
    runner.enable_dbo = True
    assert not runner.overlaps_transport
    runner.enable_dbo = False
    kernel = runner.routed_experts.quant_method.moe_kernel
    kernel.impl.prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular()
    assert not runner.overlaps_transport


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_overlap_matches_the_synchronous_transport(group, dtype, compiled):
    moe = _moe(dtype)
    routed, shared = _inputs(moe)
    forward = torch.compile(_forward, fullgraph=True) if compiled else _forward
    # The same graph: inductor rounds the code around the stages its own way.
    expected = _synchronous(moe, routed, shared, forward)
    compute_stream = torch.cuda.current_stream().cuda_stream
    group.streams.clear()

    actual = forward(moe, routed, shared)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(group.streams) == 2
    assert compute_stream not in group.streams


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_graph_replays_the_overlap(group, dtype):
    """One capture of all stages: the experts and the finish join the side
    stream inside it."""
    moe = _moe(dtype)
    routed, shared = _inputs(moe)
    _warm_up(moe, routed, shared)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _forward(moe, routed, shared)
    routed.neg_()
    shared.mul_(-0.5)

    graph.replay()

    torch.testing.assert_close(
        output, _synchronous(moe, routed, shared), atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_graph_pieces_join_the_dispatch_before_they_end(group, dtype):
    """Piecewise capture splits the stages at the mixer's attention op, here
    between the two graphs: the first must join the dispatch it forked, or
    its capture fails."""
    moe = _moe(dtype)
    routed, shared = _inputs(moe)
    _warm_up(moe, routed, shared)
    pool = torch.cuda.graph_pool_handle()
    before, after = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with set_forward_context(
        None, get_current_vllm_config(), cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE
    ):
        with torch.cuda.graph(before, pool=pool):
            dispatch = moe.begin_staged(routed)
            projection = shared * 0.5
            dispatch = moe.receive_staged(dispatch, (projection,))
        with torch.cuda.graph(after, pool=pool):
            combine = moe.run_staged_experts(dispatch, projection)
            output = moe.finish_staged(combine, shared)
    routed.neg_()
    shared.mul_(-0.5)

    before.replay()
    after.replay()

    torch.testing.assert_close(
        output, _synchronous(moe, routed, shared), atol=0, rtol=0
    )
