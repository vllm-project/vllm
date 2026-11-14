# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for MooncakePrepareAndFinalize.

Hermetic: a fake buffer with Mooncake ElasticBuffer's signatures, single
process, CPU tensors; needs neither Mooncake nor a GPU.

  T1 expanded dispatch -> Mooncake also returns a recv_topk_idx tensor that
                          its kernels leave unfilled in this mode; the expert
                          ids must be rebuilt from the per-expert counts.
  T2 combine           -> Mooncake's combine() has no allocate_on_comm_stream
                          argument; sync and async finalize must not pass it.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.mooncake import (
    MooncakePrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
)

HIDDEN = 8


class _FakeEvent:
    event = None


class _FakeHandle:
    def __init__(self, counts: list[int]):
        self.num_recv_tokens_per_expert_list = counts
        self.psum_num_recv_tokens_per_scaleup_rank = torch.tensor(
            [sum(counts)], dtype=torch.int32
        )


class _FakeMooncakeBuffer:
    """Keyword signatures of mooncake.mooncake_elastic_buffer.ElasticBuffer."""

    def __init__(self, counts: list[int], out: torch.Tensor | None = None):
        self.counts = counts
        self.out = out
        self.recv_x = torch.randn(sum(counts), HIDDEN, dtype=torch.bfloat16)
        self.dispatch_kwargs: dict = {}
        self.combine_calls: list[bool] = []

    def dispatch(
        self,
        x,
        topk_idx=None,
        topk_weights=None,
        num_experts=None,
        num_max_tokens_per_rank=None,
        expert_alignment=None,
        handle=None,
        do_expand=False,
        do_cpu_sync=None,
        num_sms=None,
        async_with_compute_stream=False,
    ):
        self.dispatch_kwargs = dict(do_expand=do_expand, do_cpu_sync=do_cpu_sync)
        num_rows = self.recv_x.shape[0]
        # Unfilled when expanded: any value here must be ignored.
        unfilled_idx = torch.full(
            (num_rows, topk_idx.shape[1]), 12345, dtype=torch.int64
        )
        recv_topk_weights = torch.ones(num_rows)
        return (
            self.recv_x,
            unfilled_idx,
            recv_topk_weights,
            _FakeHandle(self.counts),
            _FakeEvent(),
        )

    def combine(
        self,
        x,
        handle,
        topk_weights=None,
        num_sms=None,
        async_with_compute_stream=False,
    ):
        self.combine_calls.append(async_with_compute_stream)
        return self.out, None, _FakeEvent()


def _make_pf(buffer: _FakeMooncakeBuffer, use_cudagraph: bool = False):
    # Rank 1 of 2, three local experts: global ids 3..5.
    return MooncakePrepareAndFinalize(
        buffer=buffer,
        num_dispatchers=2,
        dp_size=2,
        rank_expert_offset=3,
        num_experts=6,
        num_topk=2,
        use_cudagraph=use_cudagraph,
    )


def test_expanded_dispatch_rebuilds_ids_from_counts():
    counts = [2, 0, 3]
    buffer = _FakeMooncakeBuffer(counts)
    pf = _make_pf(buffer)
    a1 = torch.randn(4, HIDDEN, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 3], [4, 5], [3, 1], [5, 2]])
    topk_weights = torch.rand(4, 2)

    expert_x, _, meta, recv_ids, recv_weights = pf.prepare(
        a1,
        topk_weights,
        topk_ids,
        6,
        None,
        False,
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )

    assert buffer.dispatch_kwargs == {"do_expand": True, "do_cpu_sync": True}
    assert torch.equal(expert_x, buffer.recv_x)
    assert torch.equal(recv_ids, torch.tensor([[3], [3], [5], [5], [5]]))
    assert recv_weights.shape == (5, 1)
    assert meta is not None
    assert meta.expert_num_tokens.tolist() == counts


@pytest.mark.parametrize("do_async", [False, True])
def test_combine_omits_allocate_on_comm_stream(do_async):
    out_src = torch.full((4, HIDDEN), 7.0, dtype=torch.bfloat16)
    buffer = _FakeMooncakeBuffer([0, 0, 0], out=out_src)
    pf = _make_pf(buffer)
    pf.handles[0] = object()
    dst = torch.zeros_like(out_src)
    finalize = pf.finalize_async if do_async else pf.finalize

    ret = finalize(
        dst,
        torch.empty(0, dtype=torch.bfloat16),
        torch.empty(0, 2),
        torch.empty(0, 2, dtype=torch.int64),
        False,
        TopKWeightAndReduceContiguous(),
    )
    if do_async:
        ret()
    else:
        assert ret is None

    assert buffer.combine_calls == [do_async]
    assert torch.equal(dst, out_src)
