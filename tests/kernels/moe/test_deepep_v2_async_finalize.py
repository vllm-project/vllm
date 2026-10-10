# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for DeepEPV2PrepareAndFinalize.finalize_async.

Hermetic: fake buffer/event, single process, no GPU collectives — only
requires deep_ep to be importable.

  T1 finalize_async  -> combine is issued with async_with_compute_stream=True
                        and the join is deferred to the receiver, which waits
                        on the combine event (device-side) before the copy.
  T2 DBO active      -> combine falls back to async_with_compute_stream=False;
                        the receiver still produces the correct output.
  T3 finalize (sync) -> combine is issued synchronously and the output is
                        copied immediately.
"""

import pytest
import torch

from vllm.utils.import_utils import has_deep_ep_v2

requires_deep_ep_v2 = pytest.mark.skipif(
    not has_deep_ep_v2(),
    reason="Requires DeepEP v2 (ElasticBuffer)",
)

if has_deep_ep_v2():
    import vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 as _dv2
    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import (
        DeepEPV2PrepareAndFinalize,
    )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceContiguous,
    )


class _FakeEvent:
    def __init__(self, has_event: bool):
        self.event = object() if has_event else None
        self.waited = False

    def current_stream_wait(self):
        assert self.event is not None
        self.waited = True


class _FakeBuffer:
    def __init__(self, out: torch.Tensor):
        self.out = out
        self.calls: list[dict] = []
        self.last_event: _FakeEvent | None = None

    def combine(
        self,
        x,
        handle,
        topk_weights,
        async_with_compute_stream,
        allocate_on_comm_stream,
    ):
        self.calls.append({"async_with_compute_stream": async_with_compute_stream})
        # DeepEP returns a completion event only for an async combine.
        self.last_event = _FakeEvent(has_event=async_with_compute_stream)
        return self.out, None, self.last_event


def _make_pf(out: torch.Tensor):
    pf = DeepEPV2PrepareAndFinalize(
        buffer=_FakeBuffer(out),
        num_dispatchers=1,
        dp_size=1,
        rank_expert_offset=0,
        num_experts=8,
        num_topk=2,
    )
    pf.handles[0] = object()
    return pf


def _run(pf, output: torch.Tensor, do_async: bool):
    empty = torch.empty(0, dtype=torch.bfloat16)
    weights = torch.empty(0, 2)
    ids = torch.empty(0, 2, dtype=torch.int64)
    fn = pf.finalize_async if do_async else pf.finalize
    return fn(output, empty, weights, ids, False, TopKWeightAndReduceContiguous())


@requires_deep_ep_v2
def test_finalize_async_defers_join_to_receiver():
    out_src = torch.full((4, 8), 7.0, dtype=torch.bfloat16)
    pf = _make_pf(out_src)
    dst = torch.zeros_like(out_src)
    recv = _run(pf, dst, do_async=True)
    assert pf.buffer.calls[0]["async_with_compute_stream"] is True
    assert not torch.equal(dst, out_src), "join must not happen before receiver"
    recv()
    assert pf.buffer.last_event.waited, "receiver must join via event wait"
    assert torch.equal(dst, out_src)


@requires_deep_ep_v2
def test_finalize_async_under_dbo_uses_sync_combine(monkeypatch):
    monkeypatch.setattr(_dv2, "dbo_enabled", lambda: True)
    out_src = torch.full((4, 8), 7.0, dtype=torch.bfloat16)
    pf = _make_pf(out_src)
    dst = torch.zeros_like(out_src)
    recv = _run(pf, dst, do_async=True)
    assert pf.buffer.calls[0]["async_with_compute_stream"] is False
    recv()
    assert not pf.buffer.last_event.waited, "no event to wait on for sync combine"
    assert torch.equal(dst, out_src)


@requires_deep_ep_v2
def test_finalize_sync():
    out_src = torch.full((4, 8), 7.0, dtype=torch.bfloat16)
    pf = _make_pf(out_src)
    dst = torch.zeros_like(out_src)
    ret = _run(pf, dst, do_async=False)
    assert ret is None
    assert pf.buffer.calls[0]["async_with_compute_stream"] is False
    assert torch.equal(dst, out_src)
