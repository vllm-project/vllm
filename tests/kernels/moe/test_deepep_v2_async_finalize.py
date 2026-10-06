# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for DeepEPV2PrepareAndFinalize.

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

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
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

    def dispatch(self, **kwargs):
        self.calls.append(kwargs)
        num_tokens = self.out.size(0)
        handle = SimpleNamespace(
            psum_num_recv_tokens_per_scaleup_rank=torch.tensor([num_tokens]),
        )
        ids = torch.zeros(num_tokens, 2, dtype=torch.int64)
        weights = torch.ones(num_tokens, 2)
        return self.out, ids, weights, handle, _FakeEvent(has_event=False)

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
def test_prepare_keeps_nonexpanded_layout_across_capture_transitions(monkeypatch):
    """Capture transitions preserve routing shape and GPU-side receive counts."""
    context = None
    capturing = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(
        _dv2, "is_forward_context_available", lambda: context is not None
    )
    monkeypatch.setattr(_dv2, "get_forward_context", lambda: context)
    monkeypatch.setattr(_dv2, "_globalize_recv_topk_idx", lambda ids, *args: ids)
    tokens = torch.ones(3, 8, dtype=torch.bfloat16)
    pf = _make_pf(tokens)
    for mode, capturing in (
        (None, False),
        (CUDAGraphMode.NONE, False),
        (CUDAGraphMode.FULL, False),
        (CUDAGraphMode.PIECEWISE, False),
        (None, True),
        (CUDAGraphMode.NONE, True),
        (CUDAGraphMode.FULL, True),
        (CUDAGraphMode.PIECEWISE, True),
        (CUDAGraphMode.NONE, False),
    ):
        context = (
            None
            if mode is None
            else SimpleNamespace(cudagraph_runtime_mode=mode, dp_metadata=None)
        )
        recv = pf.prepare_async(
            tokens,
            torch.ones(3, 2),
            torch.zeros(3, 2, dtype=torch.int64),
            8,
            None,
            False,
            FusedMoEQuantConfig.make(None),
            defer_input_quant=True,
        )
        call = pf.buffer.calls[-1]
        assert call["do_expand"] is False
        assert call["do_cpu_sync"] is False
        assert call["num_max_tokens_per_rank"] == 4

        # The receiver may run after capture and the forward context change.
        capturing = not capturing
        context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
        _, _, meta, ids, _ = recv()
        assert meta is not None
        assert ids.shape == (3, 2)
        assert meta.expert_num_tokens is None
        assert meta.psum_recv_per_rank is not None


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
