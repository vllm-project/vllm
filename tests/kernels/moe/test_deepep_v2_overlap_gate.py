# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the DeepEP v2 combine<->shared-expert overlap
(``VLLM_DEEPEP_V2_COMBINE_OVERLAP``): no-event capability-gate semantics.

Hermetic: fake buffer/event; host-sync and capture probes are monkeypatched.
Single process, no GPU collectives — only requires deep_ep to be importable.

  T1 eager no-event   -> warn once + permanent overlap self-disable BEFORE any
                         capture, one-time full-sync join, output correct, and
                         subsequent calls take the synchronous combine branch.
  T2 capture no-event -> RuntimeError; never a host sync inside a captured
                         region (the pre-capture gate must have fired first).
  T3 event present    -> receiver defers the join to
                         event.current_stream_wait() (device-side), no host
                         sync, overlap stays enabled.
"""

import pytest
import torch
from vllm.utils.import_utils import has_deep_ep_v2

requires_deep_ep_v2 = pytest.mark.skipif(
    not has_deep_ep_v2(),
    reason="Requires DeepEP v2 (ElasticBuffer)",
)

if has_deep_ep_v2():
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceContiguous,
    )

    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import (
        DeepEPV2PrepareAndFinalize,
    )


class _FakeEvent:
    def __init__(self, has_event: bool):
        self.event = object() if has_event else None
        self.waited = False

    def current_stream_wait(self):
        assert self.event is not None
        self.waited = True


class _FakeBuffer:
    def __init__(self, out: torch.Tensor, has_event: bool):
        self.out, self.has_event = out, has_event
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
        self.last_event = _FakeEvent(self.has_event)
        return self.out, None, self.last_event


@pytest.fixture
def probes(monkeypatch):
    """Arm the overlap flag and replace host-sync / capture probes."""
    monkeypatch.setenv("VLLM_DEEPEP_V2_COMBINE_OVERLAP", "1")
    syncs: list[int] = []
    capturing = {"v": False}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: syncs.append(1))
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: capturing["v"]
    )
    return syncs, capturing


def _make_pf(out: torch.Tensor, has_event: bool):
    pf = DeepEPV2PrepareAndFinalize(
        buffer=_FakeBuffer(out, has_event),
        num_dispatchers=1,
        dp_size=1,
        rank_expert_offset=0,
        num_experts=8,
        num_topk=2,
    )
    pf.handles[0] = object()
    return pf


def _run_finalize(pf, output: torch.Tensor):
    empty = torch.empty(0, dtype=torch.bfloat16)
    weights = torch.empty(0, 2)
    ids = torch.empty(0, 2, dtype=torch.int64)
    return pf.finalize_async(
        output, empty, weights, ids, False, TopKWeightAndReduceContiguous()
    )


@requires_deep_ep_v2
def test_eager_no_event_self_disables(probes):
    syncs, _ = probes
    out_src = torch.full((4, 8), 7.0, dtype=torch.bfloat16)
    pf = _make_pf(out_src, has_event=False)
    assert pf._combine_overlap is True

    dst = torch.zeros_like(out_src)
    recv = _run_finalize(pf, dst)
    recv()
    assert pf._combine_overlap is False, "must self-disable on no-event"
    assert len(syncs) == 1, "exactly one eager full-sync join"
    assert torch.equal(dst, out_src)
    assert pf.buffer.calls[0]["async_with_compute_stream"] is True

    dst2 = torch.zeros_like(out_src)
    recv2 = _run_finalize(pf, dst2)
    recv2()
    assert pf.buffer.calls[1]["async_with_compute_stream"] is False, (
        "subsequent calls must use the synchronous combine"
    )
    assert len(syncs) == 1, "no further host syncs after self-disable"
    assert torch.equal(dst2, out_src)


@requires_deep_ep_v2
def test_capture_no_event_raises(probes):
    syncs, capturing = probes
    out_src = torch.full((4, 8), 7.0, dtype=torch.bfloat16)
    pf = _make_pf(out_src, has_event=False)
    capturing["v"] = True
    with pytest.raises(RuntimeError, match="no completion event"):
        _run_finalize(pf, torch.zeros_like(out_src))
    assert len(syncs) == 0, "must NOT host-sync inside a captured region"


@requires_deep_ep_v2
def test_event_path_defers_join(probes):
    syncs, _ = probes
    out_src = torch.full((4, 8), 7.0, dtype=torch.bfloat16)
    pf = _make_pf(out_src, has_event=True)
    dst = torch.zeros_like(out_src)
    recv = _run_finalize(pf, dst)
    assert pf._combine_overlap is True
    assert not torch.equal(dst, out_src), "join must not happen before receiver"
    recv()
    assert pf.buffer.last_event.waited, "receiver must join via event wait"
    assert torch.equal(dst, out_src)
    assert len(syncs) == 0, "event path must not host-sync"
