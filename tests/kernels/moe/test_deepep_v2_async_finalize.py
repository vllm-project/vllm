# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for DeepEPV2PrepareAndFinalize and its manager.

Hermetic: fake buffer/event, single process, no GPU collectives — only
requires deep_ep to be importable.

  T1 finalize_async  -> combine is issued with async_with_compute_stream=True
                        and the join is deferred to the receiver, which waits
                        on the combine event (device-side) before the copy.
  T2 DBO active      -> combine falls back to async_with_compute_stream=False;
                        the receiver still produces the correct output.
  T3 finalize (sync) -> combine is issued synchronously and the output is
                        copied immediately.
  T4 VLLM_DEEPEP_V2_NUM_SMS -> reaches the dispatch and the manager's
                        max_sms_used().
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
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


class _FakeDispatchBuffer:
    def dispatch(self, x, **kwargs):
        self.kwargs = kwargs
        handle = SimpleNamespace(
            num_recv_tokens_per_expert_list=[],
            psum_num_recv_tokens_per_scaleup_rank=torch.tensor(
                [3, 3], dtype=torch.int32
            ),
        )
        recv_x = x.new_zeros(8, x.size(1))
        return recv_x, None, torch.ones(8), handle, _FakeEvent(has_event=False)


def _dispatch_via_factory(monkeypatch, env: dict[str, str]):
    from tests.kernels.moe.utils import make_ep_prepare_finalize_with_fake_buffer

    monkeypatch.delenv("VLLM_DEEPEP_V2_NUM_SMS", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    buffer = _FakeDispatchBuffer()
    pf = make_ep_prepare_finalize_with_fake_buffer("deepep_v2", buffer, monkeypatch)
    a1 = torch.randn(3, 256, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 4], [1, 2], [5, 6]])
    with set_forward_context(None, VllmConfig()):
        receiver = pf.prepare_async(
            a1, torch.rand(3, 2), topk_ids, 8, None, False, FusedMoEQuantConfig.make()
        )
    return buffer.kwargs, receiver


@requires_deep_ep_v2
@pytest.mark.parametrize("env, expected", [(None, 0), ("24", 24)])
def test_deepep_v2_num_sms_knob(env, expected, monkeypatch):
    kwargs, _ = _dispatch_via_factory(
        monkeypatch, {} if env is None else {"VLLM_DEEPEP_V2_NUM_SMS": env}
    )
    assert kwargs["num_sms"] == expected  # 0: DeepEP's own formula


@requires_deep_ep_v2
@pytest.mark.parametrize("env, expected", [(None, 40), ("24", 24)])
def test_deepep_v2_manager_num_sms(env, expected, monkeypatch):
    import deep_ep

    from vllm.distributed.device_communicators.all2all import (
        DeepEPV2All2AllManager,
    )
    from vllm.distributed.device_communicators.base_device_communicator import (
        Cache,
    )

    class _Buffer:
        def __init__(self, **kwargs):
            pass

        def get_theoretical_num_sms(self, num_experts, num_topk):
            return 40

    monkeypatch.setattr(deep_ep, "ElasticBuffer", _Buffer)
    monkeypatch.delenv("VLLM_DEEPEP_V2_NUM_SMS", raising=False)
    if env is not None:
        monkeypatch.setenv("VLLM_DEEPEP_V2_NUM_SMS", env)
    mgr = object.__new__(DeepEPV2All2AllManager)
    mgr._device_group = None
    mgr.cpu_group = None
    mgr.handle_cache = Cache()
    mgr._num_sms = None
    mgr._gin_checked = True
    # Keep the handle alive: the cache holds weak references.
    handle = mgr.get_handle(
        dict(
            num_max_tokens_per_rank=16,
            hidden=256,
            num_topk=2,
            num_experts=8,
            use_fp8_dispatch=False,
        )
    )
    assert isinstance(handle, _Buffer)
    assert mgr.max_sms_used() == expected
