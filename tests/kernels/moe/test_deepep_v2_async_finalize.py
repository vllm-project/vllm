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


@requires_deep_ep_v2
@pytest.mark.parametrize(
    "prefix,rows,expected_ids,expected_counts",
    [
        ([0, 0], 8, [-1] * 8, [0, 0]),
        ([3, 4, 7], 12, [20] * 3 + [-1] + [22] * 3 + [-1] * 5, [3, 0, 3]),
    ],
)
def test_expanded_metadata_masks_alignment_gaps_and_unused_capacity(
    prefix, rows, expected_ids, expected_counts
):
    ids, counts = _dv2._expanded_recv_metadata(
        torch.tensor(prefix, dtype=torch.int32), rows, 4, 20
    )
    assert ids[:, 0].tolist() == expected_ids
    assert counts.tolist() == expected_counts


@requires_deep_ep_v2
def test_expanded_receiver_masks_tensor_wide_scale_by_valid_rows(monkeypatch):
    pf = _make_pf(torch.empty(0, dtype=torch.bfloat16))
    pf.do_expand = True
    x = torch.arange(32, dtype=torch.float32).view(8, 4)
    handle = SimpleNamespace(
        psum_num_recv_tokens_per_expert=torch.tensor([3, 7], dtype=torch.int32),
        expert_alignment=4,
    )
    row_valid_calls: list[torch.Tensor | None] = []
    num_valid_calls: list[torch.Tensor | None] = []

    def masked_amax_scale(expert_x, quant_dtype, *, row_valid=None, num_valid=None):
        row_valid_calls.append(row_valid)
        num_valid_calls.append(num_valid)
        return torch.ones(1)

    def quantize_input(expert_x, scale, **kwargs):
        return expert_x, None

    monkeypatch.setattr(_dv2, "masked_amax_scale", masked_amax_scale)
    monkeypatch.setattr(_dv2, "moe_kernel_quantize_input", quantize_input)

    pf._receiver(
        _FakeEvent(False),
        x,
        None,
        None,
        handle,
        None,
        FusedMoEQuantConfig.make(torch.float8_e4m3fn),
        False,
    )

    assert num_valid_calls == [None]
    row_valid = row_valid_calls[0]
    assert row_valid is not None
    assert row_valid.tolist() == [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
    ]


@requires_deep_ep_v2
def test_expanded_receiver_clears_all_padding():
    pf = _make_pf(torch.empty(0, dtype=torch.bfloat16))
    pf.do_expand = True
    x = torch.full((4, 4), float("nan"))
    scales = torch.full((4, 1), float("nan"))
    weights = torch.full((4,), float("nan"))
    handle = SimpleNamespace(
        psum_num_recv_tokens_per_expert=torch.tensor([0, 0], dtype=torch.int32),
        expert_alignment=4,
    )

    prepared_x, prepared_scales, meta, ids, prepared_weights = pf._receiver(
        _FakeEvent(False), (x, scales), None, weights, handle, None, None, True
    )

    assert ids[:, 0].tolist() == [-1] * 4
    assert meta.expert_num_tokens.tolist() == [0, 0]
    for tensor in (prepared_x.float(), prepared_scales, prepared_weights):
        assert torch.isfinite(tensor).all()
        assert torch.count_nonzero(tensor) == 0


@requires_deep_ep_v2
def test_cpu_sync_rejects_capture_before_dispatch(monkeypatch):
    pf = _make_pf(torch.empty(0, dtype=torch.bfloat16))
    pf.do_cpu_sync = True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(ValueError, match="cannot be captured"):
        pf._do_dispatch(None, None, None, None, 8, None, None, False)
    assert not pf.buffer.calls


@requires_deep_ep_v2
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_expanded_receiver_uses_gpu_counts_and_clears_padding(dtype):
    pf = _make_pf(torch.empty(0, dtype=torch.bfloat16))
    pf.do_expand = True
    x = torch.full((8, 4), float("nan")).to(dtype)
    x[:2] = torch.ones(2, 4).to(dtype)
    scales = torch.full((8, 1), float("nan"))
    scales[:2] = 1
    weights = torch.full((8,), float("nan"))
    weights[:2] = 0.5
    handle = SimpleNamespace(
        psum_num_recv_tokens_per_expert=torch.tensor([2, 4], dtype=torch.int32),
        expert_alignment=4,
    )
    prepared_x, prepared_scales, meta, ids, prepared_weights = pf._receiver(
        _FakeEvent(False), (x, scales), None, weights, handle, None, None, True
    )
    assert ids[:, 0].tolist() == [0, 0] + [-1] * 6
    assert meta.expert_num_tokens.tolist() == [2, 0]
    assert meta.expert_num_tokens_cpu is None
    assert meta.psum_recv_per_rank is None
    assert meta.expert_input_alignment == 4
    for tensor in (prepared_x.float(), prepared_scales, prepared_weights):
        assert torch.isfinite(tensor).all()
        assert torch.count_nonzero(tensor[2:]) == 0


@requires_deep_ep_v2
def test_quantized_expanded_receiver_keeps_dispatch_activation_storage():
    pf = _make_pf(torch.empty(0, dtype=torch.bfloat16))
    pf.do_expand = True
    x = torch.ones(8, 128).to(torch.float8_e4m3fn)
    scales = torch.full((8, 1), float("nan"))
    scales[:2] = 1
    handle = SimpleNamespace(
        psum_num_recv_tokens_per_expert=torch.tensor([2, 4], dtype=torch.int32),
        expert_alignment=4,
    )
    prepared_x, prepared_scales, _, ids, _ = pf._receiver(
        _FakeEvent(False),
        (x, scales),
        None,
        None,
        handle,
        None,
        FusedMoEQuantConfig.make(torch.float8_e4m3fn, block_shape=[128, 128]),
        False,
    )
    assert prepared_x.data_ptr() == x.data_ptr()
    assert ids[:, 0].tolist() == [0, 0] + [-1] * 6
    assert torch.isfinite(prepared_scales).all()
    assert torch.count_nonzero(prepared_scales[2:]) == 0


def _run(pf, output: torch.Tensor, do_async: bool):
    empty = torch.empty(0, dtype=torch.bfloat16)
    weights = torch.empty(0, 2)
    ids = torch.empty(0, 2, dtype=torch.int64)
    fn = pf.finalize_async if do_async else pf.finalize
    return fn(output, empty, weights, ids, False, TopKWeightAndReduceContiguous())


@requires_deep_ep_v2
@pytest.mark.parametrize("defer_globalize", [False, True])
def test_prepare_keeps_nonexpanded_layout_across_capture_transitions(
    monkeypatch, defer_globalize
):
    """Capture transitions preserve routing shape and GPU-side receive counts."""
    context = None
    capturing = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(
        _dv2, "is_forward_context_available", lambda: context is not None
    )
    monkeypatch.setattr(_dv2, "get_forward_context", lambda: context)
    globalize_calls = []

    def globalize(ids, *args):
        globalize_calls.append(args)
        return ids + 4

    monkeypatch.setattr(_dv2, "_globalize_recv_topk_idx", globalize)
    tokens = torch.ones(3, 8, dtype=torch.bfloat16)
    pf = _make_pf(tokens)
    pf.defer_globalize = defer_globalize
    pf.rank_expert_offset = 4
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
        globalize_calls.clear()
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
        assert len(globalize_calls) == (0 if defer_globalize else 1)
        assert torch.all(ids == (0 if defer_globalize else 4))
        assert meta.rank_expert_offset == (4 if defer_globalize else None)


@requires_deep_ep_v2
def test_prepare_masks_tensor_wide_scale_after_nonexpanded_dispatch(monkeypatch):
    """A tensor-wide scale must ignore uninitialized rows past the receive tail."""
    tokens = torch.ones(3, 8, dtype=torch.bfloat16)
    pf = _make_pf(tokens)
    pf.defer_globalize = True
    masked_scale = torch.ones(1)
    quantize_scales: list[torch.Tensor] = []

    def masked_amax_scale(expert_x, quant_dtype, *, num_valid):
        assert num_valid.item() == 3
        return masked_scale

    def quantize_input(expert_x, scale, **kwargs):
        quantize_scales.append(scale)
        return expert_x, None

    monkeypatch.setattr(_dv2, "masked_amax_scale", masked_amax_scale)
    monkeypatch.setattr(_dv2, "moe_kernel_quantize_input", quantize_input)

    recv = pf.prepare_async(
        tokens,
        torch.ones(3, 2),
        torch.zeros(3, 2, dtype=torch.int64),
        8,
        None,
        False,
        FusedMoEQuantConfig.make(torch.float8_e4m3fn),
        defer_input_quant=False,
    )
    recv()

    assert len(quantize_scales) == 1
    assert quantize_scales[0] is masked_scale


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
