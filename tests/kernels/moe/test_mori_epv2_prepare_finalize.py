# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MORI EPv2 prepare/finalize and all2all manager, with MORI faked out."""

import sys
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.model_executor.layers.fused_moe.prepare_finalize.mori_epv2 as mori_epv2
import vllm.utils.import_utils as import_utils
from vllm.config.parallel import resolve_mori_all2all
from vllm.model_executor.layers.fused_moe.prepare_finalize.mori_epv2 import (
    MoriEPv2PrepareAndFinalize,
)
from vllm.utils.network_utils import get_open_port


class FakeOp:
    """MORI EPv2 op whose receive buffers have `capacity` rows, `recv` valid."""

    def __init__(self, capacity=256, recv=0):
        self.routing = object()
        self.recv_a1 = torch.full((capacity, 4), float("nan"), dtype=torch.bfloat16)
        self.recv_weights = torch.full((capacity, 2), float("inf"))
        self.recv_ids = torch.full((capacity, 2), 5, dtype=torch.int32)
        self.recv = recv
        self.combine_in = torch.empty(capacity, 4, dtype=torch.bfloat16)
        self.sent_ids: torch.Tensor | None = None
        self.combine_input: torch.Tensor | None = None
        self.combine_routing = None

    def combine_in_view(self):
        return self.combine_in

    def dispatch(self, input, weights, scales, indices, *, return_routing=True):
        self.sent_ids = indices
        return (
            self.recv_a1,
            self.recv_weights,
            None,
            self.recv_ids,
            torch.tensor([self.recv], dtype=torch.int32),
            self.routing,
        )

    def combine(self, input, weights=None, indices=None, *, routing):
        self.combine_input = input
        self.combine_routing = routing
        return torch.zeros(1, 4, dtype=torch.bfloat16), None


def _make_pf(op, num_local_experts=2, rank_expert_offset=0, **kwargs):
    return MoriEPv2PrepareAndFinalize(
        op, 16, 4, num_local_experts, rank_expert_offset, **kwargs
    )


def _run(monkeypatch, op, num_tokens_across_dp, topk_ids, **pf_kwargs):
    """One prepare + finalize round trip under the given DP token counts."""
    if num_tokens_across_dp is None:
        ctx = SimpleNamespace(dp_metadata=None)
    else:
        counts = torch.tensor(num_tokens_across_dp)
        ctx = SimpleNamespace(
            dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=counts)
        )
    monkeypatch.setattr(mori_epv2, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(mori_epv2, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(
        mori_epv2, "get_dp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    prepare_finalize = _make_pf(op, **pf_kwargs)
    num_tokens = topk_ids.shape[0]
    prepared = prepare_finalize.prepare(
        torch.ones(num_tokens, 4, dtype=torch.bfloat16),
        torch.ones(num_tokens, 2, dtype=torch.bfloat16),
        topk_ids,
        num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=SimpleNamespace(is_per_tensor=False, a1_scale=None),
    )
    prepare_finalize.finalize(
        torch.empty(num_tokens, 4, dtype=torch.bfloat16),
        prepared[0],
        prepared[4],
        prepared[3],
        False,
        SimpleNamespace(),
    )
    return prepared


@pytest.mark.parametrize(
    "num_tokens_across_dp,recv,rows",
    [
        # Sum of DP batches (6) rounded up to 32 rows, of 256 capacity rows.
        ([1, 3, 2, 0], 7, 32),
        ([33, 0, 0, 0], 20, 64),
        # TP=2 per DP rank: each TP rank may send its DP batch, 2 * 50 rows.
        ([30, 20], 90, 128),
        # Dispatchers not a multiple of DP ranks: dispatchers * largest batch.
        ([10, 20, 5], 60, 96),
        # Bound above capacity: keep the full buffer.
        ([100, 100, 100, 100], 200, 256),
        # No DP metadata, or stale metadata missing this rank's batch: full buffer.
        (None, 7, 256),
        ([0, 3, 2, 0], 5, 256),
    ],
)
def test_receive_rows_are_bounded_by_dp_tokens(
    monkeypatch, num_tokens_across_dp, recv, rows
):
    op = FakeOp(recv=recv)
    a1, _, meta, ids, weights = _run(
        monkeypatch, op, num_tokens_across_dp, torch.tensor([[0, 1]])
    )
    for t, buf in ((a1, op.recv_a1), (ids, op.recv_ids), (weights, op.recv_weights)):
        assert t.shape[0] == rows
        assert t.data_ptr() == buf.data_ptr()  # views, no copy
    # Rows past the receive count are cleared; rows past the bound are untouched.
    assert torch.isnan(a1[:recv]).all() and (a1[recv:] == 0).all()
    assert (ids[:recv] == 5).all() and (ids[recv:] == -1).all()
    assert (weights[recv:] == 0).all()
    assert torch.isnan(op.recv_a1[rows:]).all()
    # The experts process every bounded row, without the received-row count.
    assert meta is None
    # combine sees only the bounded expert output, under dispatch's routing.
    assert op.combine_input is not None and op.combine_input.shape[0] == rows
    assert op.combine_routing is op.routing


def _fake_aiter(monkeypatch, resolve=None):
    """AITER as `aiter_moe_input_format` sees it, answering with `resolve`."""
    import enum

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe

    class GateMode(enum.StrEnum):
        SEPARATED = "separated"
        INTERLEAVE = "interleave"

    dtypes = SimpleNamespace(
        bf16=torch.bfloat16, fp8=torch.float8_e4m3fn, fp4x2=torch.float4_e2m1fn_x2
    )
    modules = {
        "aiter": SimpleNamespace(dtypes=dtypes),
        "aiter.ops": SimpleNamespace(),
        "aiter.ops.flydsl": SimpleNamespace(),
        "aiter.ops.flydsl.moe_common": SimpleNamespace(GateMode=GateMode),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        rocm_aiter_moe, "_aiter_resolve_activation_dtype", lambda: resolve
    )
    monkeypatch.setattr(
        rocm_aiter_moe, "_resolve_aiter_activation", lambda act: (0, None)
    )
    monkeypatch.setattr(rocm_aiter_ops, "fused_moe_supports_gate_mode", lambda: True)
    monkeypatch.setattr(
        rocm_aiter_moe.current_platform, "fp8_dtype", lambda: torch.float8_e4m3fn
    )
    monkeypatch.setattr(
        rocm_aiter_ops, "is_fused_moe_situv2_gate_up_interleaved", lambda: True
    )
    return rocm_aiter_moe


def _moe_quant(kind):
    """A quant config of `kind` with what the AITER format lookup reads."""
    fields: dict[str, Any] = dict(
        use_mxfp4_w4a4=False,
        use_mxfp4_w4a16=False,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        block_shape=None,
        per_out_ch_quant=False,
        w1_bias=None,
        w2_bias=None,
    )
    kinds: dict[str, dict[str, Any]] = {
        "mxfp4": dict(use_mxfp4_w4a4=True),
        "mxfp4_w4a16": dict(use_mxfp4_w4a16=True),
        "fp8_block": dict(use_fp8_w8a8=True, block_shape=[128, 128]),
        "fp8_channel": dict(use_fp8_w8a8=True, per_out_ch_quant=True),
        "bf16": {},
    }
    fields.update(kinds[kind])
    return SimpleNamespace(**fields)


def _moe_cfg(activation="SILU", epv2=True, aiter=True, dsv4_a4w4=False):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    return SimpleNamespace(
        activation=MoEActivation[activation],
        use_mori_epv1_kernels=False,
        use_mori_epv2_kernels=epv2,
        rocm_aiter_fmoe_enabled=aiter,
        moe_backend="auto",
        use_mxfp4_w4a4_dsv4=dsv4_a4w4,
        in_dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(
    "kind,activation,answer,fmt,bound",
    [
        # GLM / DeepSeek-R1 MXFP4 W4A4 with SiLU: AITER quantizes to MXFP4.
        ("mxfp4", "SILU", torch.float4_e2m1fn_x2, "mxfp4", None),
        # FP8 block-128: FP8 with one FP32 scale per 128 elements.
        ("fp8_block", "SILU", torch.float8_e4m3fn, "fp8", None),
        # MXFP4 W4A16 (interleaved gate/up) with AITER_BF16_FP8_MOE_BOUND=0.
        ("mxfp4_w4a16", "SILU", torch.float8_e4m3fn, "mxfp8", None),
        # The dtype depends on the batch size: BF16, with a hint how to pin it.
        ("mxfp4_w4a16", "SILU", None, "bf16", "AITER_BF16_FP8_MOE_BOUND"),
        ("mxfp4", "SWIGLUOAI", None, "bf16", "GPTOSS_SWIGLU_MXFP4_BF16_BOUND"),
        # Kimi-K3's SiTU MoE by default consumes BF16.
        ("mxfp4_w4a16", "SITU", torch.bfloat16, "bf16", None),
    ],
)
def test_aiter_answer_picks_the_dispatch_format(
    monkeypatch, kind, activation, answer, fmt, bound
):
    calls = []

    def resolve(quant_type, weight_dtype, *, activation, gate_mode):
        calls.append((quant_type, weight_dtype, gate_mode))
        return answer

    rocm_aiter_moe = _fake_aiter(monkeypatch, resolve)
    choice = rocm_aiter_moe.aiter_moe_input_format(
        _moe_cfg(activation), _moe_quant(kind)
    )
    assert choice.format.value == fmt
    # Only a choice that a configuration change can lift is worth a warning.
    assert choice.actionable == (bound is not None)
    if bound is not None:
        assert f"{bound}=0" in choice.reason
    # AITER is asked with the quant type and gate mode AiterExperts hand it.
    quant_type, weight_dtype, gate_mode = calls[0]
    AiterQuantType = rocm_aiter_moe.AiterQuantType
    if kind == "fp8_block":
        assert (quant_type, weight_dtype) == (
            AiterQuantType.BLOCK_128x128,
            torch.float8_e4m3fn,
        )
    else:
        assert (quant_type, weight_dtype) == (
            AiterQuantType.BLOCK_1X32,
            torch.float4_e2m1fn_x2,
        )
    interleaved = kind == "mxfp4_w4a16"
    assert gate_mode == ("interleave" if interleaved else "separated")


@pytest.mark.parametrize("kind", ["bf16", "fp8_channel"])
def test_formats_aiter_has_no_dispatch_for_stay_bf16(monkeypatch, kind):
    rocm_aiter_moe = _fake_aiter(monkeypatch, resolve=Mock())
    choice = rocm_aiter_moe.aiter_moe_input_format(_moe_cfg(), _moe_quant(kind))
    assert choice.format.value == "bf16" and not choice.actionable


def test_forced_aiter_activation_dtype_wins(monkeypatch):
    # DeepSeek V4.1 opt-in a4w4 hands AITER quant_dtype_a=fp4x2 regardless.
    rocm_aiter_moe = _fake_aiter(monkeypatch, resolve=Mock(return_value=None))
    moe = _moe_cfg(dsv4_a4w4=True)
    choice = rocm_aiter_moe.aiter_moe_input_format(moe, _moe_quant("mxfp4_w4a16"))
    assert choice.format.value == "mxfp4"


def test_aiter_without_the_resolver_dispatches_bf16(monkeypatch):
    rocm_aiter_moe = _fake_aiter(monkeypatch, resolve=None)
    choice = rocm_aiter_moe.aiter_moe_input_format(_moe_cfg(), _moe_quant("mxfp4"))
    assert choice.format.value == "bf16" and choice.actionable


def _fake_mori_choice(monkeypatch, fmt, mxfp8_ok=True):
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        AiterInputChoice,
        AiterInputFormat,
    )

    answer = AiterInputChoice(AiterInputFormat(fmt), f"AITER consumes {fmt}")
    monkeypatch.setattr(mori_epv2, "aiter_moe_input_format", lambda moe, qc: answer)
    monkeypatch.setattr(mori_epv2, "_aiter_supports_mxfp8_dispatch", lambda: mxfp8_ok)
    monkeypatch.delenv("MORI_ENABLE_SDMA", raising=False)


@pytest.mark.parametrize(
    "aiter_fmt,transport,mxfp8_ok,fmt",
    [
        ("mxfp4", "native", True, "mxfp4"),
        ("fp8", "native", False, "fp8"),
        ("bf16", "native", True, "bf16"),
        ("mxfp8", "native", True, "mxfp8"),
        # MXFP8 needs an AITER that both emits and consumes it.
        ("mxfp8", "native", False, "bf16"),
        # AsyncLL and the internode kernels carry no FP4.
        ("mxfp4", "async_ll", True, "bf16"),
        ("mxfp4", "internode", True, "bf16"),
        ("fp8", "async_ll", True, "fp8"),
        ("mxfp8", "internode", True, "mxfp8"),
    ],
)
def test_mori_carries_what_it_can(monkeypatch, aiter_fmt, transport, mxfp8_ok, fmt):
    _fake_mori_choice(monkeypatch, aiter_fmt, mxfp8_ok)
    choice = mori_epv2.mori_epv2_dispatch_format(
        _moe_cfg(),
        None,
        async_ll=transport == "async_ll",
        internode=transport == "internode",
    )
    assert choice.format.value == fmt
    # Only a missing AITER capability is worth a warning.
    assert choice.actionable == (aiter_fmt == "mxfp8" and fmt == "bf16")


def test_mori_enable_sdma_means_async_ll(monkeypatch):
    _fake_mori_choice(monkeypatch, "mxfp4")
    monkeypatch.setenv("MORI_ENABLE_SDMA", "1")
    assert mori_epv2.mori_epv2_dispatch_format(_moe_cfg(), None).format.value == "bf16"


@pytest.mark.parametrize("epv2,aiter", [(False, True), (True, False)])
def test_only_epv2_with_aiter_experts_dispatches_quantized(monkeypatch, epv2, aiter):
    _fake_mori_choice(monkeypatch, "mxfp4")
    moe = _moe_cfg(epv2=epv2, aiter=aiter)
    assert mori_epv2.mori_epv2_dispatch_format(moe, None).format.value == "bf16"


@pytest.mark.parametrize(
    "fmt,args",
    [
        ("bf16", {}),
        ("fp8", (torch.float8_e4m3fn, 512 // 128, 4)),
        ("mxfp8", (torch.float8_e4m3fn, 512 // 32, 1)),
        ("mxfp4", (torch.float4_e2m1fn_x2, 512 // 32, 1)),
    ],
)
def test_mori_op_is_sized_for_the_format(monkeypatch, fmt, args):
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        AiterInputFormat,
    )
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "fp8_dtype", lambda: torch.float8_e4m3fn)
    got = mori_epv2.mori_epv2_dispatch_args(AiterInputFormat(fmt), 512)
    if not args:
        # BF16 keeps the op config, and with it the cached op, as it was.
        assert got == {}
        return
    dtype, scale_dim, scale_type_size = args
    assert got == dict(
        dispatch_data_type=dtype,
        combine_data_type=torch.bfloat16,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
    )


class FakeQuantOp(FakeOp):
    """MORI EPv2 op dispatching a quantized format of hidden size 256.

    Scale rows come back as int32 words, `scale_row_bytes` wide: MORI's HIP
    backend pads them to 128 B, its FlyDSL backend does not.
    """

    def __init__(self, fmt, scale_row_bytes, capacity=256, recv=0):
        super().__init__(capacity, recv)
        cols = 128 if fmt == "mxfp4" else 256
        dtype = torch.float4_e2m1fn_x2 if fmt == "mxfp4" else torch.float8_e4m3fn
        self.recv_a1 = torch.full((capacity, cols), 0x38, dtype=torch.uint8).view(dtype)
        self.recv_scales = torch.full(
            (capacity, scale_row_bytes // 4), 0x7F7F7F7F, dtype=torch.int32
        )
        self.combine_in = torch.empty(capacity, 256, dtype=torch.bfloat16)
        self.sent: tuple | None = None

    def dispatch(self, input, weights, scales, indices, *, return_routing=True):
        self.sent = (input, scales)
        out = super().dispatch(input, weights, scales, indices)
        # Two int32 words of scales a row (8 e8m0 or 2 FP32), in padded rows.
        return out[:2] + (self.recv_scales[:, :2],) + out[3:]


@pytest.mark.parametrize(
    "fmt,scale_dtype,num_scales,row_bytes",
    [
        ("mxfp4", torch.float8_e8m0fnu, 8, 8),
        ("mxfp4", torch.float8_e8m0fnu, 8, 128),
        ("mxfp8", torch.float8_e8m0fnu, 8, 128),
        ("fp8", torch.float32, 2, 8),
        ("fp8", torch.float32, 2, 128),
    ],
)
def test_quantized_dispatch(monkeypatch, fmt, scale_dtype, num_scales, row_bytes):
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        AiterInputFormat,
    )

    quantized = (torch.empty(3, 128), torch.empty(3, num_scales))
    seen = []

    def quantize(x, f):
        seen.append((x, f))
        return quantized

    monkeypatch.setattr(mori_epv2, "_quantize_for_dispatch", quantize)
    op = FakeQuantOp(fmt, row_bytes, recv=7)
    prepare_finalize = _make_pf(op, dispatch_format=AiterInputFormat(fmt))
    monkeypatch.setattr(mori_epv2, "is_forward_context_available", lambda: True)
    counts = SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([3, 3, 2, 0]))
    ctx = SimpleNamespace(dp_metadata=counts)
    monkeypatch.setattr(mori_epv2, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(
        mori_epv2, "get_dp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    a1 = torch.ones(3, 256, dtype=torch.bfloat16)
    dispatch_a1, scale, meta, ids, _ = prepare_finalize.prepare(
        a1,
        torch.ones(3, 2),
        torch.zeros(3, 2, dtype=torch.int64),
        num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=SimpleNamespace(quant_dtype=None, is_per_tensor=False),
    )
    assert seen[0][0] is a1 and seen[0][1] == AiterInputFormat(fmt)
    assert op.sent is not None
    assert op.sent[0] is quantized[0] and op.sent[1] is quantized[1]
    # The rows go to the experts as received, bounded to 32 rows; the scales
    # are the leading columns of MORI's possibly padded rows, contiguous.
    assert dispatch_a1.dtype == op.recv_a1.dtype
    assert dispatch_a1.data_ptr() == op.recv_a1.data_ptr()
    assert dispatch_a1.shape[0] == 32
    assert scale.dtype == scale_dtype and scale.shape == (32, num_scales)
    assert scale.is_contiguous()
    # The experts read every bounded row: unused rows and scales are zeroed.
    assert meta is None
    assert (scale[:7].view(torch.uint8) == 0x7F).all()
    assert (scale[7:].view(torch.uint8) == 0).all()
    assert (dispatch_a1[:7].view(torch.uint8) == 0x38).all()
    assert (dispatch_a1[7:].view(torch.uint8) == 0).all()
    assert (ids[7:] == -1).all()


def _aiter_experts(monkeypatch, aiter_fmt, kind="mxfp4"):
    """Real AiterExperts around a fake AITER that answers `aiter_fmt`."""
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        AiterMxfp4Experts,
    )

    _fake_mori_choice(monkeypatch, aiter_fmt)
    experts = object.__new__(AiterMxfp4Experts)
    experts.moe_config = _moe_cfg()
    experts.quant_config = SimpleNamespace(
        **vars(_moe_quant(kind)), quant_dtype="mxfp4", is_per_tensor=True
    )
    return experts


@pytest.mark.parametrize("aiter_fmt", ["bf16", "fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("transport", ["native", "async_ll"])
def test_experts_and_dispatch_agree_on_the_format(monkeypatch, aiter_fmt, transport):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    experts = _aiter_experts(monkeypatch, aiter_fmt)
    if transport == "async_ll":
        monkeypatch.setenv("MORI_ENABLE_SDMA", "1")
    # all2all_utils sizes the op from the same call.
    fmt = mori_epv2.mori_epv2_dispatch_format(
        experts.moe_config, experts.quant_config
    ).format
    assert experts.mori_epv2_dispatch_format == fmt
    assert experts.expects_unquantized_inputs == (fmt.value == "bf16")
    # Only packed FP4 arrives as half the hidden size.
    _, _, out = experts.workspace_shapes(4, 8, 64, 2, 8, 2, None, MoEActivation.SILU)
    assert out == (4, 128 if fmt.value == "mxfp4" else 64)
    _make_pf(FakeOp(), dispatch_format=fmt).post_init_setup(experts)


@pytest.mark.parametrize("aiter_fmt", ["bf16", "fp8", "mxfp8", "mxfp4"])
def test_fp8_input_gets_every_row_as_the_count(monkeypatch, aiter_fmt):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe

    experts = _aiter_experts(monkeypatch, aiter_fmt)
    seen: dict[str, Any] = {}

    def fake_fused_experts(**kwargs):
        seen.update(kwargs)
        return kwargs["hidden_states"]

    monkeypatch.setattr(rocm_aiter_moe, "rocm_aiter_fused_experts", fake_fused_experts)
    monkeypatch.setattr(rocm_aiter_moe, "_write_moe_output", lambda out, res: None)
    monkeypatch.setattr(
        rocm_aiter_moe.rocm_aiter_ops, "get_moe_dispatch_policy", lambda: 0
    )
    hidden = torch.zeros(5, 64, dtype=torch.uint8)
    experts.apply(
        output=hidden,
        hidden_states=hidden,
        w1=hidden,
        w2=hidden,
        topk_weights=torch.ones(5, 2),
        topk_ids=torch.zeros(5, 2, dtype=torch.int32),
        activation=MoEActivation.SILU,
        global_num_experts=8,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=hidden,
        workspace2=hidden,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    # AITER's FP8 block MoE needs a row count to transpose pre-quantized
    # scales; without one from MORI it covers every row.
    count = seen["num_local_tokens"]
    if aiter_fmt == "fp8":
        assert count.dtype == torch.int32 and count.tolist() == [5]
    else:
        assert count is None


def test_quantized_dispatch_needs_aiter_experts():
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        AiterInputFormat,
    )

    experts = SimpleNamespace(
        moe_config=SimpleNamespace(in_dtype=torch.bfloat16),
        quant_config=SimpleNamespace(quant_dtype="mxfp4", is_per_tensor=True),
        expects_unquantized_inputs=False,
    )
    prepare_finalize = _make_pf(FakeOp(), dispatch_format=AiterInputFormat.MXFP4)
    with pytest.raises(ValueError, match="VLLM_MORI_EP_V2=0"):
        prepare_finalize.post_init_setup(experts)


def test_padding_tokens_are_routed_to_local_experts(monkeypatch):
    op = FakeOp(recv=2)
    # Rank 1 of 4 with 2 experts each owns experts 2 and 3. The second token is
    # DP padding, which the router marks with expert id -1.
    topk_ids = torch.tensor([[0, 7], [-1, -1]])
    _run(monkeypatch, op, None, topk_ids, num_local_experts=2, rank_expert_offset=2)
    assert op.sent_ids is not None and op.sent_ids.dtype == torch.int32
    assert op.sent_ids.tolist() == [[0, 7], [2, 3]]


@pytest.mark.parametrize("world_size,lsa_size", [(8, 8), (8, 4)])
def test_manager_requires_one_lsa_team(monkeypatch, world_size, lsa_size):
    from vllm.distributed.device_communicators import all2all
    from vllm.distributed.device_communicators.base_device_communicator import (
        All2AllManagerBase,
    )

    monkeypatch.setattr(
        All2AllManagerBase,
        "__init__",
        lambda self, group: self.__dict__.update(
            rank=0, world_size=world_size, cpu_group=group
        ),
    )
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", lambda *a, **k: None
    )
    monkeypatch.setattr(all2all, "CudaRTLibrary", Mock)
    device_comm = Mock(lsa_size=lsa_size)
    calls = Mock()  # records the order of op.close and comm.destroy
    comm = calls.comm
    comm.create_dev_comm.return_value = device_comm
    communicator = Mock()
    communicator.init.return_value = comm
    monkeypatch.setitem(
        sys.modules,
        "mori.cco",
        SimpleNamespace(
            Communicator=communicator,
            CCODevCommRequirements=SimpleNamespace,
            GDA_CONNECTION_NONE=0,
        ),
    )
    if lsa_size != world_size:
        with pytest.raises(ValueError, match="one LSA team"):
            all2all.MoriEPv2All2AllManager(object())
    else:
        cfg = dict(
            scale_dim=0,
            scale_type_size=4,
            num_experts_per_token=8,
            token_nbytes=4096,
            combine_token_nbytes=4096,
            effective_max_recv=1024,
            dispatch_dtype=None,
            combine_dtype=None,
            schedule=None,
        )
        monkeypatch.setitem(
            sys.modules,
            "mori.ops.dispatch_combine_v2",
            SimpleNamespace(
                EpDispatchCombineConfig=lambda **kw: SimpleNamespace(**(cfg | kw)),
                EpDispatchCombineOp=lambda config, comm: calls.op,
            ),
        )
        manager = all2all.MoriEPv2All2AllManager(object())
        assert manager.get_handle({}) is calls.op
        # The cache holds ops weakly; the manager keeps them alive itself.
        manager.handle_cache._cache.clear()
        manager.destroy()
        # Ops live in the communicator's window: they must go first.
        order = [c[0] for c in calls.mock_calls]
        assert order.index("op.close") < order.index("comm.destroy")
    device_comm.close.assert_called_once()
    comm.destroy.assert_called_once()


def _unique_id_worker(rank, init_method):
    from vllm.distributed.device_communicators import all2all
    from vllm.distributed.device_communicators.base_device_communicator import (
        All2AllManagerBase,
    )

    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    # With PP=2 and DP=2 the EP groups are [0, 2] and [1, 3].
    groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    group = groups[rank % 2]

    def fake_init(self, g):
        self.__dict__.update(rank=g.rank(), world_size=g.size(), cpu_group=g)

    All2AllManagerBase.__init__ = fake_init  # type: ignore[method-assign]
    all2all.CudaRTLibrary = Mock  # type: ignore[misc]
    communicator = Mock()
    communicator.get_unique_id.return_value = f"uid-of-rank-{rank}"
    communicator.init.return_value.create_dev_comm.return_value = Mock(lsa_size=2)
    sys.modules["mori.cco"] = SimpleNamespace(
        Communicator=communicator,
        CCODevCommRequirements=SimpleNamespace,
        GDA_CONNECTION_NONE=0,
    )
    all2all.MoriEPv2All2AllManager(group)
    # Every rank joins the communicator with the ID its group's first rank made.
    assert communicator.init.call_args.args[2] == f"uid-of-rank-{rank % 2}"
    dist.destroy_process_group()


def test_unique_id_comes_from_the_ep_groups_first_rank():
    mp.spawn(_unique_id_worker, args=(f"tcp://127.0.0.1:{get_open_port()}",), nprocs=4)


def test_experts_write_into_combine_staging_only_when_it_fits():
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceContiguous,
        TopKWeightAndReduceNoOP,
    )

    op = FakeOp()
    alloc = _make_pf(op).allocate_fused_expert_output
    device = op.combine_in.device
    out = alloc((32, 4), torch.bfloat16, device, TopKWeightAndReduceNoOP())
    assert out is not None and out.shape == (32, 4)
    assert out.data_ptr() == op.combine_in.data_ptr()
    # Unreduced expert output, other dtypes or too many rows keep the workspace.
    for args in (
        ((32, 4), torch.bfloat16, device, TopKWeightAndReduceContiguous()),
        ((32, 4), torch.float32, device, TopKWeightAndReduceNoOP()),
        ((257, 4), torch.bfloat16, device, TopKWeightAndReduceNoOP()),
    ):
        assert alloc(*args) is None


@pytest.mark.parametrize(
    "in_dtype,quant_dtype,quantizes_itself,per_tensor,ok",
    [
        (torch.bfloat16, None, False, False, True),
        # e.g. AITER, which quantizes after the all2all.
        (torch.bfloat16, torch.float8_e4m3fn, True, False, True),
        # AITER cannot quantize the received rows per tensor.
        (torch.bfloat16, torch.float8_e4m3fn, True, True, False),
        # OCP MX (MXFP4) reports a per-tensor group shape but scales each 1x32
        # block, which AITER quantizes after the all2all.
        (torch.bfloat16, "mxfp4", True, True, True),
        # e.g. FP8 Triton experts, which expect the dispatch to quantize.
        (torch.bfloat16, torch.float8_e4m3fn, False, False, False),
        (torch.float16, None, False, False, False),
    ],
)
def test_experts_it_cannot_feed_are_rejected(
    in_dtype, quant_dtype, quantizes_itself, per_tensor, ok
):
    experts = SimpleNamespace(
        moe_config=SimpleNamespace(in_dtype=in_dtype),
        quant_config=SimpleNamespace(
            quant_dtype=quant_dtype,
            is_per_tensor=per_tensor,
            use_fp8_w8a8=quant_dtype == torch.float8_e4m3fn,
            use_int8_w8a8=quant_dtype == torch.int8,
        ),
        expects_unquantized_inputs=quantizes_itself,
    )
    prepare_finalize = _make_pf(FakeOp(16))
    if ok:
        prepare_finalize.post_init_setup(experts)
    else:
        with pytest.raises(ValueError, match="VLLM_MORI_EP_V2=0"):
            prepare_finalize.post_init_setup(experts)


def _parallel_config(**overrides):
    fields = dict(
        all2all_backend="mori",
        mori_ep_v2=None,
        tensor_parallel_size=1,
        data_parallel_size=8,
        data_parallel_size_local=8,
        data_parallel_external_lb=False,
        data_parallel_backend="mp",
        nnodes_within_dp=1,
        prefill_context_parallel_size=1,
    )
    return SimpleNamespace(**(fields | overrides))


def _model_config(quant_cfg=None, **overrides):
    fields = {"dtype": torch.bfloat16, "quantization": None}
    arch = SimpleNamespace(quantization_config=quant_cfg)
    return SimpleNamespace(model_arch_config=arch, **(fields | overrides))


_EXTERNAL_LB = {"data_parallel_size_local": 1, "data_parallel_external_lb": True}
_FP8_BLOCK = {"quantization": "fp8", "quant_cfg": {"weight_block_size": [128, 128]}}
_MODELOPT_FP8 = {"quantization": "modelopt", "quant_cfg": {"quant_algo": "FP8"}}
_MODELOPT_NVFP4 = {"quantization": "modelopt", "quant_cfg": {"quant_algo": "NVFP4"}}


def _ct(strategy):
    act = {"type": "float", "num_bits": 8, "strategy": strategy}
    cfg = {"config_groups": {"group_0": {"input_activations": act}}}
    return {"quantization": "compressed-tensors", "quant_cfg": cfg}


def _quark(qscheme):
    inputs = {"dtype": "fp8_e4m3", "qscheme": qscheme}
    cfg = {"global_quant_config": {"input_tensors": inputs}}
    return {"quantization": "quark", "quant_cfg": cfg}


@pytest.mark.parametrize(
    "parallel,model,moe_backend,env,ep_v2",
    [
        ({}, {}, "aiter", {}, True),
        # EP across hosts: DP over two nodes, even with ep_size <= 8.
        ({"data_parallel_size": 16}, {}, "aiter", {}, False),
        ({"data_parallel_size_local": 4}, {}, "aiter", {}, False),
        # A DP replica (TP/PP) spanning nodes.
        ({"nnodes_within_dp": 2}, {}, "aiter", {}, False),
        # External LB only sees its own rank: fall back to the EP size.
        (_EXTERNAL_LB, {}, "aiter", {}, True),
        (_EXTERNAL_LB | {"data_parallel_size": 16}, {}, "aiter", {}, False),
        # The LSA team may span hosts (gfx1250); the manager checks it.
        ({"data_parallel_size": 16, "lsa_spans_hosts": True}, {}, "aiter", {}, True),
        ({}, {}, "aiter", {"MORI_ENABLE_SDMA": "1"}, False),
        # gfx1250 keeps EPv2 under SDMA.
        ({"sdma_picks_epv1": False}, {}, "aiter", {"MORI_ENABLE_SDMA": "1"}, True),
        ({}, {}, "aiter_mega_moe", {}, False),
        ({}, {"dtype": torch.float16}, "aiter", {}, False),
        # An explicit VLLM_MORI_EP_V2 wins over the deployment checks.
        ({}, {"dtype": torch.float16}, "aiter", {"VLLM_MORI_EP_V2": "1"}, True),
        ({}, {}, "aiter", {"VLLM_MORI_EP_V2": "0"}, False),
        # Already resolved: kept as is.
        ({"mori_ep_v2": True}, {"dtype": torch.float16}, "aiter", {}, True),
        # Other MoRI backends always run EPv1.
        ({"all2all_backend": "mori_high_throughput"}, {}, "aiter", {}, None),
        ({"validated_arch": False}, {}, "aiter", {}, False),
        ({"mori_reason": "too old"}, {}, "aiter", {}, False),
        # Per-tensor FP8 activations; block FP8 and NVFP4 are fine.
        ({}, {"quantization": "fp8"}, "aiter", {}, False),
        ({}, _FP8_BLOCK, "aiter", {}, True),
        ({}, _MODELOPT_FP8, "aiter", {}, False),
        ({}, _MODELOPT_NVFP4, "aiter", {}, True),
        ({}, _ct("tensor"), "aiter", {}, False),
        ({}, _ct("group"), "aiter", {}, True),
        ({}, _quark("per_tensor"), "aiter", {}, False),
        ({}, _quark("per_group"), "aiter", {}, True),
    ],
)
def test_mori_picks_epv2_where_supported(
    monkeypatch, parallel, model, moe_backend, env, ep_v2
):
    monkeypatch.delenv("VLLM_MORI_EP_V2", raising=False)
    monkeypatch.delenv("MORI_ENABLE_SDMA", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    parallel = dict(parallel)
    lsa_spans_hosts = parallel.pop("lsa_spans_hosts", False)
    sdma_picks_epv1 = parallel.pop("sdma_picks_epv1", True)
    validated_arch = parallel.pop("validated_arch", True)
    mori_reason = parallel.pop("mori_reason", None)
    monkeypatch.setattr(
        import_utils, "mori_epv2_unavailable_reason", lambda: mori_reason
    )
    parallel_config = _parallel_config(**parallel)
    model_config = _model_config(**model)
    resolve_mori_all2all(
        parallel_config,
        model_config,
        moe_backend,
        lsa_spans_hosts,
        sdma_picks_epv1,
        validated_arch,
    )
    assert parallel_config.mori_ep_v2 is ep_v2


@pytest.mark.parametrize(
    "modules,version,usable",
    [
        ({"mori", "mori.cco", "mori.ops.dispatch_combine_v2"}, "1.2.4", True),
        ({"mori", "mori.cco", "mori.ops.dispatch_combine_v2"}, "1.2.5.dev3+g1", True),
        ({"mori", "mori.cco", "mori.ops.dispatch_combine_v2"}, "1.2.3.post1", False),
        ({"mori", "mori.cco", "mori.ops.dispatch_combine_v2"}, "unknown", False),
        ({"mori"}, "1.2.4", False),
        (set(), "1.2.4", False),
    ],
)
def test_mori_epv2_needs_a_recent_mori(monkeypatch, modules, version, usable):
    monkeypatch.setattr(import_utils, "_has_module", lambda name: name in modules)
    monkeypatch.setitem(sys.modules, "mori", SimpleNamespace(__version__=version))
    assert (import_utils.mori_epv2_unavailable_reason() is None) is usable


@pytest.mark.parametrize(
    "backend,internode,max_tokens,kernel",
    [
        ("mori", False, 4096, "IntraNode"),
        ("mori", True, 256, "InterNodeV1LL"),
        ("mori", True, 257, "InterNodeV1"),
        ("mori_high_throughput", True, 128, "InterNodeV1"),
        ("mori_low_latency", True, 4096, "InterNodeV1LL"),
    ],
)
def test_mori_epv1_kernel_choice(monkeypatch, backend, internode, max_tokens, kernel):
    from vllm.distributed.device_communicators import all2all

    kernel_types = SimpleNamespace(
        IntraNode="IntraNode", InterNodeV1="InterNodeV1", InterNodeV1LL="InterNodeV1LL"
    )
    fake_mori = SimpleNamespace(
        ops=SimpleNamespace(EpDispatchCombineKernelType=kernel_types)
    )
    monkeypatch.setitem(sys.modules, "mori", fake_mori)
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms.rocm",
        SimpleNamespace(on_gfx942=lambda: False, on_gfx950=lambda: True),
    )
    monkeypatch.delenv(
        "VLLM_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD", raising=False
    )
    manager = object.__new__(all2all.MoriAll2AllManager)
    manager.internode = internode
    manager._all2all_backend = backend
    kwargs = manager._make_all2all_kwargs(
        rank=0,
        num_ep_ranks=16,
        input_dtype=torch.bfloat16,
        quant_dtype=torch.bfloat16,
        token_hidden_size=7168,
        scale_dim=0,
        scale_type_size=4,
        max_num_tokens_per_dp_rank=max_tokens,
        num_local_experts=16,
        num_experts_per_token=8,
    )
    assert kwargs["kernel_type"] == kernel
