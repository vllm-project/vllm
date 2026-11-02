# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 activation dispatch for MoRI (ROCm).

MXFP4 w4a16 experts declare BF16 activations, but AITER quantizes them to
group-32 MXFP8 before stage 1 -- once on every rank a token was replicated to.
Quantizing in prepare() instead moves that to the rank that owns the token and
halves the dispatch payload.

The pre-quantized operand is only safe where AITER would have produced MXFP8
itself, so the gate asks AITER's own resolver. These tests pin the resolver's
answers against the gate's, check that each guard independently fails closed,
and cover the MoRI buffer sizing and the quant prepare() applies.
"""

import dataclasses
import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("MoRI MXFP8 dispatch is ROCm only.", allow_module_level=True)

from tests.kernels.moe.utils import make_dummy_moe_config  # noqa: E402
from vllm._aiter_ops import rocm_aiter_ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.all2all_utils import (  # noqa: E402
    _AITER_MOE_ACTIVATION_NAMES,
    MXFP8_GROUP_SIZE,
    maybe_make_prepare_finalize,
    mori_use_mxfp8_dispatch,
)
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    mxfp4_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
)
from vllm.platforms.rocm import on_gfx950  # noqa: E402
from vllm.utils.import_utils import has_mori  # noqa: E402

HIDDEN_DIM = 2048


def _aiter_moe_resolver_available() -> bool:
    try:
        from aiter import QuantType, dtypes  # noqa: F401
        from aiter.fused_moe import resolve_activation_dtype  # noqa: F401
        from aiter.ops.flydsl.moe_common import GateMode  # noqa: F401
    except ImportError:
        return False
    return True


# resolve_activation_dtype() only answers fp8 on gfx950; everywhere else the
# MXFP4 path keeps BF16 activations and there is nothing to pre-quantize.
requires_aiter_on_gfx950 = pytest.mark.skipif(
    not (on_gfx950() and _aiter_moe_resolver_available()),
    reason="needs an AITER build with resolve_activation_dtype() on gfx950",
)


@pytest.fixture
def aiter_moe_env(monkeypatch):
    """Set AITER's MoE knobs, yielding a setter for the per-test ones.

    VLLM_ROCM_USE_AITER_MOE_SITUV2 is mirrored into the AITER_SITUV2_* env
    AITER itself reads, and rocm_aiter_ops caches it on the class, so both have
    to be re-derived on every change and once more after monkeypatch rolls the
    environment back.
    """

    def configure(**env: str):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        rocm_aiter_ops.refresh_env_variables()

    monkeypatch.delenv("VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4", raising=False)
    monkeypatch.delenv("AITER_BF16_FP8_MOE_BOUND", raising=False)
    configure(
        VLLM_ROCM_USE_AITER="1",
        VLLM_ROCM_USE_AITER_MOE="1",
        VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS="0",
    )
    yield configure
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


def _w4a16_moe(activation=MoEActivation.SILU, hidden_dim=HIDDEN_DIM):
    """A w4a16 MoE config in the shape the DeepSeek-V4.1-Flash experts use."""
    return make_dummy_moe_config(
        num_experts=32,
        num_local_experts=16,
        experts_per_token=6,
        hidden_dim=hidden_dim,
        intermediate_size=512,
        activation=activation,
    )


def _w4a16_quant() -> FusedMoEQuantConfig:
    # Scales are never read here: nothing in this file runs an expert.
    return mxfp4_w4a16_moe_quant_config(w1_scale=None, w2_scale=None)


# Every expert configuration AiterExperts can reach from a w4a16 quant config,
# against the dtype AITER resolves its activations to. Only fp8 may be
# dispatched pre-quantized: a bf16 kernel would cast the payload back and never
# apply the scales, and an fp4x2 one would hand it to an MXFP4 quant that takes
# BF16/FP16 only. "bound" is AITER_BF16_FP8_MOE_BOUND, which decides the
# Silu/Swiglu-on-INTERLEAVE branch and which vLLM pins to 0 for these models.
_GATE_CASES = [
    # (activation, SiTUv2 mode, bound, gate mode, dtype AITER resolves)
    pytest.param(MoEActivation.SILU, None, "0", "INTERLEAVE", "fp8", id="silu"),
    pytest.param(MoEActivation.GELU, None, "0", "INTERLEAVE", "fp8", id="gelu"),
    pytest.param(MoEActivation.SWIGLUOAI, None, "0", "INTERLEAVE", "fp8", id="swiglu"),
    # Unpinned, the branch depends on the token count, which does not exist
    # when the dispatch buffers are sized: AITER declines to answer at all.
    pytest.param(MoEActivation.SILU, None, None, "INTERLEAVE", None, id="unbounded"),
    # SiTU reads AITER_SITUV2_* instead, and only a8w4 resolves to fp8. Asking
    # about a stand-in activation here reads AITER_BF16_FP8_MOE_BOUND, which
    # says nothing about these experts -- that is how a4w4 was let through.
    pytest.param(MoEActivation.SITU, "a8w4", "0", "INTERLEAVE", "fp8", id="situ-a8w4"),
    pytest.param(MoEActivation.SITU, "a4w4", "0", "SEPARATED", "fp4x2", id="situ-a4w4"),
    pytest.param(
        MoEActivation.SITU, "a16w4", "0", "SEPARATED", "bf16", id="situ-a16w4"
    ),
]


@requires_aiter_on_gfx950
@pytest.mark.parametrize("activation,situv2,bound,gate_mode,resolved", _GATE_CASES)
def test_gate_follows_aiter_activation_dtype(
    aiter_moe_env, activation, situv2, bound, gate_mode, resolved
):
    """The gate must turn on exactly where AITER resolves these experts to fp8.

    Asserting AITER's answer as well as the gate's keeps the two failures
    apart: AITER changing its mind about a configuration reads differently from
    the gate drifting away from it.
    """
    env = {} if bound is None else {"AITER_BF16_FP8_MOE_BOUND": bound}
    if situv2 is not None:
        env["VLLM_ROCM_USE_AITER_MOE_SITUV2"] = situv2
    aiter_moe_env(**env)

    from aiter import QuantType, dtypes
    from aiter.fused_moe import resolve_activation_dtype
    from aiter.ops.flydsl.moe_common import GateMode

    expected_dtype = None if resolved is None else getattr(dtypes, resolved)
    assert expected_dtype == resolve_activation_dtype(
        QuantType.per_1x32,
        dtypes.fp4x2,
        activation=rocm_aiter_ops.get_aiter_activation_type(
            _AITER_MOE_ACTIVATION_NAMES[activation]
        ),
        gate_mode=getattr(GateMode, gate_mode),
        M=None,
    ), f"AITER no longer resolves this configuration's activations to {resolved}"

    enabled = mori_use_mxfp8_dispatch(_w4a16_moe(activation=activation), _w4a16_quant())
    assert enabled is (expected_dtype == dtypes.fp8), (
        f"the dispatch must follow AITER, which quantizes {activation.value} "
        f"activations to {resolved} here"
    )


@requires_aiter_on_gfx950
@pytest.mark.parametrize(
    "case",
    ["w4a4", "fp8_block", "dsv4_a4w4", "ragged_hidden_dim", "foreign_activation"],
)
def test_gate_fails_closed(aiter_moe_env, case):
    """Each guard alone must turn off a configuration that is otherwise on.

    None of these resolve to fp8 through the query above, so dispatching a
    pre-quantized payload would either lose the scales or hand an FP8 tensor
    to a quant kernel that only accepts BF16/FP16.
    """
    aiter_moe_env(AITER_BF16_FP8_MOE_BOUND="0")

    moe, quant = _w4a16_moe(), _w4a16_quant()
    assert mori_use_mxfp8_dispatch(moe, quant), (
        "the DeepSeek-V4.1-Flash shape (w4a16, silu, hidden 2048) with the "
        "pinned FP8 bound is the configuration this dispatch exists for, so "
        f"the {case} guard below has nothing to turn off"
    )

    if case == "w4a4":
        # The experts already consume dispatched MXFP4; MXFP8 is not an option.
        quant = mxfp4_moe_quant_config(w1_scale=None, w2_scale=None)
    elif case == "fp8_block":
        # Not an MXFP4 model at all -- the per-1x128 FP8 dispatch owns this.
        quant = fp8_w8a8_moe_quant_config(
            w1_scale=None, w2_scale=None, block_shape=[128, 128]
        )
    elif case == "dsv4_a4w4":
        # This opt-in overrides q_dtype_a on the fused_moe() call itself, so no
        # query of the resolver can see the decision.
        moe = dataclasses.replace(moe, use_mxfp4_w4a4_dsv4=True)
    elif case == "ragged_hidden_dim":
        # E8M0 scales cover exactly 32 elements; a partial group has no scale.
        moe = _w4a16_moe(hidden_dim=HIDDEN_DIM + 16)
    elif case == "foreign_activation":
        # AiterExperts rejects this activation, so nothing resolves a dtype.
        moe = _w4a16_moe(activation=MoEActivation.SWIGLUSTEP)

    assert not mori_use_mxfp8_dispatch(moe, quant), (
        f"{case} must fall back to the BF16 dispatch: AITER would not have "
        "produced MXFP8 activations for it"
    )


class _RecordingAll2AllManager:
    """Captures the buffer description MoRI would be initialized with."""

    rank = 0
    world_size = 2
    dp_world_size = 2

    def __init__(self):
        self.all_to_all_args: dict | None = None

    def get_handle(self, all_to_all_args):
        self.all_to_all_args = all_to_all_args
        return SimpleNamespace()


def _mori_moe():
    """The w4a16 config above, deployed over two EP ranks on MoRI."""
    return dataclasses.replace(
        _w4a16_moe(),
        moe_parallel_config=dataclasses.replace(
            FusedMoEParallelConfig.make_no_parallel(),
            dp_size=2,
            ep_size=2,
            use_ep=True,
            all2all_backend="mori_high_throughput",
        ),
    )


requires_mori = pytest.mark.skipif(not has_mori(), reason="mori is not installed")


def _make_mori_dispatch():
    """Return the MoRI adapter and the buffer args MoRI would be built with."""
    from vllm.model_executor.layers.fused_moe.prepare_finalize.mori import (
        MoriPrepareAndFinalize,
    )

    manager = _RecordingAll2AllManager()
    prepare_finalize = maybe_make_prepare_finalize(
        _mori_moe(), _w4a16_quant(), all2all_manager=manager
    )
    assert isinstance(prepare_finalize, MoriPrepareAndFinalize)
    assert manager.all_to_all_args is not None
    return prepare_finalize, manager.all_to_all_args


@requires_aiter_on_gfx950
@requires_mori
def test_mori_buffers_carry_fp8_tokens_and_byte_scales(aiter_moe_env):
    """MoRI sizes its buffers from these args, so they must describe MXFP8.

    E8M0 exponents are one byte each, not the fp32 the per-token and per-tensor
    FP8 dispatches transport, and there is one per 32-element group.
    """
    aiter_moe_env(AITER_BF16_FP8_MOE_BOUND="0")
    prepare_finalize, args = _make_mori_dispatch()

    assert prepare_finalize.use_mxfp8_dispatch
    assert not prepare_finalize.use_fp8_dispatch
    assert not prepare_finalize.use_fp4_dispatch

    assert args["quant_dtype"] == current_platform.fp8_dtype()
    # MXFP8 is one byte per element and is not packed, so the token buffer
    # stays the full hidden size (unlike the FP4 dispatch's two-per-byte).
    assert args["token_hidden_size"] == HIDDEN_DIM
    assert args["scale_dim"] == HIDDEN_DIM // MXFP8_GROUP_SIZE
    assert args["scale_type_size"] == torch.float8_e8m0fnu.itemsize == 1


@requires_aiter_on_gfx950
@requires_mori
def test_mori_buffers_stay_bf16_when_the_gate_is_off(aiter_moe_env):
    """With the gate off nothing is pre-quantized, so no scales are carried."""
    prepare_finalize, args = _make_mori_dispatch()

    assert not prepare_finalize.use_mxfp8_dispatch
    assert args["quant_dtype"] == torch.bfloat16
    assert args["scale_dim"] == 0
    assert args["scale_type_size"] == 0


def _import_mori_prepare_finalize(monkeypatch):
    """Import the adapter against a stub MoRI, as test_mori_prepare_finalize."""
    fake_mori = types.ModuleType("mori")
    fake_mori.ops = SimpleNamespace(  # type: ignore[attr-defined]
        EpDispatchCombineOp=object
    )
    monkeypatch.setitem(sys.modules, "mori", fake_mori)
    module_name = "vllm.model_executor.layers.fused_moe.prepare_finalize.mori"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class _RecordingMoriOp:
    """Records the payload handed to dispatch() and echoes it back."""

    def __init__(self):
        self.a1 = None
        self.scale = None

    def dispatch(self, a1, weights, scale, indices):
        self.a1, self.scale = a1, scale
        return a1, weights, scale, indices, torch.tensor([a1.shape[0]])


def _prepare(adapter, a1, **kwargs):
    return adapter.prepare(
        a1,
        torch.ones(a1.shape[0], 2, device=a1.device),
        torch.zeros(a1.shape[0], 2, dtype=torch.int32, device=a1.device),
        num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=_w4a16_quant(),
        **kwargs,
    )


@requires_aiter_on_gfx950
def test_prepare_dispatches_mxfp8_with_e8m0_scales(monkeypatch):
    """prepare() must emit exactly what AITER's stage 1 expects to sort.

    An FP8 payload with one E8M0 exponent per 32-element group: AITER sorts
    those scales instead of running its own quant, so a wrong layout here
    silently corrupts the scaling rather than being rejected.
    """
    mori_module = _import_mori_prepare_finalize(monkeypatch)
    mori_op = _RecordingMoriOp()
    adapter = mori_module.MoriPrepareAndFinalize(
        mori_op=mori_op,
        max_tokens_per_rank=16,
        num_dispatchers=1,
        use_mxfp8_dispatch=True,
    )

    num_tokens = 7
    torch.manual_seed(0)
    a1 = torch.randn(num_tokens, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    dispatch_a1, dispatch_scale = _prepare(adapter, a1)[:2]

    assert dispatch_a1.dtype == current_platform.fp8_dtype()
    assert dispatch_a1.shape == (num_tokens, HIDDEN_DIM)
    assert dispatch_scale.dtype == torch.float8_e8m0fnu
    assert dispatch_scale.shape == (
        num_tokens,
        HIDDEN_DIM // MXFP8_GROUP_SIZE,
    )

    # The payload still has to mean what it did before the dispatch moved, so
    # each element must land on the E4M3 grid its own group's exponent defines:
    # steps of scale * 2**-9 below E4M3's smallest normal, 2**-3 relative above
    # it. Pairing the scales with the wrong group would blow past that.
    groups = dispatch_a1.to(torch.float32).view(num_tokens, -1, MXFP8_GROUP_SIZE)
    scales = torch.exp2(
        dispatch_scale.view(torch.uint8).to(torch.float32) - 127.0
    ).unsqueeze(-1)
    reference = a1.to(torch.float32).view(num_tokens, -1, MXFP8_GROUP_SIZE)
    step = torch.maximum(scales * 2**-9, reference.abs() * 2**-3)

    error = (groups * scales - reference).abs()
    assert (error <= step).all(), (
        "dequantizing the dispatched payload with its own scales must recover "
        f"the activations to MXFP8 precision, but the worst element is off by "
        f"{(error / step).max().item():.2f} E4M3 steps"
    )


@requires_aiter_on_gfx950
def test_prepare_leaves_quant_to_the_experts_when_deferred(monkeypatch):
    """defer_input_quant means the experts quantize, so dispatch stays BF16."""
    mori_module = _import_mori_prepare_finalize(monkeypatch)
    mori_op = _RecordingMoriOp()
    adapter = mori_module.MoriPrepareAndFinalize(
        mori_op=mori_op,
        max_tokens_per_rank=16,
        num_dispatchers=1,
        use_mxfp8_dispatch=True,
    )

    a1 = torch.randn(4, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    _prepare(adapter, a1, defer_input_quant=True)

    assert mori_op.a1 is a1
    assert mori_op.scale is None


@pytest.mark.parametrize(
    "modes",
    [
        {"use_fp8_dispatch": True, "use_mxfp8_dispatch": True},
        {"use_fp4_dispatch": True, "use_mxfp8_dispatch": True},
        {"use_fp8_dispatch": True, "use_fp4_dispatch": True},
    ],
)
def test_dispatch_modes_are_mutually_exclusive(monkeypatch, modes):
    """Each mode quantizes a1 in place, so two of them would stack."""
    mori_module = _import_mori_prepare_finalize(monkeypatch)
    with pytest.raises(AssertionError):
        mori_module.MoriPrepareAndFinalize(
            mori_op=_RecordingMoriOp(),
            max_tokens_per_rank=16,
            num_dispatchers=1,
            **modes,
        )
