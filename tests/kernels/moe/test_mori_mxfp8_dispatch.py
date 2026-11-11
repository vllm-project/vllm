# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 activation dispatch for MoRI (ROCm).

AITER's MXFP4 w4a16 experts take BF16 input and quantize it to group-32 MXFP8
before stage 1 -- once on every rank a token was replicated to. Quantizing in
prepare() instead moves that to the rank that owns the token and halves the
dispatch payload.

Whether to comes from the MoE activation key: the model's default (MXFP8 for
DeepSeek-V4.1-Flash) or the user's quantization_config.moe.activation, which
takes precedence. These tests cover that choice, the MoRI buffer sizing, and
the quant prepare() applies.
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
from vllm.model_executor.layers.fused_moe.all2all_utils import (  # noqa: E402
    MXFP8_GROUP_SIZE,
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    mxfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    kMxfp4Dynamic,
    kMxfp8Dynamic,
)
from vllm.platforms.rocm import on_gfx950  # noqa: E402
from vllm.utils.import_utils import has_mori  # noqa: E402

HIDDEN_DIM = 2048


def _aiter_available() -> bool:
    try:
        import aiter  # noqa: F401
    except ImportError:
        return False
    return True


# AITER's MXFP4 w4a16 experts, and so the MXFP8 they run, are gfx950 only.
requires_aiter_on_gfx950 = pytest.mark.skipif(
    not (on_gfx950() and _aiter_available()),
    reason="needs AITER on gfx950",
)


@pytest.fixture
def aiter_moe_env(monkeypatch):
    """Enable AITER MoE, which MoRI EPv1 requires.

    rocm_aiter_ops caches the env on the class, so it has to be re-derived
    after the change and once more after monkeypatch rolls it back.
    """
    monkeypatch.delenv("VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4", raising=False)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "0")
    rocm_aiter_ops.refresh_env_variables()
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


def _w4a16_moe():
    """A w4a16 MoE config in the shape the DeepSeek-V4.1-Flash experts use."""
    return make_dummy_moe_config(
        num_experts=32,
        num_local_experts=16,
        experts_per_token=6,
        hidden_dim=HIDDEN_DIM,
        intermediate_size=512,
    )


def _w4a16_quant(
    deferred_activation_key=None,
) -> FusedMoEQuantConfig:
    # Scales are never read here: nothing in this file runs an expert.
    return dataclasses.replace(
        mxfp4_w4a16_moe_quant_config(w1_scale=None, w2_scale=None),
        deferred_activation_key=deferred_activation_key,
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


def _make_mori_dispatch(quant_config: FusedMoEQuantConfig):
    """Return the MoRI adapter and the buffer args MoRI would be built with."""
    from vllm.model_executor.layers.fused_moe.prepare_finalize.mori import (
        MoriPrepareAndFinalize,
    )

    manager = _RecordingAll2AllManager()
    prepare_finalize = maybe_make_prepare_finalize(
        _mori_moe(), quant_config, all2all_manager=manager
    )
    assert isinstance(prepare_finalize, MoriPrepareAndFinalize)
    assert manager.all_to_all_args is not None
    return prepare_finalize, manager.all_to_all_args


# quantization_config from DeepSeek-V4.1-Flash's config.json.
DSV41_FLASH_QUANT_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}


def _dsv41_flash_experts_quant_config(monkeypatch, moe_activation):
    """The quant config DeepSeek-V4.1-Flash's routed experts build, with
    quantization_config.moe.activation set to `moe_activation` (or unset)."""
    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        AiterMxfp4Experts,
    )
    from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as mxfp4_oracle
    from vllm.model_executor.layers.quantization import mxfp4 as mxfp4_quant
    from vllm.models.deepseek_v41 import quant_config as dsv41_quant_config

    user_args = (
        None
        if moe_activation is None
        else QuantizationConfigArgs(moe={"activation": moe_activation})
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                expert_dtype="fp4", quantization_config=DSV41_FLASH_QUANT_CONFIG
            ),
            quantization_config=user_args,
        )
    )
    monkeypatch.setattr(
        dsv41_quant_config, "get_current_vllm_config", lambda: vllm_config
    )
    monkeypatch.setattr(mxfp4_oracle, "get_current_vllm_config", lambda: vllm_config)
    # What MI355X selects for these experts, without depending on the device
    # or AITER build the test runs on.
    monkeypatch.setattr(
        mxfp4_quant,
        "select_deepseek_v4_mxfp4_moe_backend",
        lambda moe: (mxfp4_oracle.Mxfp4MoeBackend.AITER_MXFP4_BF16, AiterMxfp4Experts),
    )
    monkeypatch.setattr(
        rocm_aiter_ops, "fused_moe_supports_quant_dtype_a", lambda: True
    )

    quant_config = dsv41_quant_config.DeepseekV4FP8Config.from_config(
        dict(DSV41_FLASH_QUANT_CONFIG)
    )
    # get_quant_method() reads nothing from the layer but its moe_config.
    layer = RoutedExperts.__new__(RoutedExperts)
    layer.__dict__["moe_config"] = _mori_moe()
    method = quant_config.get_quant_method(layer, prefix="model.layers.3.ffn.experts")
    assert isinstance(method, mxfp4_quant.Mxfp4MoEMethod)
    return method.get_fused_moe_quant_config(
        SimpleNamespace(w13_weight_scale=None, w2_weight_scale=None)
    )


@requires_aiter_on_gfx950
@requires_mori
@pytest.mark.parametrize(
    "moe_activation,dispatch_dtype",
    [
        # The checkpoint names no activation scheme for its experts, so this is
        # the MXFP8 DeepseekV4FP8Config defaults to.
        pytest.param(None, current_platform.fp8_dtype(), id="checkpoint-default"),
        # These experts cannot run MXFP4 activations, so the override turns the
        # MXFP8 default off rather than being outvoted by it.
        pytest.param("mxfp4", torch.bfloat16, id="user-override"),
    ],
)
def test_dsv41_flash_dispatch_dtype(
    aiter_moe_env, monkeypatch, moe_activation, dispatch_dtype
):
    """The dispatch dtype follows the MoE activation key, user over model."""
    quant_config = _dsv41_flash_experts_quant_config(monkeypatch, moe_activation)
    prepare_finalize, args = _make_mori_dispatch(quant_config)

    assert args["quant_dtype"] == dispatch_dtype
    assert prepare_finalize.use_mxfp8_dispatch is (dispatch_dtype != torch.bfloat16)


@requires_aiter_on_gfx950
@requires_mori
def test_mori_buffers_carry_fp8_tokens_and_byte_scales(aiter_moe_env):
    """MoRI sizes its buffers from these args, so they must describe MXFP8.

    E8M0 exponents are one byte each, not the fp32 the per-token and per-tensor
    FP8 dispatches transport, and there is one per 32-element group.
    """
    prepare_finalize, args = _make_mori_dispatch(_w4a16_quant(kMxfp8Dynamic))

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
@pytest.mark.parametrize("deferred_activation_key", [None, kMxfp4Dynamic])
def test_mori_buffers_stay_bf16_without_mxfp8_key(
    aiter_moe_env, deferred_activation_key
):
    """Without MXFP8 experts nothing is pre-quantized, so no scales are carried."""
    prepare_finalize, args = _make_mori_dispatch(_w4a16_quant(deferred_activation_key))

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
