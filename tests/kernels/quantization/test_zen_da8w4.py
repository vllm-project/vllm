# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the zentorch DA8W4 (W4A8) int4 linear and MoE paths on Zen CPUs.

The zentorch ops are mocked with reference implementations when zentorch is not
installed, so the layout and dispatch contracts are covered in CI.
"""

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

from tests.kernels.quant_utils import ref_dynamic_per_token_quant
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.zentorch import (
    ZentorchWNA16LinearKernel,
    _import_unpack_from_int32,
)
from vllm.model_executor.layers.fused_moe.oracle import int_wna16
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import WNA16MoEBackend
from vllm.scalar_type import scalar_types

GROUP_SIZE = 128
IN_FEATURES = 512
OUT_FEATURES = 128


def _unpack_s4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """int8 [N, K/2] (or int32 [N, K/8]) packed s4 -> float32 [N, K]."""
    words = packed.view(torch.int32)
    out = torch.zeros(words.shape[0], in_features, dtype=torch.float32)
    for i in range(8):
        nibble = (words >> (4 * i)) & 0xF
        out[:, i::8] = torch.where(nibble > 7, nibble - 16, nibble).float()
    return out


def _repack_s4(unpacked: torch.Tensor) -> torch.Tensor:
    """Reference zentorch_woq_repack_weight: int8 [N, K] -> int32 [N, K/8]."""
    n, k = unpacked.shape
    values = unpacked.to(torch.int32).reshape(n, k // 8, 8)
    out = torch.zeros(n, k // 8, dtype=torch.int32)
    for i in range(8):
        out |= (values[:, :, i] & 0xF) << (4 * i)
    return out


def _quantize_per_group(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int4 per-group quantization -> (int8 in [-8, 7], scale [N, G])."""
    out_features, in_features = weight.shape
    grouped = weight.float().reshape(
        out_features, in_features // group_size, group_size
    )
    scale = grouped.abs().amax(dim=-1) / 8.0
    quantized = (grouped / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    return quantized.reshape(out_features, in_features), scale.to(torch.bfloat16)


def _dequantize_per_group(
    quantized: torch.Tensor, scale: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Dequantize with [N, G] scales into a float32 [N, K] weight."""
    return quantized.float() * scale.float().repeat_interleave(group_size, dim=1)


def _dynamic_quant_matmul(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """DA8W4 reference: per-token int8 activation quant against a bf16 weight."""
    x_q, scale = ref_dynamic_per_token_quant(x.float(), torch.int8)
    out = (x_q.float() @ weight.t()) * scale
    if bias is not None:
        out = out + bias.float()
    return out


@pytest.fixture
def mock_zentorch_ops():
    """Register reference zentorch ops when zentorch is not installed."""
    if hasattr(torch.ops.zentorch, "zentorch_dynamic_qlinear"):
        yield None
        return

    calls: dict[str, tuple] = {}

    def _dynamic_qlinear(input, weight, weight_scales, bias=None, zentorch_op_name=""):
        calls["dynamic_qlinear"] = (input, weight, weight_scales, bias)
        in_features = weight.view(torch.int32).shape[1] * 8
        group_size = in_features // weight_scales.shape[0]
        # weight_scales is [G, N]; expand to a [N, K] dequantized weight.
        scales = weight_scales.float().t().repeat_interleave(group_size, dim=1)
        weight_deq = _unpack_s4(weight, in_features) * scales
        out = _dynamic_quant_matmul(input, weight_deq, bias)
        return out.to(input.dtype)

    def _fused_moe(
        output,
        input,
        w13,
        w2,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_id,
        skip_weighted,
        act,
        w13_scales=None,
        w2_scales=None,
        zentorch_op_name="",
    ):
        calls["fused_moe"] = (
            input,
            w13,
            w2,
            topk_weights,
            topk_id,
            skip_weighted,
            act,
            w13_scales,
            w2_scales,
        )
        output.zero_()

    lib_def = torch.library.Library("zentorch", "DEF")
    lib_def.define("zentorch_woq_repack_weight(Tensor unpacked_weight) -> Tensor")
    lib_def.define(
        "zentorch_dynamic_qlinear(Tensor input, Tensor weight, "
        "Tensor weight_scales, Tensor? bias=None, *, "
        "str zentorch_op_name='zentorch::zentorch_dynamic_qlinear') -> Tensor"
    )
    lib_def.define(
        "zentorch_fused_moe(Tensor(a!) output, Tensor input, Tensor w13, "
        "Tensor w2, Tensor? w13_bias, Tensor? w2_bias, Tensor topk_weights, "
        "Tensor topk_id, bool skip_weighted, str act, "
        "Tensor? w13_scales=None, Tensor? w2_scales=None, *, "
        "str zentorch_op_name='zentorch::zentorch_fused_moe') -> ()"
    )

    lib_impl = torch.library.Library("zentorch", "IMPL", "CPU")
    lib_impl.impl("zentorch_woq_repack_weight", _repack_s4)
    lib_impl.impl("zentorch_dynamic_qlinear", _dynamic_qlinear)
    lib_impl.impl("zentorch_fused_moe", _fused_moe)

    yield calls

    lib_impl._destroy()
    lib_def._destroy()


def _make_config(
    act_type: torch.dtype = torch.bfloat16,
    weight_type=scalar_types.uint4b8,
    zero_points: bool = False,
    group_size: int = GROUP_SIZE,
) -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=(IN_FEATURES, OUT_FEATURES),
        partition_weight_shape=(IN_FEATURES, OUT_FEATURES),
        weight_type=weight_type,
        act_type=act_type,
        group_size=group_size,
        zero_points=zero_points,
        has_g_idx=False,
    )


def _make_layer(group_size: int = GROUP_SIZE) -> tuple[torch.nn.Module, torch.Tensor]:
    """Build a layer holding a CT-packed int4 weight, plus its dequantized form."""
    torch.manual_seed(0)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES, dtype=torch.bfloat16)
    quantized, scale = _quantize_per_group(weight, group_size)

    # compressed-tensors stores [N, K//8] int32, packed along the input dim.
    packed = pack_to_int32(quantized, 4, packed_dim=1)

    layer = torch.nn.Module()
    layer.weight_packed = torch.nn.Parameter(packed, requires_grad=False)
    layer.weight_packed.input_dim = 1
    layer.weight_packed.packed_dim = 1
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    layer.weight_zero_point = None
    layer.weight_g_idx = None
    return layer, _dequantize_per_group(quantized, scale, group_size)


def _make_kernel(config: MPLinearLayerConfig) -> ZentorchWNA16LinearKernel:
    return ZentorchWNA16LinearKernel(
        config, "weight_packed", "weight_scale", "weight_zero_point", "weight_g_idx"
    )


# ---------------------------------------------------------------------------
# Dense DA8W4
# ---------------------------------------------------------------------------


def test_da8w4_eligible_for_symmetric_bf16_layer(mock_zentorch_ops, monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mixed_precision.zentorch."
        "current_platform.is_zen_cpu",
        lambda: True,
    )
    layer, _ = _make_layer()
    kernel = _make_kernel(_make_config())
    assert kernel._zentorch_da8w4_eligible(layer)


@pytest.mark.parametrize(
    "config_kwargs,reason",
    [
        ({"act_type": torch.float32}, "f32 activations are rejected"),
        (
            {"weight_type": scalar_types.uint4, "zero_points": True},
            "asymmetric int4 is unsupported",
        ),
    ],
)
def test_da8w4_not_eligible(mock_zentorch_ops, monkeypatch, config_kwargs, reason):
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mixed_precision.zentorch."
        "current_platform.is_zen_cpu",
        lambda: True,
    )
    layer, _ = _make_layer()
    kernel = _make_kernel(_make_config(**config_kwargs))
    assert not kernel._zentorch_da8w4_eligible(layer), reason


def test_da8w4_not_eligible_when_env_disabled(mock_zentorch_ops, monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mixed_precision.zentorch."
        "current_platform.is_zen_cpu",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mixed_precision.zentorch."
        "envs.VLLM_CPU_INT4_W4A8",
        False,
    )
    layer, _ = _make_layer()
    kernel = _make_kernel(_make_config())
    assert not kernel._zentorch_da8w4_eligible(layer)


def test_da8w4_process_weights_layout(mock_zentorch_ops, monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mixed_precision.zentorch."
        "current_platform.is_zen_cpu",
        lambda: True,
    )
    layer, _ = _make_layer()
    kernel = _make_kernel(_make_config())
    kernel.process_weights_after_loading(layer)

    assert layer._zentorch_da8w4
    assert layer._zentorch_kind == "compressed_tensors_w4a8_da8w4"
    # Packed s4 holds two nibbles per byte, and scales transpose to {G, N}.
    assert layer._zentorch_da8w4_packed.dtype == torch.int8
    assert layer._zentorch_da8w4_packed.shape == (OUT_FEATURES, IN_FEATURES // 2)
    assert layer._zentorch_da8w4_scale.dtype == torch.bfloat16
    assert layer._zentorch_da8w4_scale.shape == (
        IN_FEATURES // GROUP_SIZE,
        OUT_FEATURES,
    )
    assert layer._zentorch_da8w4_packed.is_contiguous()
    assert layer._zentorch_da8w4_scale.is_contiguous()
    # The checkpoint parameters are released once repacked.
    assert layer.weight_packed.numel() == 0
    assert layer.weight_scale.numel() == 0


def test_da8w4_apply_weights_matches_dequantized_reference(
    mock_zentorch_ops, monkeypatch
):
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mixed_precision.zentorch."
        "current_platform.is_zen_cpu",
        lambda: True,
    )
    layer, weight_deq = _make_layer()
    kernel = _make_kernel(_make_config())
    kernel.process_weights_after_loading(layer)

    x = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16)
    bias = torch.randn(OUT_FEATURES, dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x, bias)

    expected = _dynamic_quant_matmul(x, weight_deq, bias)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


def test_da8w4_falls_back_to_w4a16_when_op_missing(monkeypatch):
    """Without zentorch_dynamic_qlinear the layer must not take the DA8W4 path."""
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mixed_precision.zentorch.has_zentorch_op",
        lambda ops: "zentorch_dynamic_qlinear" not in ops,
    )
    layer, _ = _make_layer()
    kernel = _make_kernel(_make_config())
    assert not kernel._zentorch_da8w4_eligible(layer)


def test_can_implement_requires_zen_cpu(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mixed_precision.zentorch."
        "current_platform.is_zen_cpu",
        lambda: False,
    )
    ok, reason = ZentorchWNA16LinearKernel.can_implement(_make_config())
    assert not ok
    assert reason is not None


# ---------------------------------------------------------------------------
# MoE DA8W4
# ---------------------------------------------------------------------------

NUM_EXPERTS = 4
HIDDEN = 256
INTERMEDIATE = 128


def _make_moe_weights(group_size: int = 32):
    """Build CT-layout MoE weights: w13 [E, H//8, 2I], w2 [E, I//8, H]."""
    torch.manual_seed(0)
    w13_q, w13_s, w2_q, w2_s = [], [], [], []
    for _ in range(NUM_EXPERTS):
        w13 = torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
        q13, s13 = _quantize_per_group(w13, group_size)
        # CT packs along the input dim, storing [K//8, N] per expert.
        w13_q.append(pack_to_int32(q13.t().contiguous(), 4, packed_dim=0))
        w13_s.append(s13.t().contiguous())

        w2 = torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16)
        q2, s2 = _quantize_per_group(w2, group_size)
        w2_q.append(pack_to_int32(q2.t().contiguous(), 4, packed_dim=0))
        w2_s.append(s2.t().contiguous())
    return (
        torch.stack(w13_q),
        torch.stack(w2_q),
        torch.stack(w13_s),
        torch.stack(w2_s),
    )


def test_zen_cpu_first_in_cpu_backend_priority(monkeypatch):
    monkeypatch.setattr(int_wna16.current_platform, "is_cpu", lambda: True)
    backends = int_wna16._get_priority_backends()
    assert backends[0] == WNA16MoEBackend.ZEN_CPU
    # The generic CPU backend stays available as a fallback.
    assert WNA16MoEBackend.CPU in backends


def test_zen_cpu_process_weights_layout(mock_zentorch_ops):
    group_size = 32
    w13, w2, w13_scale, w2_scale = _make_moe_weights(group_size)
    assert w13.shape == (NUM_EXPERTS, HIDDEN // 8, 2 * INTERMEDIATE)
    assert w2.shape == (NUM_EXPERTS, INTERMEDIATE // 8, HIDDEN)

    converted = int_wna16._process_weights_zen_cpu(w13, w2, w13_scale, w2_scale)
    w13_out, w2_out, w13_s_out, w2_s_out = converted[:4]

    # zentorch consumes [E, N, K/8] packed s4 with per-group [E, G, N] scales.
    assert w13_out.shape == (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 8)
    assert w2_out.shape == (NUM_EXPERTS, HIDDEN, INTERMEDIATE // 8)
    assert w13_out.dtype == w2_out.dtype == torch.int32
    assert w13_s_out.shape == (NUM_EXPERTS, HIDDEN // group_size, 2 * INTERMEDIATE)
    assert w2_s_out.shape == (NUM_EXPERTS, INTERMEDIATE // group_size, HIDDEN)
    assert w13_s_out.dtype == w2_s_out.dtype == torch.bfloat16
    # Symmetric checkpoints carry no zero points into the kernel.
    assert converted[8] is None and converted[9] is None


def test_zen_cpu_repack_is_value_exact(mock_zentorch_ops):
    """The repacked weight must dequantize back to the checkpoint values."""
    group_size = 32
    w13, w2, w13_scale, w2_scale = _make_moe_weights(group_size)
    w13_out = int_wna16._process_weights_zen_cpu(w13, w2, w13_scale, w2_scale)[0]

    expected = _import_unpack_from_int32()(
        w13,
        4,
        torch.Size([NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE]),
        packed_dim=0,
    ).transpose(1, 2)
    for expert in range(NUM_EXPERTS):
        torch.testing.assert_close(
            _unpack_s4(w13_out[expert], HIDDEN),
            expected[expert].float(),
        )


@pytest.mark.parametrize(
    "group_size,expected_ok",
    [(32, True), (128, True), (-1, False), (2, False)],
)
def test_zen_cpu_group_size_gating(group_size, expected_ok):
    quant_config = type("Args", (), {"group_size": group_size})()
    reason = int_wna16._backend_incompatibility_reason(
        WNA16MoEBackend.ZEN_CPU,
        moe_config=None,
        quant_config=quant_config,
        may_have_zp=False,
        may_have_bias=False,
        allow_tile_padding=True,
    )
    assert (reason is None) == expected_ok


@pytest.mark.parametrize(
    "may_have_zp,may_have_bias,expected_ok",
    [(True, False, False), (False, True, True), (True, True, False)],
)
def test_zen_cpu_rejects_zero_points_but_takes_bias(
    may_have_zp, may_have_bias, expected_ok
):
    quant_config = type("Args", (), {"group_size": 128})()
    reason = int_wna16._backend_incompatibility_reason(
        WNA16MoEBackend.ZEN_CPU,
        moe_config=None,
        quant_config=quant_config,
        may_have_zp=may_have_zp,
        may_have_bias=may_have_bias,
        allow_tile_padding=True,
    )
    assert (reason is None) == expected_ok


def test_zen_cpu_rejects_moe_wna16_layout():
    from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config

    quant_config = MoeWNA16Config(
        linear_quant_method="gptq",
        weight_bits=4,
        group_size=128,
        has_zp=False,
        lm_head_quantized=False,
        modules_to_not_convert=None,
        full_config={},
    )
    reason = int_wna16._backend_incompatibility_reason(
        WNA16MoEBackend.ZEN_CPU,
        moe_config=None,
        quant_config=quant_config,
        may_have_zp=False,
        may_have_bias=False,
        allow_tile_padding=True,
    )
    assert reason is not None


def test_zen_cpu_disabled_by_env(monkeypatch):
    monkeypatch.setattr(int_wna16.envs, "VLLM_CPU_INT4_W4A8", False)
    quant_config = type("Args", (), {"group_size": 128})()
    reason = int_wna16._backend_incompatibility_reason(
        WNA16MoEBackend.ZEN_CPU,
        moe_config=None,
        quant_config=quant_config,
        may_have_zp=False,
        may_have_bias=False,
        allow_tile_padding=True,
    )
    assert reason is not None


def test_zen_cpu_backend_maps_to_experts_class():
    from vllm.model_executor.layers.fused_moe.experts.zentorch_moe import (
        ZentorchExpertsInt4,
    )

    assert int_wna16.backend_to_kernel_cls(WNA16MoEBackend.ZEN_CPU) == [
        ZentorchExpertsInt4
    ]


def test_zen_experts_support_predicates():
    from vllm.model_executor.kernels.linear.zentorch_utils import (
        _ZENTORCH_MOE_ACTIVATIONS,
    )
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.zentorch_moe import (
        ZentorchExpertsInt4,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt4Static,
        kInt4Static32,
        kInt8StaticChannelSym,
    )

    assert ZentorchExpertsInt4._supports_quant_scheme(kInt4Static, None)
    assert ZentorchExpertsInt4._supports_quant_scheme(kInt4Static32, None)
    assert not ZentorchExpertsInt4._supports_quant_scheme(kInt8StaticChannelSym, None)
    for act in MoEActivation:
        expected = act.value in _ZENTORCH_MOE_ACTIVATIONS
        assert ZentorchExpertsInt4._supports_activation(act) == expected
    assert ZentorchExpertsInt4.requires_interleaved_w13
    assert not ZentorchExpertsInt4._supports_no_act_and_mul()
    assert ZentorchExpertsInt4._supports_parallel_config(
        type("Par", (), {"use_ep": False})()
    )
    assert not ZentorchExpertsInt4._supports_parallel_config(
        type("Par", (), {"use_ep": True})()
    )


def test_zen_experts_take_custom_routing():
    """Models with their own router (gemma-4) reach select_experts through the
    config captured off the layer, since apply() cannot be handed the callable."""
    from types import SimpleNamespace

    from tests.kernels.moe.test_cpu_fused_moe import _StubMoELayer
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.fused_moe.experts.zentorch_moe import (
        ZentorchExpertsInt4,
    )

    assert ZentorchExpertsInt4._supports_routing_method(
        RoutingMethodType.Custom, None, None
    )

    def routing_fn(**kwargs):
        raise AssertionError("not called")

    layer = _StubMoELayer(
        torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        torch.zeros(NUM_EXPERTS, HIDDEN, INTERMEDIATE),
        MoEActivation.SILU,
    )
    layer.renormalize = True
    layer.custom_routing_function = routing_fn

    experts = SimpleNamespace(renormalize=False, custom_routing_function=None)
    ZentorchExpertsInt4.process_weights_after_loading(experts, layer)

    assert experts.custom_routing_function is routing_fn
    assert experts.renormalize is True


def _swigluoai_perm(activation, experts_cls=None):
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        swigluoai_w13_interleave_perm,
    )
    from vllm.model_executor.layers.fused_moe.experts.zentorch_moe import (
        ZentorchExpertsInt4,
    )

    return swigluoai_w13_interleave_perm(
        experts_cls or ZentorchExpertsInt4,
        activation,
        2 * INTERMEDIATE,
        torch.device("cpu"),
    )


def test_swigluoai_perm_only_for_zen_swigluoai():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    assert _swigluoai_perm(MoEActivation.SWIGLUOAI) is not None
    # Other activations keep the half-split layout the loader leaves.
    assert _swigluoai_perm(MoEActivation.SILU) is None
    # object stands in for any kernel that never set requires_interleaved_w13.
    assert _swigluoai_perm(MoEActivation.SWIGLUOAI, experts_cls=object) is None


def test_swigluoai_perm_interleaves_weights_scales_and_bias(mock_zentorch_ops):
    """The permutation is a gather on w13's output-channel axis, which is the
    last dim for the packed weights and their group scales."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    group_size = 32
    w13, w2, w13_scale, w2_scale = _make_moe_weights(group_size)
    bias = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, dtype=torch.bfloat16)
    perm = _swigluoai_perm(MoEActivation.SWIGLUOAI)

    baseline = int_wna16._process_weights_zen_cpu(w13, w2, w13_scale, w2_scale)[0]
    permuted = int_wna16._process_weights_zen_cpu(
        w13[..., perm].contiguous(), w2, w13_scale[..., perm].contiguous(), w2_scale
    )[0]

    for expert in range(NUM_EXPERTS):
        before = _unpack_s4(baseline[expert], HIDDEN)
        after = _unpack_s4(permuted[expert], HIDDEN)
        torch.testing.assert_close(after[0::2], before[:INTERMEDIATE])
        torch.testing.assert_close(after[1::2], before[INTERMEDIATE:])

    torch.testing.assert_close(bias[:, perm][:, 0::2], bias[:, :INTERMEDIATE])
    torch.testing.assert_close(bias[:, perm][:, 1::2], bias[:, INTERMEDIATE:])
