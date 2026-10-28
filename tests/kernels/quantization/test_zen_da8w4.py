# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the zentorch DA8W4 (W4A8) int4 linear and MoE paths on Zen CPUs.

The zentorch ops are mocked with reference implementations when zentorch is not
installed, so the layout and dispatch contracts are covered in CI.
"""

import dataclasses

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
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

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
    weight: torch.Tensor, group_size: int, num_bits: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-group quantization -> (int8 holding one s{num_bits} value
    per element, scale [N, G])."""
    qmax = 1 << (num_bits - 1)
    out_features, in_features = weight.shape
    grouped = weight.float().reshape(
        out_features, in_features // group_size, group_size
    )
    scale = grouped.abs().amax(dim=-1) / qmax
    quantized = (
        (grouped / scale.unsqueeze(-1)).round().clamp(-qmax, qmax - 1).to(torch.int8)
    )
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
    return layer, _dequantize_per_group(quantized, scale, group_size)


def _make_kernel(config: MPLinearLayerConfig) -> ZentorchWNA16LinearKernel:
    return ZentorchWNA16LinearKernel(
        config, "weight_packed", "weight_scale", "weight_zero_point"
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
    assert kernel._maybe_process_da8w4_weights(layer)


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
    assert not kernel._maybe_process_da8w4_weights(layer), reason


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
    assert not kernel._maybe_process_da8w4_weights(layer)


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
    assert not kernel._maybe_process_da8w4_weights(layer)


def test_can_implement_requires_zen_cpu(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mixed_precision.zentorch."
        "current_platform.is_zen_cpu",
        lambda: False,
    )
    ok, reason = ZentorchWNA16LinearKernel.can_implement(_make_config())
    assert not ok
    assert reason is not None


def test_can_implement_accepts_shapes_cpuwna16_rejects(monkeypatch):
    """CPUWNA16's N/K % 32 rule is a oneDNN packing constraint zentorch lacks."""
    from vllm.model_executor.kernels.linear.mixed_precision.cpu import (
        CPUWNA16LinearKernel,
    )

    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mixed_precision.zentorch."
        "current_platform.is_zen_cpu",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mixed_precision.cpu."
        "current_platform.is_cpu",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mixed_precision.zentorch.has_zentorch_op",
        lambda ops: True,
    )

    unaligned = dataclasses.replace(
        _make_config(), partition_weight_shape=(IN_FEATURES, OUT_FEATURES - 16)
    )
    assert unaligned.partition_weight_shape[1] % 32 != 0

    ok, _ = CPUWNA16LinearKernel.can_implement(unaligned)
    assert not ok, "the parent is expected to turn this shape away"

    ok, reason = ZentorchWNA16LinearKernel.can_implement(unaligned)
    assert ok, reason


@pytest.mark.parametrize(
    "group_size,expected_ok",
    [
        (-1, True),  # per-channel, one group spanning K
        (GROUP_SIZE, True),
        (IN_FEATURES, True),
        (6, False),  # even, so the parent's multiple-of-2 rule would allow it
        (0, False),
    ],
)
def test_can_implement_group_size_must_divide_k(monkeypatch, group_size, expected_ok):
    """ZenDNN WOQ groups along K, so a group that straddles the end is unusable."""
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mixed_precision.zentorch."
        "current_platform.is_zen_cpu",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mixed_precision.zentorch.has_zentorch_op",
        lambda ops: True,
    )

    ok, reason = ZentorchWNA16LinearKernel.can_implement(
        _make_config(group_size=group_size)
    )
    assert ok is expected_ok, reason
    if not expected_ok:
        assert reason is not None
        assert "must divide input size" in reason


# ---------------------------------------------------------------------------
# MoE DA8W4
# ---------------------------------------------------------------------------

NUM_EXPERTS = 4
HIDDEN = 256
INTERMEDIATE = 128


def _make_moe_weights(group_size: int = 32, num_bits: int = 4):
    """Build CT-layout MoE weights: w13 [E, 2I, H//pack], w2 [E, H, I//pack],
    with pack = 32 // num_bits values per int32 word."""
    torch.manual_seed(0)
    w13_q, w13_s, w2_q, w2_s = [], [], [], []
    for _ in range(NUM_EXPERTS):
        w13 = torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
        q13, s13 = _quantize_per_group(w13, group_size, num_bits)
        # CT is N-first and packs along the input dim: [N, K//pack] per expert.
        w13_q.append(pack_to_int32(q13, num_bits, packed_dim=1))
        w13_s.append(s13)

        w2 = torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16)
        q2, s2 = _quantize_per_group(w2, group_size, num_bits)
        w2_q.append(pack_to_int32(q2, num_bits, packed_dim=1))
        w2_s.append(s2)
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
    assert w13.shape == (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 8)
    assert w2.shape == (NUM_EXPERTS, HIDDEN, INTERMEDIATE // 8)

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
        torch.Size([NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN]),
        packed_dim=1,
    )
    for expert in range(NUM_EXPERTS):
        torch.testing.assert_close(
            _unpack_s4(w13_out[expert], HIDDEN),
            expected[expert].float(),
        )


def test_zen_cpu_int8_decode_keeps_full_width_values(mock_zentorch_ops):
    """Grouped W8A16 decodes to full-width s8, which is the trailing dim that
    makes the op read it as DA8W8; CT packs biased unsigned fields, so the
    values have to survive as two's complement."""
    group_size = 32
    weight = torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
    quantized, scale = _quantize_per_group(weight, group_size, num_bits=8)
    w13 = torch.stack([pack_to_int32(quantized, 8, packed_dim=1)] * NUM_EXPERTS)
    w13_scale = torch.stack([scale] * NUM_EXPERTS)
    _, w2, _, w2_scale = _make_moe_weights(group_size, num_bits=8)

    w13_out = int_wna16._process_weights_zen_cpu(
        w13, w2, w13_scale, w2_scale, num_bits=8
    )[0]

    assert w13_out.shape == (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    assert w13_out.dtype == torch.int8
    for expert in range(NUM_EXPERTS):
        torch.testing.assert_close(w13_out[expert], quantized)


@pytest.mark.parametrize("num_bits", [4, 8])
def test_zen_cpu_conversion_logs_the_width(mock_zentorch_ops, caplog, num_bits):
    """VLLM_CPU_INT4_W4A8 reads as int4-only, so the width it actually serves
    has to be visible at load time."""
    w13, w2, w13_scale, w2_scale = _make_moe_weights(32, num_bits=num_bits)

    import vllm.logger

    vllm.logger._print_info_once.cache_clear()
    with caplog.at_level("INFO"):
        int_wna16.convert_to_wna16_moe_kernel_format(
            backend=WNA16MoEBackend.ZEN_CPU,
            layer=torch.nn.Module(),
            quant_config=_ct_quant_args(num_bits=num_bits, group_size=32),
            input_dtype=torch.bfloat16,
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
        )

    assert f"DA8W{num_bits}" in caplog.text
    assert f"W{num_bits}A16" in caplog.text


def _ct_quant_args(num_bits: int = 4, group_size: int | None = 128):
    """A real compressed-tensors config: the Zen gate reads the width by
    checkpoint format, so a duck-typed stub reads as an unknown format."""
    from compressed_tensors.quantization import QuantizationArgs

    if group_size is None or group_size <= 0:
        return QuantizationArgs(num_bits=num_bits, strategy="channel", symmetric=True)
    return QuantizationArgs(
        num_bits=num_bits, group_size=group_size, strategy="group", symmetric=True
    )


@pytest.mark.parametrize(
    "num_bits,expected_ok",
    [(8, True), (2, False)],
)
def test_zen_cpu_num_bits_gating(num_bits, expected_ok):
    reason = int_wna16._backend_incompatibility_reason(
        WNA16MoEBackend.ZEN_CPU,
        moe_config=None,
        quant_config=_ct_quant_args(num_bits=num_bits),
        may_have_zp=False,
        may_have_bias=False,
        allow_tile_padding=True,
    )
    assert (reason is None) == expected_ok


def test_zen_cpu_rejects_unknown_checkpoint_format():
    """No recognised width means no safe decode, so selection has to decline
    and leave the layer to another backend."""
    reason = int_wna16._backend_incompatibility_reason(
        WNA16MoEBackend.ZEN_CPU,
        moe_config=None,
        quant_config=type("Args", (), {"group_size": 128, "num_bits": 4})(),
        may_have_zp=False,
        may_have_bias=False,
        allow_tile_padding=True,
    )
    # Naming the format beats reporting a width of None we never read.
    assert reason is not None
    assert "Args" in reason


@pytest.mark.parametrize(
    "group_size,expected_ok",
    [(32, True), (128, True), (-1, False), (2, False)],
)
def test_zen_cpu_group_size_gating(group_size, expected_ok):
    reason = int_wna16._backend_incompatibility_reason(
        WNA16MoEBackend.ZEN_CPU,
        moe_config=None,
        quant_config=_ct_quant_args(group_size=group_size),
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
    reason = int_wna16._backend_incompatibility_reason(
        WNA16MoEBackend.ZEN_CPU,
        moe_config=None,
        quant_config=_ct_quant_args(),
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
    reason = int_wna16._backend_incompatibility_reason(
        WNA16MoEBackend.ZEN_CPU,
        moe_config=None,
        quant_config=_ct_quant_args(),
        may_have_zp=False,
        may_have_bias=False,
        allow_tile_padding=True,
    )
    assert reason is not None


def test_zen_cpu_backend_maps_to_experts_class():
    from vllm.model_executor.layers.fused_moe.experts.zentorch_moe import (
        ZentorchExpertsWNA16,
    )

    assert int_wna16.backend_to_kernel_cls(WNA16MoEBackend.ZEN_CPU) == [
        ZentorchExpertsWNA16
    ]


def test_zen_experts_support_predicates():
    from vllm.model_executor.kernels.linear.zentorch_utils import (
        _ZENTORCH_MOE_ACTIVATIONS,
    )
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.zentorch_moe import (
        ZentorchExpertsWNA16,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt4Static,
        kInt4Static32,
        kInt8Static,
        kInt8StaticChannelSym,
    )

    assert ZentorchExpertsWNA16._supports_quant_scheme(kInt4Static, None)
    assert ZentorchExpertsWNA16._supports_quant_scheme(kInt4Static32, None)
    assert ZentorchExpertsWNA16._supports_quant_scheme(kInt8Static, None)
    # Grouped int8 only; a per-channel int8 weight has no group scales to match.
    assert not ZentorchExpertsWNA16._supports_quant_scheme(kInt8StaticChannelSym, None)
    for act in MoEActivation:
        expected = act.value in _ZENTORCH_MOE_ACTIVATIONS
        assert ZentorchExpertsWNA16._supports_activation(act) == expected
    assert ZentorchExpertsWNA16.requires_interleaved_w13
    assert not ZentorchExpertsWNA16._supports_no_act_and_mul()
    assert ZentorchExpertsWNA16._supports_parallel_config(
        type("Par", (), {"use_ep": False})()
    )
    assert not ZentorchExpertsWNA16._supports_parallel_config(
        type("Par", (), {"use_ep": True})()
    )


def test_zen_experts_score_every_routing_method_they_accept():
    """Quant methods other than compressed-tensors never call
    process_weights_after_loading, so the constructor defaults decide how
    apply() routes. Accepting a method without a scoring entry would silently
    route it as softmax."""
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.fused_moe.experts import zentorch_moe

    for routing_method in RoutingMethodType:
        if routing_method == RoutingMethodType.Custom:
            continue
        accepted = zentorch_moe.ZentorchExpertsWNA16._supports_routing_method(
            routing_method, None, None
        )
        assert accepted == (routing_method in zentorch_moe._ROUTING_SCORING)

    # Laguna and MiniMax-M2 renormalize a biased sigmoid, not a softmax.
    assert zentorch_moe._ROUTING_SCORING[RoutingMethodType.MiniMax2] == (
        "sigmoid",
        True,
    )


def test_zen_experts_take_routing_config_off_the_layer():
    """Models with their own router (gemma-4) or a non-softmax scoring function
    (Laguna) reach select_experts through the config captured off the layer,
    since apply() cannot be handed either."""
    from types import SimpleNamespace

    from tests.kernels.moe.test_cpu_fused_moe import _StubMoELayer
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.fused_moe.experts.zentorch_moe import (
        ZentorchExpertsWNA16,
    )

    assert ZentorchExpertsWNA16._supports_routing_method(
        RoutingMethodType.Custom, None, None
    )
    assert ZentorchExpertsWNA16._supports_routing_method(
        RoutingMethodType.MiniMax2, None, None
    )

    def routing_fn(**kwargs):
        raise AssertionError("not called")

    layer = _StubMoELayer(
        torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        torch.zeros(NUM_EXPERTS, HIDDEN, INTERMEDIATE),
        MoEActivation.SILU,
    )
    layer.renormalize = True
    layer.scoring_func = "sigmoid"
    layer.custom_routing_function = routing_fn

    experts = SimpleNamespace(
        renormalize=False, scoring_func="softmax", custom_routing_function=None
    )
    ZentorchExpertsWNA16.process_weights_after_loading(experts, layer)

    assert experts.custom_routing_function is routing_fn
    assert experts.renormalize is True
    assert experts.scoring_func == "sigmoid"


def _swigluoai_perm(activation, experts_cls=None):
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        swigluoai_w13_interleave_perm,
    )
    from vllm.model_executor.layers.fused_moe.experts.zentorch_moe import (
        ZentorchExpertsWNA16,
    )

    return swigluoai_w13_interleave_perm(
        experts_cls or ZentorchExpertsWNA16,
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
    """The permutation is a gather on w13's output-channel axis, which is dim 1
    for the N-first packed weights and their group scales."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    group_size = 32
    w13, w2, w13_scale, w2_scale = _make_moe_weights(group_size)
    bias = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, dtype=torch.bfloat16)
    perm = _swigluoai_perm(MoEActivation.SWIGLUOAI)

    baseline = int_wna16._process_weights_zen_cpu(w13, w2, w13_scale, w2_scale)[0]
    permuted = int_wna16._process_weights_zen_cpu(
        w13[:, perm].contiguous(), w2, w13_scale[:, perm].contiguous(), w2_scale
    )[0]

    for expert in range(NUM_EXPERTS):
        before = _unpack_s4(baseline[expert], HIDDEN)
        after = _unpack_s4(permuted[expert], HIDDEN)
        torch.testing.assert_close(after[0::2], before[:INTERMEDIATE])
        torch.testing.assert_close(after[1::2], before[INTERMEDIATE:])

    torch.testing.assert_close(bias[:, perm][:, 0::2], bias[:, :INTERMEDIATE])
    torch.testing.assert_close(bias[:, perm][:, 1::2], bias[:, INTERMEDIATE:])
