# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm._custom_ops import (
    cpu_fused_moe,
    cpu_fused_moe_int8,
    cpu_prepack_moe_weight,
    cpu_prepack_moe_weight_int8,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
    biased_moe_quant_config,
    int8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
    ArmCPUUnquantizedExperts,
    CPUEmulationExpertsMxfp4,
    CPUExpertsInt8,
    CPUExpertsMxfp4,
    CPUUnquantizedExperts,
    PowerCPUUnquantizedExperts,
    X86CPUUnquantizedExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    convert_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_quant_config,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.fused_moe.router.cpu_router import _softmax_topk
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

EXPERT_NUM = [
    8,
]
HIDDEN_DIM = [128, 2880]
INTERMEDIATE_DIM = [128, 2880]
BATCH_SIZE = [1, 64, 256]
ACT = [
    MoEActivation.SILU,
    MoEActivation.SWIGLUOAI,
    MoEActivation.GELU,
    MoEActivation.GELU_TANH,
]
USE_BIAS = [False, True]
ISA = ["vec"]
if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
    ISA.append("neon")
if torch.cpu._is_amx_tile_supported():
    ISA.append("amx")
if current_platform.get_cpu_architecture() == CpuArchEnum.POWERPC:
    ISA.append("vsx")

DTYPE = [torch.bfloat16]


def _ref_moe_activation(
    input: torch.Tensor,
    activation: MoEActivation,
) -> torch.Tensor:
    if activation == MoEActivation.SWIGLUOAI:
        gate = input[..., ::2].clamp(max=7.0)
        up = input[..., 1::2].clamp(min=-7.0, max=7.0)
        glu = gate * torch.sigmoid(gate * 1.702)
        return (up + 1) * glu

    d = input.shape[-1] // 2
    gate, up = input[..., :d], input[..., d:]
    if activation == MoEActivation.SILU:
        return torch.nn.functional.silu(gate) * up
    if activation == MoEActivation.GELU:
        return torch.nn.functional.gelu(gate, approximate="none") * up
    if activation == MoEActivation.GELU_TANH:
        return torch.nn.functional.gelu(gate, approximate="tanh") * up
    raise ValueError(f"Unsupported activation: {activation}")


def ref_fused_moe(
    input: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
) -> torch.Tensor:
    len_experts = w13.size(0)

    cnts = topk_ids.new_zeros((topk_ids.shape[0], len_experts))
    cnts.scatter_(1, topk_ids.to(torch.int64), 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()

    sorted_tokens = input[idxs // topk_ids.shape[1]]
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    outputs = []
    start_idx = 0

    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx].float()
        curr_w13 = w13[i].float()
        curr_w2 = w2[i].float()

        curr_w13_bias = None
        if w13_bias is not None:
            curr_w13_bias = w13_bias[i].float()

        curr_w2_bias = None
        if w2_bias is not None:
            curr_w2_bias = w2_bias[i].float()

        gate_up = torch.nn.functional.linear(
            tokens_for_this_expert, curr_w13, curr_w13_bias
        )
        # Note: to simulate the kernel implementation
        gate_up = _ref_moe_activation(gate_up, activation)
        gate_up = gate_up.to(dtype=input.dtype).float()
        expert_out = torch.nn.functional.linear(gate_up, curr_w2, curr_w2_bias)

        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)

    new_x[idxs] = outs
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .mul_(topk_weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(input.dtype)
    )
    return final_out


def quantize_per_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize each weight output channel."""
    weight_fp32 = weight.float()
    scale = weight_fp32.abs().amax(dim=-1).clamp_min(1e-12) / 127.0
    quantized = (
        (weight_fp32 / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    )
    return quantized, scale


def quantize_per_token(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize each input token."""
    input_fp32 = input.float()
    scale = input_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-7) / 127.0
    quantized = (input_fp32 / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def ref_fused_moe_int8(
    input: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
) -> torch.Tensor:
    """Reference the two dynamically quantized INT8 GEMMs."""
    input_int8, input_scale = quantize_per_token(input)
    expert_num = w13.size(0)

    counts = topk_ids.new_zeros((topk_ids.size(0), expert_num))
    counts.scatter_(1, topk_ids.to(torch.int64), 1)
    tokens_per_expert = counts.sum(dim=0).tolist()
    sorted_route_ids = topk_ids.view(-1).argsort()
    sorted_token_ids = sorted_route_ids // topk_ids.size(1)

    outputs = []
    start_idx = 0
    for expert_idx, token_count in enumerate(tokens_per_expert):
        end_idx = start_idx + token_count
        if token_count == 0:
            continue

        token_ids = sorted_token_ids[start_idx:end_idx]
        gate_up = torch.matmul(
            input_int8[token_ids].float(),
            w13[expert_idx].float().T,
        )
        gate_up *= input_scale[token_ids] * w13_scale[expert_idx]
        if w13_bias is not None:
            gate_up += w13_bias[expert_idx].float()

        intermediate = _ref_moe_activation(gate_up, activation).to(input.dtype)
        intermediate_int8, intermediate_scale = quantize_per_token(intermediate)
        output = torch.matmul(
            intermediate_int8.float(),
            w2[expert_idx].float().T,
        )
        output *= intermediate_scale * w2_scale[expert_idx]
        if w2_bias is not None:
            output += w2_bias[expert_idx].float()

        outputs.append(output)
        start_idx = end_idx

    sorted_output = torch.cat(outputs, dim=0)
    routed_output = torch.empty_like(sorted_output)
    routed_output[sorted_route_ids] = sorted_output
    return (
        routed_output.view(*topk_ids.shape, -1)
        .mul_(topk_weights.unsqueeze(-1))
        .sum(dim=1)
        .to(input.dtype)
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("expert_num", EXPERT_NUM)
@pytest.mark.parametrize("hidden_size", HIDDEN_DIM)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_DIM)
@pytest.mark.parametrize("use_bias", USE_BIAS)
@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("act", ACT)
@pytest.mark.parametrize("isa", ISA)
def test_cpu_fused_moe(
    default_vllm_config,
    batch_size: int,
    expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    use_bias: bool,
    dtype: torch.dtype,
    act: MoEActivation,
    isa: str,
):
    set_random_seed(0)

    topk_num = max(expert_num // 2, 1)
    up_dim = 2 * intermediate_size

    input = torch.randn((batch_size, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w13 = torch.randn((expert_num, up_dim, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w2 = torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype) / (
        0.5 * intermediate_size**0.5
    )
    router_logits = torch.randn((batch_size, expert_num), dtype=dtype)
    w13_bias = None
    w2_bias = None
    if use_bias:
        w13_bias = torch.randn((expert_num, up_dim), dtype=dtype) / (0.5 * up_dim**0.5)
        w2_bias = torch.randn((expert_num, hidden_size), dtype=dtype) / (
            0.5 * hidden_size**0.5
        )
    score = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk_num)
    topk_ids = topk_ids.to(torch.int32)

    ref_output = ref_fused_moe(
        input,
        w13,
        w2,
        w13_bias,
        w2_bias,
        topk_weight,
        topk_ids,
        act,
    )

    packed_w13 = cpu_prepack_moe_weight(w13, isa)
    packed_w2 = cpu_prepack_moe_weight(w2, isa)
    output = torch.empty_like(input)
    cpu_fused_moe(
        output,
        input,
        packed_w13,
        packed_w2,
        w13_bias,
        w2_bias,
        topk_weight,
        topk_ids,
        act.value,
        isa,
    )

    atol, rtol = get_default_atol(output), get_default_rtol(output)
    (
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.skipif(
    current_platform.get_cpu_architecture()
    not in (CpuArchEnum.ARM, CpuArchEnum.POWERPC),
    reason="Requires Arm or POWER CPU",
)
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("expert_num", EXPERT_NUM)
@pytest.mark.parametrize("hidden_size", HIDDEN_DIM)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_DIM)
@pytest.mark.parametrize("use_bias", USE_BIAS)
@pytest.mark.parametrize(
    "dtype",
    (
        [torch.float32, torch.float16, torch.bfloat16]
        if current_platform.get_cpu_architecture() == CpuArchEnum.ARM
        else [torch.float32, torch.bfloat16]
    ),
)
@pytest.mark.parametrize("act", ACT)
@pytest.mark.parametrize(
    "isa",
    ["neon"] if current_platform.get_cpu_architecture() == CpuArchEnum.ARM else ["vsx"],
)
def test_cpu_fused_moe_int8(
    batch_size: int,
    expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    use_bias: bool,
    dtype: torch.dtype,
    act: MoEActivation,
    isa: str,
):
    set_random_seed(0)
    topk_num = max(expert_num // 2, 1)
    up_dim = 2 * intermediate_size

    input = torch.randn((batch_size, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w13_fp = torch.randn((expert_num, up_dim, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w2_fp = torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype) / (
        0.5 * intermediate_size**0.5
    )
    w13, w13_scale = quantize_per_channel(w13_fp)
    w2, w2_scale = quantize_per_channel(w2_fp)

    w13_bias = None
    w2_bias = None
    if use_bias:
        w13_bias = torch.randn((expert_num, up_dim), dtype=dtype) / (0.5 * up_dim**0.5)
        w2_bias = torch.randn((expert_num, hidden_size), dtype=dtype) / (
            0.5 * hidden_size**0.5
        )

    router_logits = torch.randn((batch_size, expert_num), dtype=dtype)
    score = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score, topk_num)
    topk_ids = topk_ids.to(torch.int32)

    ref_output = ref_fused_moe_int8(
        input,
        w13,
        w2,
        w13_scale,
        w2_scale,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        act,
    )
    packed_w13 = cpu_prepack_moe_weight_int8(w13, isa)
    packed_w2 = cpu_prepack_moe_weight_int8(w2, isa)
    output = torch.empty_like(input)
    cpu_fused_moe_int8(
        output,
        input,
        packed_w13,
        packed_w2,
        w13_scale,
        w2_scale,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        act.value,
        isa,
    )

    atol = rtol = 1e-1 if isa == "vsx" else 2e-2
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)


# moe_intermediate_size not a multiple of 32, e.g. what tensor-parallel
# sharding of moe_intermediate_size=704 at tp=4 produces (704 // 4 == 176).
UNALIGNED_INTERMEDIATE_DIM = 176


class _StubMoELayer(torch.nn.Module):
    """Minimal stand-in for the real MoE layer module, exposing just what
    the unquantized CPU experts read/replace (w13_weight, w2_weight, the
    router configuration, and optionally w13_bias/w2_bias)."""

    def __init__(
        self,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        activation: MoEActivation,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        self.activation = activation
        if w13_bias is not None:
            self.w13_bias = torch.nn.Parameter(w13_bias, requires_grad=False)
        if w2_bias is not None:
            self.w2_bias = torch.nn.Parameter(w2_bias, requires_grad=False)
        self.use_grouped_topk = False
        self.renormalize = False
        self.scoring_func = "softmax"
        self.custom_routing_function = None


def _make_moe_config(
    expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    topk_num: int,
    dtype: torch.dtype,
    act: MoEActivation,
    has_bias: bool = False,
) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=expert_num,
        experts_per_token=topk_num,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=expert_num,
        num_logical_experts=expert_num,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=act,
        in_dtype=dtype,
        device="cpu",
        routing_method=RoutingMethodType.Default,
        has_bias=has_bias,
    )


@pytest.mark.parametrize(
    ("hidden_size", "intermediate_size", "act", "expected"),
    [
        (128, 128, MoEActivation.SILU, True),
        # Hidden size is not 32-aligned.
        (112, 128, MoEActivation.SILU, False),
        (128, 176, MoEActivation.SILU, True),
        # SwiGLUOAI's interleaved gate/up layout cannot be padded.
        (128, 176, MoEActivation.SWIGLUOAI, False),
    ],
)
def test_cpu_vec_fused_moe_shape_support(
    hidden_size: int,
    intermediate_size: int,
    act: MoEActivation,
    expected: bool,
):
    moe_config = _make_moe_config(
        expert_num=8,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        topk_num=4,
        dtype=torch.bfloat16,
        act=act,
    )
    supported, _ = CPUUnquantizedExperts.is_supported_config(
        CPUUnquantizedExperts,
        moe_config,
        None,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )
    assert supported is expected


@pytest.mark.parametrize("amx_supported", [False, True])
@pytest.mark.parametrize("backend", ["auto", "cpu"])
def test_cpu_mxfp4_backend_selection(monkeypatch, amx_supported, backend):
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4._user_moe_activation_override",
        lambda: None,
    )
    monkeypatch.setattr(torch.cpu, "_is_amx_tile_supported", lambda: amx_supported)
    monkeypatch.setattr(
        current_platform, "get_cpu_architecture", lambda: CpuArchEnum.X86
    )
    config = _make_moe_config(4, 64, 64, 2, torch.bfloat16, MoEActivation.SWIGLUOAI)
    config.moe_backend = backend
    selected, experts = select_mxfp4_moe_backend(config)
    assert selected == (
        Mxfp4MoeBackend.CPU if amx_supported else Mxfp4MoeBackend.CPU_EMULATION
    )
    assert experts is (CPUExpertsMxfp4 if amx_supported else CPUEmulationExpertsMxfp4)


@pytest.mark.parametrize("act", [MoEActivation.SILU, MoEActivation.SWIGLUOAI])
@pytest.mark.parametrize("use_bias", [False, True])
def test_cpu_mxfp4_emulation(default_vllm_config, act, use_bias):
    from tests.quantization.reference_mxfp4 import dq_mxfp4_torch

    set_random_seed(0)
    experts, hidden, intermediate, topk = 4, 64, 96, 2
    w13 = torch.randint(
        0, 256, (experts, 2 * intermediate, hidden // 2), dtype=torch.uint8
    )
    w2 = torch.randint(0, 256, (experts, hidden, intermediate // 2), dtype=torch.uint8)
    s13 = torch.randint(
        120, 125, (experts, 2 * intermediate, hidden // 32), dtype=torch.uint8
    )
    s2 = torch.randint(
        120, 125, (experts, hidden, intermediate // 32), dtype=torch.uint8
    )
    # Decode through the independent bit-level reference, not the fallback helper.
    ref_w13 = dq_mxfp4_torch(w13, s13, torch.bfloat16)
    ref_w2 = dq_mxfp4_torch(w2, s2, torch.bfloat16)
    b13 = torch.randn(experts, 2 * intermediate) if use_bias else None
    b2 = torch.randn(experts, hidden) if use_bias else None
    layer = _StubMoELayer(w13, w2, act, b13, b2)
    convert = (
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format
        if act == MoEActivation.SWIGLUOAI
        else convert_weight_to_mxfp4_moe_kernel_format
    )
    w13, w2, s13, s2, converted_b13, converted_b2 = convert(
        Mxfp4MoeBackend.CPU_EMULATION, layer, w13, w2, s13, s2, b13, b2
    )
    layer = _StubMoELayer(w13, w2, act, converted_b13, converted_b2)
    layer.w13_weight_scale = torch.nn.Parameter(s13, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(s2, requires_grad=False)
    config = _make_moe_config(
        experts, hidden, intermediate, topk, torch.bfloat16, act, use_bias
    )
    quant_config = make_mxfp4_moe_quant_config(
        Mxfp4MoeBackend.CPU_EMULATION,
        s13,
        s2,
        w1_bias=getattr(layer, "w13_bias", None),
        w2_bias=getattr(layer, "w2_bias", None),
    )
    impl = CPUEmulationExpertsMxfp4(config, quant_config)
    impl.process_weights_after_loading(layer)
    assert layer.w13_weight.dtype == layer.w2_weight.dtype == torch.bfloat16
    x = torch.randn(7, hidden, dtype=torch.bfloat16)
    router = torch.randn(7, experts)
    weights, ids = _softmax_topk(router, topk, False)
    expected = ref_fused_moe(
        x,
        ref_w13,
        ref_w2,
        b13.to(torch.bfloat16) if b13 is not None else None,
        b2.to(torch.bfloat16) if b2 is not None else None,
        weights,
        ids,
        act,
    )
    output = torch.empty_like(x)
    impl.apply(
        output=output,
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=weights,
        topk_ids=ids,
        activation=act,
        global_num_experts=experts,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=torch.empty(0),
        workspace2=torch.empty(0),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e8m0fnu])
@pytest.mark.parametrize("exponent", [0, 120, 127, 133, 255])
def test_cpu_mxfp4_decode(scale_dtype, exponent):
    from tests.quantization.reference_mxfp4 import dq_mxfp4_torch
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        dequant_mxfp4_experts,
    )

    # Every pair of E2M1 values, including negative zero, with E8M0 edge cases.
    packed = torch.arange(256).to(torch.uint8).reshape(2, 2, 64)
    scales = torch.full((2, 2, 4), exponent, dtype=torch.uint8)
    expected = dq_mxfp4_torch(packed, scales, torch.bfloat16)
    actual = dequant_mxfp4_experts(packed, scales.view(scale_dtype))
    if exponent == 255:
        # E8M0 reserves 0xff for NaN; the reference's exponent-only
        # conversion models infinity instead of this reserved encoding.
        assert actual.isnan().all()
        return
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("expert_num", EXPERT_NUM)
@pytest.mark.parametrize("hidden_size", HIDDEN_DIM)
@pytest.mark.parametrize("use_bias", USE_BIAS)
@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("isa", ISA)
@pytest.mark.parametrize(
    "act",
    [MoEActivation.SILU, MoEActivation.GELU, MoEActivation.GELU_TANH],
)
def test_cpu_fused_moe_unaligned_intermediate_size(
    default_vllm_config,
    expert_num: int,
    hidden_size: int,
    use_bias: bool,
    dtype: torch.dtype,
    isa: str,
    act: MoEActivation,
):
    """CPU kernels handle unaligned intermediate sizes by zero-padding the
    weights before prepacking."""
    set_random_seed(0)
    batch_size = 64
    intermediate_size = UNALIGNED_INTERMEDIATE_DIM
    topk_num = max(expert_num // 2, 1)
    up_dim = 2 * intermediate_size

    input = torch.randn((batch_size, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w13 = torch.randn((expert_num, up_dim, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w2 = torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype) / (
        0.5 * intermediate_size**0.5
    )
    router_logits = torch.randn((batch_size, expert_num), dtype=dtype)
    w13_bias = None
    w2_bias = None
    if use_bias:
        w13_bias = torch.randn((expert_num, up_dim), dtype=dtype) / (0.5 * up_dim**0.5)
        w2_bias = torch.randn((expert_num, hidden_size), dtype=dtype) / (
            0.5 * hidden_size**0.5
        )
    # Route with the same helper the router uses internally, so the
    # reference only differs from the kernel in how the experts are
    # evaluated.
    topk_weight, topk_ids = _softmax_topk(router_logits, topk_num, False)

    ref_output = ref_fused_moe(
        input, w13, w2, w13_bias, w2_bias, topk_weight, topk_ids, act
    )

    layer = _StubMoELayer(
        w13.clone(),
        w2.clone(),
        act,
        w13_bias.clone() if w13_bias is not None else None,
        w2_bias.clone() if w2_bias is not None else None,
    )

    moe_config = _make_moe_config(
        expert_num, hidden_size, intermediate_size, topk_num, dtype, act, use_bias
    )
    experts_cls = {
        "vec": CPUUnquantizedExperts,
        "amx": X86CPUUnquantizedExperts,
        "neon": ArmCPUUnquantizedExperts,
        "vsx": PowerCPUUnquantizedExperts,
    }[isa]
    supported, reason = experts_cls.is_supported_config(
        experts_cls,
        moe_config,
        None,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )
    assert supported, (
        f"expected the padded intermediate size to be supported, got {reason}"
    )

    # Mirror UnquantizedFusedMoEMethod._setup_kernel: the quant config is
    # built from the layer first, then the experts shuffle the layer into the
    # kernel's runtime format.
    quant_config = (
        biased_moe_quant_config(layer.w13_bias, layer.w2_bias)
        if use_bias
        else FusedMoEQuantConfig.make()
    )
    experts = experts_cls(moe_config, quant_config)
    assert experts.isa == isa
    experts.process_weights_after_loading(layer)

    output = torch.empty_like(input)
    experts.apply(
        output=output,
        hidden_states=input,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weight,
        topk_ids=topk_ids,
        activation=act,
        global_num_experts=expert_num,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=torch.empty(0),
        workspace2=torch.empty(0),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    atol, rtol = get_default_atol(output), get_default_rtol(output)
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)


@pytest.mark.skipif(
    current_platform.get_cpu_architecture() != CpuArchEnum.X86
    or not torch.cpu._is_amx_tile_supported(),
    reason="CPUExpertsInt8 requires x86 AMX",
)
@pytest.mark.parametrize("act", [MoEActivation.SILU, MoEActivation.GELU_TANH])
def test_cpu_experts_int8_unaligned_intermediate_size(
    default_vllm_config, act: MoEActivation
):
    """INT8 experts zero-pad the intermediate dim (weights and w13 scale)
    before prepacking."""
    set_random_seed(0)
    dtype = torch.bfloat16
    expert_num, hidden_size, batch_size = 8, 128, 64
    intermediate_size = UNALIGNED_INTERMEDIATE_DIM
    topk_num = expert_num // 2

    input = torch.randn((batch_size, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w13, w13_scale = quantize_per_channel(
        torch.randn((expert_num, 2 * intermediate_size, hidden_size), dtype=dtype)
        / (0.5 * hidden_size**0.5)
    )
    w2, w2_scale = quantize_per_channel(
        torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype)
        / (0.5 * intermediate_size**0.5)
    )
    router_logits = torch.randn((batch_size, expert_num), dtype=dtype)
    topk_weight, topk_ids = _softmax_topk(router_logits, topk_num, False)

    ref_output = ref_fused_moe_int8(
        input, w13, w2, w13_scale, w2_scale, None, None, topk_weight, topk_ids, act
    )

    # The compressed-tensors method stores the per-channel scales as [E, N, 1].
    layer = _StubMoELayer(w13.clone(), w2.clone(), act)
    layer.w13_weight_scale = torch.nn.Parameter(
        w13_scale.unsqueeze(-1).clone(), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        w2_scale.unsqueeze(-1).clone(), requires_grad=False
    )

    moe_config = _make_moe_config(
        expert_num, hidden_size, intermediate_size, topk_num, dtype, act
    )
    supported, reason = CPUExpertsInt8.is_supported_config(
        CPUExpertsInt8,
        moe_config,
        kInt8StaticChannelSym,
        kInt8DynamicTokenSym,
        mk.FusedMoEActivationFormat.Standard,
    )
    assert supported, (
        f"expected the padded intermediate size to be supported, got {reason}"
    )

    # Mirror the quantization method: the quant config holds the scale
    # Parameters before the experts pad and prepack the layer.
    quant_config = int8_w8a8_moe_quant_config(
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        a1_scale=None,
        a2_scale=None,
        per_act_token_quant=True,
    )
    experts = CPUExpertsInt8(moe_config, quant_config)
    experts.process_weights_after_loading(layer)

    output = torch.empty_like(input)
    experts.apply(
        output=output,
        hidden_states=input,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weight,
        topk_ids=topk_ids,
        activation=act,
        global_num_experts=expert_num,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=torch.empty(0),
        workspace2=torch.empty(0),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=2e-2)
