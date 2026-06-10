# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm._custom_ops import cpu_fused_moe, cpu_prepack_moe_weight
from vllm.model_executor.layers.fused_moe import cpu_fused_moe as cpu_fused_moe_mod
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.cpu_fused_moe import (
    _CPU_MOE_ACT_FN,
    CPUFusedMOE,
)
from vllm.platforms import current_platform
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
USE_BIAS = [True, False]
ISA = ["amx", "vec"] if torch.cpu._is_amx_tile_supported() else ["vec"]
DTYPE = [torch.bfloat16]


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
        gate_up = _CPU_MOE_ACT_FN[activation](gate_up).to(dtype=input.dtype).float()
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
    output = cpu_fused_moe(
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


ZENTORCH_ACT = [
    MoEActivation.SILU,
    MoEActivation.GELU,
    MoEActivation.GELU_TANH,
    MoEActivation.SWIGLUOAI,
]


@pytest.fixture
def _mock_zentorch_fused_moe():
    """Register a mock ``zentorch_fused_moe`` op when zentorch is unavailable.

    Lets the dispatch test exercise ``forward_zentorch`` in CI without a real
    zentorch build. Skips registration when the op is already present. The mock
    delegates to ``ref_fused_moe`` so the written output is meaningful and
    records the activation string it was invoked with.
    """
    calls: list[dict] = []

    def _impl(
        output,
        input,
        w13_weight,
        w2_weight,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        skip_weighted,
        activation,
    ):
        calls.append({"activation": activation, "skip_weighted": skip_weighted})
        result = ref_fused_moe(
            input,
            w13_weight,
            w2_weight,
            w13_bias,
            w2_bias,
            topk_weights,
            topk_ids,
            MoEActivation.from_str(activation),
        )
        output.copy_(result)

    if hasattr(torch.ops.zentorch, "zentorch_fused_moe"):
        yield calls
        return

    lib_def = torch.library.Library("zentorch", "DEF")
    lib_def.define(
        "zentorch_fused_moe("
        "Tensor(a!) output, "
        "Tensor input, "
        "Tensor w13_weight, "
        "Tensor w2_weight, "
        "Tensor? w13_bias, "
        "Tensor? w2_bias, "
        "Tensor topk_weights, "
        "Tensor topk_ids, "
        "bool skip_weighted, "
        "str activation"
        ") -> ()"
    )
    lib_impl = torch.library.Library("zentorch", "IMPL", "CPU")
    lib_impl.impl("zentorch_fused_moe", _impl)

    yield calls

    lib_impl._destroy()
    lib_def._destroy()


@pytest.mark.parametrize("act", ZENTORCH_ACT)
def test_cpu_fused_moe_dispatches_to_zentorch(monkeypatch, act: MoEActivation):
    """When zentorch MoE is supported, ``CPUFusedMOE`` selects the zentorch
    forward and records the lowercased activation string."""
    monkeypatch.setattr(
        cpu_fused_moe_mod, "is_zentorch_moe_supported", lambda layer: True
    )

    layer = SimpleNamespace(activation=act)
    moe = CPUFusedMOE(layer)

    assert moe.isa == "none"
    assert moe.act == act.value.lower()
    assert moe.forward_method.__func__ is CPUFusedMOE.forward_zentorch


@pytest.mark.parametrize("use_bias", USE_BIAS)
def test_cpu_fused_moe_zentorch_forward(
    default_vllm_config,
    monkeypatch,
    _mock_zentorch_fused_moe,
    use_bias: bool,
):
    """The zentorch forward passes weights/activation through to the zentorch
    op and returns its (mutated) output."""
    monkeypatch.setattr(
        cpu_fused_moe_mod, "is_zentorch_moe_supported", lambda layer: True
    )
    set_random_seed(0)

    act = MoEActivation.SILU
    expert_num = 8
    hidden_size = 128
    intermediate_size = 128
    batch_size = 64
    dtype = torch.bfloat16
    topk_num = expert_num // 2
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

    layer = SimpleNamespace(activation=act, w13_weight=w13, w2_weight=w2)
    if use_bias:
        layer.w13_bias = torch.randn((expert_num, up_dim), dtype=dtype) / (
            0.5 * up_dim**0.5
        )
        layer.w2_bias = torch.randn((expert_num, hidden_size), dtype=dtype) / (
            0.5 * hidden_size**0.5
        )

    moe = CPUFusedMOE(layer)

    score = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk_num)
    topk_ids = topk_ids.to(torch.int32)

    output = moe.forward_method(
        layer,
        input,
        topk_weight,
        topk_ids,
        act,
    )

    ref_output = ref_fused_moe(
        input,
        w13,
        w2,
        getattr(layer, "w13_bias", None),
        getattr(layer, "w2_bias", None),
        topk_weight,
        topk_ids,
        act,
    )

    assert _mock_zentorch_fused_moe[-1]["activation"] == act.value.lower()
    assert _mock_zentorch_fused_moe[-1]["skip_weighted"] is False
    torch.testing.assert_close(output, ref_output)
