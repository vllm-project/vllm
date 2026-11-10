# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the FlashInfer CuTe-DSL unquantized (bf16/fp16) MoE on SM90."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_map,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_sm90_moe import (  # noqa: E501
    FlashInferCuteDSLSm90Experts,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    convert_to_unquantized_kernel_format,
    make_unquantized_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutedsl_moe_bf16
from vllm.utils.torch_utils import set_random_seed

if not has_flashinfer_cutedsl_moe_bf16() or not (
    current_platform.is_device_capability_family(90)
):
    pytest.skip(
        "Requires FlashInfer CuTe-DSL unquantized MoE on SM90",
        allow_module_level=True,
    )

pytestmark = pytest.mark.usefixtures("vllm_config", "workspace_init")

# The GEMM1 output is O(0.5) at these tensor scales, so this limit clamps.
_SWIGLU_LIMIT = 0.3

# (activation, gemm1_alpha, gemm1_beta, gemm1_clamp_limit)
_ACT_CASES = [
    pytest.param(MoEActivation.SILU, None, None, None, id="silu"),
    pytest.param(MoEActivation.SILU, None, None, _SWIGLU_LIMIT, id="silu-clamp"),
    pytest.param(MoEActivation.GELU_TANH, None, None, None, id="gelu_tanh"),
    pytest.param(MoEActivation.RELU2_NO_MUL, None, None, None, id="relu2_no_mul"),
    pytest.param(MoEActivation.SITU, None, None, None, id="situ"),
    # gpt-oss: interleaved gate/up, constants fixed by `swigluoai_and_mul`.
    pytest.param(MoEActivation.SWIGLUOAI, None, None, None, id="swigluoai"),
    pytest.param(
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        1.702,
        1.0,
        _SWIGLU_LIMIT,
        id="swigluoai_uninterleave",
    ),
]


@pytest.fixture
def vllm_config():
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        yield


def _make_moe_config(
    e: int,
    topk: int,
    k: int,
    n: int,
    activation: MoEActivation = MoEActivation.SILU,
    dtype: torch.dtype = torch.bfloat16,
    ep_size: int = 1,
    ep_rank: int = 0,
    num_local_experts: int | None = None,
    n_unpadded: int | None = None,
) -> FusedMoEConfig:
    parallel = FusedMoEParallelConfig.make_no_parallel()
    parallel.use_ep = ep_size > 1
    parallel.ep_size = ep_size
    parallel.ep_rank = ep_rank
    is_situ = activation == MoEActivation.SITU
    return FusedMoEConfig(
        num_experts=e,
        experts_per_token=topk,
        hidden_dim=k,
        intermediate_size=n,
        num_local_experts=e if num_local_experts is None else num_local_experts,
        num_logical_experts=e,
        activation=activation,
        device="cuda",
        moe_parallel_config=parallel,
        in_dtype=dtype,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=64,
        intermediate_size_per_partition_unpadded=n_unpadded,
        activation_situ_beta=4.0 if is_situ else None,
        activation_situ_linear_beta=25.0 if is_situ else None,
    )


def _random_moe_inputs(m, n, k, e, topk, activation, dtype=torch.bfloat16):
    """`(hidden_states, w13, w2, topk_weights, topk_ids)`, weights in the
    checkpoint layout."""
    hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    # Keep the GEMM1 output variance stable across hidden sizes.
    hidden_states *= (512 / k) ** 0.5
    w13_rows = 2 * n if activation.is_gated else n
    w13 = torch.randn((e, w13_rows, k), device="cuda", dtype=dtype) / 15
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 15
    score = torch.randn((m, e), device="cuda", dtype=dtype)
    topk_weights, topk_ids, _ = fused_topk(hidden_states, score, topk, renormalize=True)
    return hidden_states, w13, w2, topk_weights, topk_ids


def _reference(
    moe_config: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig,
    a: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts: range | None = None,
) -> torch.Tensor:
    """fp32 MoE with vLLM's `apply_moe_activation` and the kernel's bf16
    hand-off of the activated intermediate; experts outside `experts`
    contribute nothing."""
    activation_config = ApplyMoEActivationConfig.from_configs(moe_config, quant_config)
    m, topk = topk_ids.shape
    x = a.view(m, 1, -1).repeat(1, topk, 1).reshape(m * topk, -1)
    flat_ids = topk_ids.reshape(-1)
    out = torch.zeros(m * topk, w2.shape[1], dtype=torch.float32, device=a.device)
    for expert in experts or range(w13.shape[0]):
        mask = flat_ids == expert
        if mask.any():
            gemm1 = x[mask].float() @ w13[expert].float().T
            h = gemm1.new_empty(gemm1.shape[0], w2.shape[2])
            apply_moe_activation(
                moe_config.activation, h, gemm1, activation_config=activation_config
            )
            out[mask] = h.to(a.dtype).float() @ w2[expert].float().T
    return (out.view(m, topk, -1) * topk_weights.view(m, topk, 1).float()).sum(dim=1)


def _run_kernel(
    moe_config: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig,
    a: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Convert checkpoint-layout weights and run the oracle-built kernel."""
    backend = UnquantizedMoeBackend.FLASHINFER_CUTEDSL
    w13, w2 = convert_to_unquantized_kernel_format(
        backend, moe_config, w13.clone(), w2.clone()
    )
    kernel = make_unquantized_moe_kernel(
        quant_config, moe_config, backend, FlashInferCuteDSLSm90Experts
    )
    output = kernel.apply(
        hidden_states=a,
        w1=w13,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=moe_config.activation,
        global_num_experts=moe_config.num_experts,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    assert isinstance(output, torch.Tensor)
    return output.float()


@pytest.mark.parametrize(
    "m,n,k,e,topk",
    [
        (16, 128, 512, 8, 2),
        (3, 96, 2048, 16, 8),  # Qwen3-30B-A3B TP8 shard, decode
        (64, 96, 2048, 16, 8),
        (16, 1792, 4096, 8, 2),  # Mixtral-8x7B TP8 shard
    ],
)
@pytest.mark.parametrize("activation,alpha,beta,limit", _ACT_CASES)
@torch.inference_mode()
def test_flashinfer_cutedsl_sm90_moe(m, n, k, e, topk, activation, alpha, beta, limit):
    set_random_seed(7)
    inputs = _random_moe_inputs(m, n, k, e, topk, activation)
    moe_config = _make_moe_config(e, topk, k, n, activation)
    quant_config = FusedMoEQuantConfig.make(
        gemm1_alpha=alpha, gemm1_beta=beta, gemm1_clamp_limit=limit
    )
    torch.testing.assert_close(
        _run_kernel(moe_config, quant_config, *inputs),
        _reference(moe_config, quant_config, *inputs),
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_flashinfer_cutedsl_sm90_moe_expert_parallel(dtype: torch.dtype):
    """Each EP rank (linear placement, uneven split) computes its expert shard
    from the global top-k ids; the shard outputs sum to the full MoE."""
    set_random_seed(11)
    m, n, k, e, topk, ep_size = 32, 128, 1024, 7, 2, 2
    a, w13, w2, topk_weights, topk_ids = _random_moe_inputs(
        m, n, k, e, topk, MoEActivation.SILU, dtype
    )
    quant_config = FusedMoEQuantConfig.make()
    total = torch.zeros(m, k, dtype=torch.float32, device="cuda")
    for ep_rank in range(ep_size):
        num_local, expert_map, _ = determine_expert_map(ep_size, ep_rank, e)
        assert expert_map is not None
        first = int(torch.nonzero(expert_map >= 0)[0])
        local = range(first, first + num_local)
        moe_config = _make_moe_config(
            e,
            topk,
            k,
            n,
            dtype=dtype,
            ep_size=ep_size,
            ep_rank=ep_rank,
            num_local_experts=num_local,
        )
        shard = _run_kernel(
            moe_config,
            quant_config,
            a,
            w13[first : local.stop],
            w2[first : local.stop],
            topk_weights,
            topk_ids,
            expert_map=expert_map.to("cuda"),
        )
        torch.testing.assert_close(
            shard,
            _reference(
                moe_config, quant_config, a, w13, w2, topk_weights, topk_ids, local
            ),
            atol=2e-2,
            rtol=2e-2,
        )
        total += shard
    full_config = _make_moe_config(e, topk, k, n, dtype=dtype)
    reference = _reference(
        full_config, quant_config, a, w13, w2, topk_weights, topk_ids
    )
    torch.testing.assert_close(total, reference, atol=3e-2, rtol=2e-2)


@torch.inference_mode()
def test_flashinfer_cutedsl_sm90_moe_padded_intermediate():
    """An intermediate size that is not a multiple of 32 is padded; the
    padding holds garbage until the weight conversion zeroes it."""
    set_random_seed(3)
    m, n_unpadded, n, k, e, topk = 24, 88, 96, 512, 4, 2
    a, w13, w2, topk_weights, topk_ids = _random_moe_inputs(
        m, n_unpadded, k, e, topk, MoEActivation.SILU
    )
    w13_padded = w13.new_full((e, 2 * n, k), float("nan"))
    w13_padded[:, :n_unpadded] = w13[:, :n_unpadded]
    w13_padded[:, n : n + n_unpadded] = w13[:, n_unpadded:]
    w2_padded = w2.new_full((e, k, n), float("nan"))
    w2_padded[:, :, :n_unpadded] = w2
    moe_config = _make_moe_config(e, topk, k, n, n_unpadded=n_unpadded)
    quant_config = FusedMoEQuantConfig.make()
    torch.testing.assert_close(
        _run_kernel(
            moe_config, quant_config, a, w13_padded, w2_padded, topk_weights, topk_ids
        ),
        _reference(moe_config, quant_config, a, w13, w2, topk_weights, topk_ids),
        atol=2e-2,
        rtol=2e-2,
    )


@torch.inference_mode()
def test_flashinfer_cutedsl_sm90_moe_router_weight_on_input():
    """With the router weights applied to the input (top-1, e.g. Llama 4),
    the kernel combines with unit scales."""
    set_random_seed(5)
    m, n, k, e, topk = 16, 128, 512, 8, 1
    a, w13, w2, topk_weights, topk_ids = _random_moe_inputs(
        m, n, k, e, topk, MoEActivation.SILU
    )
    moe_config = _make_moe_config(e, topk, k, n)
    quant_config = FusedMoEQuantConfig.make()
    output = _run_kernel(
        moe_config,
        quant_config,
        a,
        w13,
        w2,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=True,
    )
    weighted_a = (a * topk_weights).to(a.dtype)
    ones = torch.ones_like(topk_weights)
    reference = _reference(
        moe_config, quant_config, weighted_a, w13, w2, ones, topk_ids
    )
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("strategy", ["linear", "round_robin"])
def test_flashinfer_cutedsl_sm90_requires_linear_expert_placement(strategy: str):
    """The kernel addresses a contiguous expert shard, so another placement
    is refused when the weights are processed."""
    e, ep_size, ep_rank = 8, 3, 1
    num_local, expert_map, _ = determine_expert_map(
        ep_size, ep_rank, e, expert_placement_strategy=strategy
    )
    moe_config = _make_moe_config(
        e, 2, 128, 128, ep_size=ep_size, ep_rank=ep_rank, num_local_experts=num_local
    )
    experts = FlashInferCuteDSLSm90Experts(moe_config, FusedMoEQuantConfig.make())
    layer = SimpleNamespace(expert_map=expert_map)
    if strategy == "linear":
        experts.process_weights_after_loading(layer)
    else:
        with pytest.raises(ValueError, match="linear expert placement"):
            experts.process_weights_after_loading(layer)
