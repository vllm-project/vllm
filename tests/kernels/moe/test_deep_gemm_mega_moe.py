# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit-test DeepGEMM mega_moe kernel via the DeepGemmMegaExperts modular kernel.
Compare mega_moe output against BF16 fused_experts reference.
"""

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import (
    per_token_cast_to_fp4,
)
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    fused_experts,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_mega_moe import (
    DeepGemmMegaExperts,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    make_moe_prepare_and_finalize_no_dp_ep,
)
from vllm.utils.deep_gemm import (
    calc_diff,
    is_deep_gemm_mega_moe_supported,
)
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

from .modular_kernel_tools.parallel_utils import (
    ProcessGroupInfo,
    parallel_launch_with_config,
)

BLOCK_SIZE = [128, 128]


def cast_grouped_weights_to_fp4(
    bf16_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    import deep_gemm

    num_groups, n, k = bf16_weights.shape
    w = torch.empty((num_groups, n, k // 2), device="cuda", dtype=torch.int8)
    w_sf = torch.empty((num_groups, n, k // 32), device="cuda", dtype=torch.float)
    for i in range(num_groups):
        w[i], w_sf[i] = per_token_cast_to_fp4(
            bf16_weights[i], use_ue8m0=True, gran_k=32
        )
    w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
    return w, w_sf


def chunk_by_rank(
    t: torch.Tensor,
    r: int,
    w: int,
    dim: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    chunk = cdiv(t.shape[dim], w)
    t = t.narrow(dim, r * chunk, chunk)
    if device is not None:
        t = t.to(device)
    return t


# activation fp8 x weights fp4
def run_single_case(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    m: int,
    n: int,
    k: int,
    topk: int,
    num_experts: int,
    block_size: list[int],
):
    """
    Run one (M,N,K) configuration on a single GPU and assert DeepGEMM
    mega_moe (via DeepGemmMegaExperts modular kernel) produces correct
    results compared to BF16 reference.
    """
    import deep_gemm

    device = torch.device(f"cuda:{pgi.local_rank}")
    init_workspace_manager(device)

    dp_rank = pgi.rank
    dp_size = get_dp_group().world_size
    num_local_experts = num_experts // dp_size

    # Use fixed seed so all ranks generate identical "global" data
    set_random_seed(7)

    # Generate global data (same on all ranks due to shared seed)
    tokens_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / 10
    w1_bf16 = (
        torch.randn((num_experts, 2 * n, k), device="cuda", dtype=torch.bfloat16) / 15
    )
    w2_bf16 = torch.randn((num_experts, k, n), device="cuda", dtype=torch.bfloat16) / 15

    # Global routing
    scores = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(
        scores, topk, dim=-1, largest=True, sorted=False
    )

    # Shard tokens by DP rank
    tokens_bf16 = chunk_by_rank(tokens_bf16, dp_rank, dp_size)
    topk_weights = chunk_by_rank(topk_weights, dp_rank, dp_size)
    topk_ids = chunk_by_rank(topk_ids, dp_rank, dp_size)
    local_m = tokens_bf16.shape[0]

    # BF16 unquantized reference (uses all experts, global topk_ids)
    out_ref = fused_experts(
        hidden_states=tokens_bf16,
        w1=w1_bf16,
        w2=w2_bf16,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
    )

    # Shard weights by DP rank for mega_moe
    # C++ assertion: buffer.num_experts == weight_experts_per_rank * num_ranks
    w1_local = chunk_by_rank(w1_bf16, dp_rank, dp_size)
    w2_local = chunk_by_rank(w2_bf16, dp_rank, dp_size)

    # Cast to FP4 and transform for mega_moe layout
    w1_weights = cast_grouped_weights_to_fp4(w1_local)
    w2_weights = cast_grouped_weights_to_fp4(w2_local)
    (dg_w1, dg_w1_s), (dg_w2, dg_w2_s) = deep_gemm.transform_weights_for_mega_moe(
        w1_weights, w2_weights
    )

    # Build FusedMoEQuantConfig with FP4 weight scales
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=None,  # torch.float8_e4m3fn,
        per_act_token_quant=False,
        block_shape=block_size,
        w1_scale=dg_w1_s,
        w2_scale=dg_w2_s,
        weight_dtype="mxfp4",
    )

    moe_parallel_config = FusedMoEParallelConfig.make(
        tp_size_=get_tensor_model_parallel_world_size(),
        pcp_size_=1,
        dp_size_=dp_size,
        sp_size_=1,
        vllm_parallel_config=vllm_config.parallel_config,
    )

    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=k,
        intermediate_size_per_partition=n,
        num_local_experts=num_local_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=moe_parallel_config,
        in_dtype=torch.bfloat16,
        max_num_tokens=next_power_of_2(local_m),
        activation=MoEActivation.SILU,
        device=vllm_config.device_config.device,
        routing_method=RoutingMethodType.DeepSeekV3,
    )

    # Build modular kernel with DeepGemmMegaExperts
    deep_gemm_kernel = mk.FusedMoEKernel(
        prepare_finalize=make_moe_prepare_and_finalize_no_dp_ep(False),
        fused_experts=DeepGemmMegaExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        ),
        inplace=False,
    )

    # Run through the modular kernel
    out_deepgemm = deep_gemm_kernel.apply(
        hidden_states=tokens_bf16,
        w1=dg_w1,
        w2=dg_w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        apply_router_weight_on_input=False,
        expert_map=None,
    )

    # Compare mega_moe against BF16 reference using cosine similarity
    diff = calc_diff(out_ref, out_deepgemm)
    print(
        f"RANK={dp_rank} calc_diff={diff:.6f} "
        f"ref_std={out_ref.float().std():.4f} "
        f"dg_std={out_deepgemm.float().std():.4f}"
    )
    # FP4 weights + FP8 activations introduce quantization error with random
    # weights. Threshold accounts for this.
    assert diff < 0.1, f"calc_diff={diff} too large (threshold 0.1)"


MNKs = [
    (512, 1024, 1024),
    (4096, 4096, 1024),
    (512, 2048, 2048),
]

TOPKS = [2, 6]
NUM_EXPERTS = [32]


@pytest.mark.parametrize(("m", "n", "k"), MNKs)
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("dp_size", [2])
@pytest.mark.skipif(
    not is_deep_gemm_mega_moe_supported(), reason="Requires deep_gemm kernels"
)
def test_deep_gemm_mega_moe(
    m: int,
    n: int,
    k: int,
    topk: int,
    num_experts: int,
    dp_size: int,
    workspace_init,
):
    set_random_seed(7)

    world_size = dp_size

    if topk > num_experts:
        pytest.skip(f"topk={topk} > num_experts={num_experts}")

    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=dp_size,
        tensor_parallel_size=1,
    )

    vllm_config = VllmConfig(parallel_config=parallel_config)

    parallel_launch_with_config(
        world_size,
        run_single_case,
        vllm_config,
        None,  # env
        m,
        n,
        k,
        topk,
        num_experts,
        BLOCK_SIZE,
    )
