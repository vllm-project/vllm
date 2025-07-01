# SPDX-License-Identifier: Apache-2.0
"""
Test DeepEP + DeepGEMM integration 
DeepGEMM are gemm kernels specialized for the
fp8 block-quantized case.
"""

import dataclasses
from typing import Optional

import pytest
import torch.distributed
from torch.distributed import ProcessGroup
from typing_extensions import ParamSpec

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.platforms import current_platform
from vllm.utils import has_deep_ep, has_deep_gemm

from .utils import ProcessGroupInfo, parallel_launch

if has_deep_ep():
    from vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize import (  # noqa: E501
        DeepEPHTPrepareAndFinalize)
    from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import (  # noqa: E501
        DeepEPLLPrepareAndFinalize)

    from .deepep_utils import DeepEPHTArgs, DeepEPLLArgs, make_deepep_a2a

if has_deep_gemm():
    import deep_gemm

    from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
        BatchedDeepGemmExperts)
    from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
        DeepGemmExperts)

requires_deep_ep = pytest.mark.skipif(
    not has_deep_ep(),
    reason="Requires deep_ep kernels",
)

requires_deep_gemm = pytest.mark.skipif(
    not has_deep_gemm(),
    reason="Requires deep_gemm kernels",
)

P = ParamSpec("P")


def next_power_of_2(x):
    import math
    if x == 0:
        return 1
    return 2**math.ceil(math.log2(x))


def per_block_cast_to_fp8(
        x: torch.Tensor,
        block_size_n: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (deep_gemm.ceil_div(m, 128) * 128,
         deep_gemm.ceil_div(n, block_size_n) * block_size_n),
        dtype=x.dtype,
        device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, block_size_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    x_scaled_sub = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    scales = (x_amax / 448.0).view(x_view.size(0), x_view.size(2))
    return x_scaled_sub, scales


def make_block_quant_fp8_weights(
    e: int,
    n: int,
    k: int,
    block_size: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return weights w1, w2, w1q, w2q, w1_scale, w2_scale
    """
    dtype = torch.bfloat16

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    w1_bf16 = torch.randn((e, 2 * n, k), dtype=dtype) / 10
    w1_bf16 = w1_bf16.clamp(min=fp8_min, max=fp8_max).to(dtype=dtype)

    w2_bf16 = torch.randn((e, k, n), dtype=dtype) / 10
    w2_bf16 = w2_bf16.clamp(min=fp8_min, max=fp8_max).to(dtype=dtype)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles_w1 = ((2 * n) + block_n - 1) // block_n
    k_tiles_w1 = (k + block_k - 1) // block_k
    n_tiles_w2 = (k + block_n - 1) // block_n
    k_tiles_w2 = (n + block_k - 1) // block_k

    w1 = torch.empty_like(w1_bf16, dtype=torch.float8_e4m3fn)
    w2 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn)

    w1_s = torch.empty((e, n_tiles_w1, k_tiles_w1),
                       device="cuda",
                       dtype=torch.float32)
    w2_s = torch.empty((e, n_tiles_w2, k_tiles_w2),
                       device="cuda",
                       dtype=torch.float32)

    assert w1_s.shape == (e, (2 * n + 127) // 128, (k + 127) // 128)
    assert (w2.shape[-2] + block_n - 1) // block_n == w2_s.shape[-2]

    for i in range(e):
        w1[i], w1_s[i] = per_block_cast_to_fp8(w1_bf16[i])
        w2[i], w2_s[i] = per_block_cast_to_fp8(w2_bf16[i])

    return w1, w2, w1_s, w2_s


@dataclasses.dataclass
class TestConfig:
    topk: int
    m: int
    k: int
    n: int
    num_experts: int
    block_size: list[int]
    # configs for testing low-latency kernels
    low_latency: bool
    use_fp8_dispatch: Optional[bool] = False


@dataclasses.dataclass
class TestTensors:
    rank_tokens: torch.Tensor  # all ranks make this many tokens
    rank_token_scales: Optional[torch.Tensor]
    topk: torch.Tensor
    topk_weights: torch.Tensor
    config: TestConfig

    @staticmethod
    def make(config: TestConfig, rank) -> "TestTensors":

        dtype = torch.bfloat16
        topk, m, k, block_size = (config.topk, config.m, config.k,
                                  config.block_size)

        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        rank_tokens = torch.randn(
            (m, k), device=torch.cuda.current_device(), dtype=dtype) / 10.0
        rank_tokens = rank_tokens.clamp(min=fp8_min, max=fp8_max)

        block_k = block_size[1]
        _, rank_token_scales = per_token_group_quant_fp8(rank_tokens, block_k)

        topk_ids = torch.randint(
            low=0,
            high=config.num_experts,
            size=(m, topk),
            device=torch.cuda.current_device()).to(dtype=torch.int64)

        topk_weights = torch.randn(topk_ids.shape,
                                   dtype=torch.float32,
                                   device=torch.cuda.current_device())

        return TestTensors(rank_tokens=rank_tokens,
                           rank_token_scales=rank_token_scales,
                           topk=topk_ids,
                           topk_weights=topk_weights,
                           config=config)


def make_ll_modular_kernel(pg: ProcessGroup, pgi: ProcessGroupInfo,
                           max_tokens_per_rank: int, dp_size: int,
                           hidden_size: int, q_dtype: Optional[torch.dtype],
                           test_config: TestConfig) -> FusedMoEModularKernel:

    assert test_config.low_latency
    assert test_config.use_fp8_dispatch is not None

    a2a: DeepEPLLPrepareAndFinalize = make_deepep_a2a(
        pg=pg,
        pgi=pgi,
        dp_size=dp_size,
        deepep_ht_args=None,
        deepep_ll_args=DeepEPLLArgs(
            max_tokens_per_rank=max_tokens_per_rank,
            hidden_size=hidden_size,
            num_experts=test_config.num_experts,
            use_fp8_dispatch=test_config.use_fp8_dispatch),
        q_dtype=q_dtype,
        block_shape=test_config.block_size)

    fused_experts = BatchedDeepGemmExperts(max_num_tokens=max_tokens_per_rank,
                                           world_size=pgi.world_size,
                                           dp_size=dp_size,
                                           block_shape=test_config.block_size)
    mk = FusedMoEModularKernel(prepare_finalize=a2a,
                               fused_experts=fused_experts)
    return mk


def make_ht_modular_kernel(pg: ProcessGroup, pgi: ProcessGroupInfo,
                           dp_size: int, num_local_experts: int,
                           q_dtype: Optional[torch.dtype],
                           test_config: TestConfig) -> FusedMoEModularKernel:

    assert not test_config.low_latency
    assert test_config.use_fp8_dispatch is None

    a2a: DeepEPHTPrepareAndFinalize = make_deepep_a2a(
        pg=pg,
        pgi=pgi,
        dp_size=dp_size,
        deepep_ht_args=DeepEPHTArgs(num_local_experts=num_local_experts),
        deepep_ll_args=None,
        q_dtype=q_dtype,
        block_shape=test_config.block_size)

    fused_experts = DeepGemmExperts()
    mk = FusedMoEModularKernel(prepare_finalize=a2a,
                               fused_experts=fused_experts)
    return mk


def make_modular_kernel(pg: ProcessGroup, pgi: ProcessGroupInfo, dp_size: int,
                        num_local_experts: int,
                        test_tensors: TestTensors) -> FusedMoEModularKernel:

    q_dtype = torch.float8_e4m3fn
    test_config = test_tensors.config

    mk: FusedMoEModularKernel
    # Make modular kernel
    if test_config.low_latency:
        max_tokens_per_rank = max(
            64, next_power_of_2(test_tensors.rank_tokens.size(0)))
        hidden_size = test_tensors.rank_tokens.size(-1)

        mk = make_ll_modular_kernel(pg=pg,
                                    pgi=pgi,
                                    max_tokens_per_rank=max_tokens_per_rank,
                                    dp_size=dp_size,
                                    hidden_size=hidden_size,
                                    q_dtype=q_dtype,
                                    test_config=test_config)
    else:
        mk = make_ht_modular_kernel(pg, pgi, dp_size, num_local_experts,
                                    q_dtype, test_config)

    return mk


def deepep_deepgemm_moe_impl(pg: ProcessGroup, pgi: ProcessGroupInfo,
                             dp_size: int, test_tensors: TestTensors,
                             w1: torch.Tensor, w2: torch.Tensor,
                             w1_scale: Optional[torch.Tensor],
                             w2_scale: Optional[torch.Tensor]) -> torch.Tensor:

    test_config = test_tensors.config
    num_experts = test_config.num_experts
    num_local_experts = w1.size(0)

    def build_expert_map():
        num_local_experts = w1.size(0)
        expert_map = torch.full((num_experts, ),
                                fill_value=-1,
                                dtype=torch.int32)
        s = pgi.rank * num_local_experts
        e = s + num_local_experts
        expert_map[s:e] = torch.tensor(list(range(num_local_experts)))
        return expert_map.to(device=torch.cuda.current_device(),
                             dtype=torch.int32)

    # Make modular kernel
    mk: FusedMoEModularKernel = make_modular_kernel(
        pg=pg,
        pgi=pgi,
        dp_size=dp_size,
        num_local_experts=num_local_experts,
        test_tensors=test_tensors)

    # Low-Latency kernels can't dispatch scales.
    a1_scale = (None
                if test_config.low_latency else test_tensors.rank_token_scales)

    out = mk.forward(hidden_states=test_tensors.rank_tokens,
                     w1=w1,
                     w2=w2,
                     topk_weights=test_tensors.topk_weights,
                     topk_ids=test_tensors.topk,
                     inplace=False,
                     activation="silu",
                     global_num_experts=num_experts,
                     expert_map=build_expert_map(),
                     w1_scale=w1_scale,
                     w2_scale=w2_scale,
                     w1_zp=None,
                     w2_zp=None,
                     a1_scale=a1_scale,
                     a2_scale=None,
                     apply_router_weight_on_input=False)
    return out


def triton_impl(a: torch.Tensor, topk_ids: torch.Tensor,
                topk_weights: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                w1_scale: torch.Tensor, w2_scale: torch.Tensor,
                a1_scale: torch.Tensor, block_shape: list[int]):

    return fused_experts(
        hidden_states=a,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        block_shape=block_shape,
        # Make sure this is set to False so we
        # dont end up comparing the same implementation.
        allow_deep_gemm=False)


def _test_deepep_deepgemm_moe(
    pgi: ProcessGroupInfo,
    dp_size: int,
    config: TestConfig,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
):
    current_platform.seed_everything(pgi.rank)

    w1 = w1.to(device=torch.cuda.current_device())
    w2 = w2.to(device=torch.cuda.current_device())
    w1_scale = w1_scale.to(device=torch.cuda.current_device())
    w2_scale = w2_scale.to(device=torch.cuda.current_device())

    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    test_tensors = TestTensors.make(config, pgi.rank)
    block_shape = [
        w1.size(1) // w1_scale.size(1),
        w1.size(2) // w1_scale.size(2)
    ]

    with set_current_vllm_config(VllmConfig()):
        # Reference
        triton_moe = triton_impl(a=test_tensors.rank_tokens,
                                 topk_ids=test_tensors.topk,
                                 topk_weights=test_tensors.topk_weights,
                                 w1=w1,
                                 w2=w2,
                                 w1_scale=w1_scale,
                                 w2_scale=w2_scale,
                                 a1_scale=test_tensors.rank_token_scales,
                                 block_shape=block_shape)

        # Slice experts for this rank.
        num_local_experts = config.num_experts // pgi.world_size
        e_start = num_local_experts * pgi.rank
        e_end = e_start + num_local_experts
        w1_ep = w1[e_start:e_end]
        w2_ep = w2[e_start:e_end]
        w1_scale_ep = w1_scale[e_start:e_end]
        w2_scale_ep = w2_scale[e_start:e_end]

        deepep_moe = deepep_deepgemm_moe_impl(
            pg,
            pgi,
            dp_size,
            test_tensors,
            w1_ep,
            w2_ep,
            w1_scale_ep,
            w2_scale_ep,
        )

    torch.testing.assert_close(
        triton_moe,
        deepep_moe,
        atol=6e-2,
        rtol=6e-2,
    )


MNKs = [
    (8, 128, 128),
    (8, 128, 512),
    (8, 512, 512),
    (3, 1024, 2048),
    (32, 128, 1024),
    (45, 512, 2048),
    (64, 1024, 1024),
    (129, 128, 256),
    (129, 1024, 2048),
    (222, 1024, 2048),
]

TOPKS = [2, 6]
NUM_EXPERTS = [32]


@pytest.mark.parametrize("mnk", MNKs)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("world_dp_size", [(2, 1)])
@requires_deep_ep
@requires_deep_gemm
def test_ht_deepep_deepgemm_moe(mnk: tuple[int, int, int], num_experts: int,
                                topk: int, world_dp_size: tuple[int, int]):
    """
    Tests for High-Throughput DeepEP + DeepGemm integration.
    """

    m, n, k = mnk
    current_platform.seed_everything(7)

    if topk > num_experts:
        pytest.skip(f"Skipping test: topk={topk} > E={num_experts}")

    block_m = deep_gemm.get_m_alignment_for_contiguous_layout()
    block_size = [block_m, block_m]

    world_size, dp_size = world_dp_size
    config = TestConfig(topk=topk,
                        m=m,
                        k=k,
                        n=n,
                        num_experts=num_experts,
                        block_size=block_size,
                        low_latency=False,
                        use_fp8_dispatch=None)

    w1, w2, w1_scale, w2_scale = make_block_quant_fp8_weights(
        num_experts, n, k, block_size)

    parallel_launch(world_size, _test_deepep_deepgemm_moe, dp_size, config, w1,
                    w2, w1_scale, w2_scale)


MNKs = [
    (1, 128, 2560),
    (2, 128, 2560),
    (3, 1024, 2560),
    (32, 128, 2560),
    (45, 512, 2560),
    (64, 1024, 2560),
    (222, 1024, 2560),
]
# Fix tests for USE_FP8_DISPATCH=True
USE_FP8_DISPATCH = [False]


@pytest.mark.parametrize("mnk", MNKs)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("use_fp8_dispatch", USE_FP8_DISPATCH)
@pytest.mark.parametrize("block_size", [[128, 128]])
@pytest.mark.parametrize("world_dp_size", [(2, 1)])
@requires_deep_ep
@requires_deep_gemm
def test_ll_deepep_deepgemm_moe(mnk: tuple[int, int,
                                           int], num_experts: int, topk: int,
                                use_fp8_dispatch: bool, block_size: list[int],
                                world_dp_size: tuple[int, int]):
    """
    Tests for Low-Latency DeepEP + DeepGemm integration.
    """

    m, n, k = mnk
    current_platform.seed_everything(7)

    if topk > num_experts:
        pytest.skip(f"Skipping test: topk={topk} > E={num_experts}")

    world_size, dp_size = world_dp_size
    config = TestConfig(
        topk=topk,
        m=m,
        k=k,
        n=n,
        num_experts=num_experts,
        block_size=block_size,
        low_latency=True,
        use_fp8_dispatch=use_fp8_dispatch,
    )

    w1, w2, w1_scale, w2_scale = make_block_quant_fp8_weights(
        num_experts, n, k, block_size)

    parallel_launch(world_size, _test_deepep_deepgemm_moe, dp_size, config, w1,
                    w2, w1_scale, w2_scale)
