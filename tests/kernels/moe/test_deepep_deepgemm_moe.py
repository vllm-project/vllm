# SPDX-License-Identifier: Apache-2.0
"""
Test DeepEP + DeepGEMM integration 
"""

import dataclasses
import traceback
from typing import Callable, Optional

import pytest
import torch.distributed
from torch.distributed import ProcessGroup
from torch.multiprocessing import (
    spawn)  # pyright: ignore[reportPrivateImportUsage]
from typing_extensions import Concatenate, ParamSpec

from tests.kernels.quant_utils import native_w8a8_block_matmul
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import DeepGemmExperts
from vllm.model_executor.layers.fused_moe.deepep_prepare_finalize import (
    DeepEPPrepareAndFinalize)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.platforms import current_platform

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

try:
    import deep_ep
    has_deep_ep = True
except ImportError:
    has_deep_ep = False

try:
    import deep_gemm
    has_deep_gemm = True
except ImportError:
    has_deep_gemm = False

requires_deep_ep = pytest.mark.skipif(
    not has_deep_ep,
    reason="Requires deep_ep kernels",
)

requires_deep_gemm = pytest.mark.skipif(
    not has_deep_gemm,
    reason="Requires deep_gemm kernels",
)

P = ParamSpec("P")


@dataclasses.dataclass
class ProcessGroupInfo:
    world_size: int
    world_local_size: int
    rank: int
    node_rank: int
    local_rank: int
    device: torch.device


def _worker_parallel_launch(
    local_rank: int,
    world_size: int,
    world_local_size: int,
    node_rank: int,
    init_method: str,
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    rank = node_rank * world_local_size + local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    barrier = torch.tensor([rank], device=device)
    torch.distributed.all_reduce(barrier)

    try:
        worker(
            ProcessGroupInfo(
                world_size=world_size,
                world_local_size=world_local_size,
                rank=rank,
                node_rank=node_rank,
                local_rank=local_rank,
                device=device,
            ),
            *args,
            **kwargs,
        )
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        raise
    finally:
        torch.distributed.destroy_process_group()


def parallel_launch(
    world_size: int,
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    assert not kwargs
    spawn(
        _worker_parallel_launch,
        args=(
            world_size,
            world_size,
            0,
            "tcp://localhost:29500",
            worker,
        ) + args,
        nprocs=world_size,
        join=True,
    )


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


def native_per_token_group_quant_fp8(x,
                                     group_size,
                                     eps=1e-10,
                                     dtype=torch.float8_e4m3fn):
    """Function to perform per-token-group quantization on an input tensor
    `x` using native torch."""
    assert x.shape[-1] % group_size == 0, ("the last dimension of `x` cannot "
                                           "be divisible by `group_size`")
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1,
                        keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size, ))

    return x_q, x_s


def make_block_quant_fp8_weights(
    e: int,
    n: int,
    k: int,
    block_size: list[int, int],
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

    #w1 = w1_bf16.to(torch.float8_e4m3fn)
    #w2 = w2_bf16.to(torch.float8_e4m3fn)

    #factor_for_scale = 1e-2
    #w1_s = torch.rand(
    #    (e, n_tiles_w1, k_tiles_w1), dtype=torch.float32,
    #    device="cuda") * factor_for_scale
    #w2_s = torch.rand(
    #    (e, n_tiles_w2, k_tiles_w2), dtype=torch.float32,
    #    device="cuda") * factor_for_scale

    #return w1, w2, w1_s, w2_s

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
    block_size: list[int, int]


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

        #local_experts = 16
        #es = rank * local_experts 
        #ee = es + local_experts

        topk = torch.randint(
            low=0,
            high=config.num_experts,
            size=(m, topk),
            device=torch.cuda.current_device()).to(dtype=torch.int64)

        topk_weights = torch.randn(topk.shape,
                                   dtype=torch.float32,
                                   device=torch.cuda.current_device())
        
        return TestTensors(rank_tokens=rank_tokens,
                           rank_token_scales=rank_token_scales,
                           topk=topk,
                           topk_weights=topk_weights,
                           config=config)


def make_deepep_a2a(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    dp_size: int,
    num_local_experts: int,
    block_shape: list[int, int],
    q_dtype: Optional[torch.dtype] = None,
):
    num_nvl_bytes = 1024 * 1024 * 1024  # 1GB
    num_rdma_bytes, low_latency_mode, num_qps_per_rank = 0, False, 1
    buffer = deep_ep.Buffer(group=pg,
                            num_nvl_bytes=num_nvl_bytes,
                            num_rdma_bytes=num_rdma_bytes,
                            low_latency_mode=low_latency_mode,
                            num_qps_per_rank=num_qps_per_rank)
    return DeepEPPrepareAndFinalize(buffer=buffer,
                                    world_size=pgi.world_size,
                                    rank=pgi.rank,
                                    dp_size=dp_size,
                                    rank_expert_offset=pgi.rank *
                                    num_local_experts,
                                    quant_dtype=q_dtype,
                                    block_shape=block_shape)


def make_modular_kernel(pg: ProcessGroup, pgi: ProcessGroupInfo, dp_size: int,
                        num_local_experts: int, q_dtype: Optional[torch.dtype],
                        block_shape: list[int, int]) -> FusedMoEModularKernel:

    a2a: DeepEPPrepareAndFinalize = \
        make_deepep_a2a(pg, pgi,
                        dp_size,
                        num_local_experts,
                        block_shape,
                        q_dtype)

    fused_experts = DeepGemmExperts()
    mk = FusedMoEModularKernel(prepare_finalize=a2a,
                               fused_experts=fused_experts)
    return mk


def deep_ep_moe_impl(pg: ProcessGroup, pgi: ProcessGroupInfo, dp_size: int,
                     test_tensors: TestTensors, w1: torch.Tensor,
                     w2: torch.Tensor, w1_scale: Optional[torch.Tensor],
                     w2_scale: Optional[torch.Tensor],
                     num_experts: int) -> torch.Tensor:

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

    q_dtype = torch.float8_e4m3fn

    # Make modular kernel
    mk: FusedMoEModularKernel = make_modular_kernel(
        pg, pgi, dp_size, num_local_experts, q_dtype,
        test_tensors.config.block_size)

    a1_scale = test_tensors.rank_token_scales

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


def torch_moe2(a:torch.Tensor,
               topk_ids:torch.Tensor,
               topk_weights: torch.Tensor,
               w1: torch.Tensor,
               w2: torch.Tensor,
               w1_scale: torch.Tensor,
               w2_scale: torch.Tensor,
               a1_scale: torch.Tensor,
               block_shape: list[int]):

    return fused_experts(hidden_states = a,
                  w1 = w1,
                  w2 = w2,
                  topk_weights = topk_weights,
                  topk_ids = topk_ids,
                  inplace = False,
                  use_fp8_w8a8 = True,
                  w1_scale = w1_scale, 
                  w2_scale = w2_scale, 
                  a1_scale = a1_scale,
                  block_shape = block_shape,
                  allow_deep_gemm = True)

def torch_moe_impl(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    M, K = a.shape
    topk = topk_ids.shape[1]

    a = a.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K)
    a, a_scale = per_token_group_quant_fp8(a, block_shape[1])
    torch.cuda.synchronize()
    did = 79
    top_ks = [9, 12]
    do_debug = torch.cuda.current_device() == 0 and False
    if do_debug:
        print(f"a {a.shape} {a[did*2]} | a_scale {a_scale.shape} {a_scale[did*2]}")

    out = torch.zeros(M * topk,
                      w2.shape[1],
                      dtype=torch.bfloat16,
                      device=a.device)
    num_experts = w1.shape[0]
    for i in range(num_experts):
        mask = (topk_ids == i).view(-1)

        tmp_idx = None 
        if do_debug:
            if i in top_ks:
                if i == top_ks[0]:
                    assert mask[did*2] == True
                if i == top_ks[1]:
                    assert mask[did*2 + 1] == True
                tmp_idx = torch.count_nonzero(mask[:did*2+1+1]) - 1


        if mask.sum():

            torch.cuda.synchronize()
            if tmp_idx is not None:
                print(f"tmp is {tmp_idx} {a[mask][tmp_idx]}  ")
                print(f"w1 scale is {w1_scale[i]}")

            tmp1 = native_w8a8_block_matmul(a[mask], w1[i], a_scale[mask],
                                            w1_scale[i], block_shape,
                                            torch.bfloat16)

            torch.cuda.synchronize()
            if do_debug and i in top_ks:
                print (f"expert {i} -- mm1 {tmp1[tmp_idx]}")

            tmp2 = SiluAndMul()(tmp1)
            tmp2, b_scale = per_token_group_quant_fp8(tmp2, block_shape[1])

            torch.cuda.synchronize()
            if do_debug and i in top_ks:
                print (f"expert {i} -- b {tmp2[tmp_idx]} | b_scale {b_scale[tmp_idx]}")

            tmp3 = native_w8a8_block_matmul(tmp2, w2[i], b_scale,
                                                 w2_scale[i], block_shape,
                                                 torch.bfloat16)

            torch.cuda.synchronize()
            if do_debug and i in top_ks:
                print (f"expert {i} -- mm2 {tmp3[tmp_idx]}")

            out[mask] = tmp3


    torch.cuda.synchronize()

    if do_debug:
        print(f"out {out.shape} {out[did*2]}", flush=True)
        print(f"out {out.shape} {out[did*2 + 1]}", flush=True)

    return (out.view(M, -1, w2.shape[1]).to(dtype=torch.float32) *
            topk_weight.view(M, -1, 1)).sum(dim=1).to(dtype=out.dtype)


def _deep_ep_moe(
    pgi: ProcessGroupInfo,
    dp_size: int,
    config: TestConfig,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
):
    current_platform.seed_everything(pgi.rank)
    #current_platform.seed_everything(7)

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

        torch_moe = torch_moe2(a = test_tensors.rank_tokens,
               topk_ids = test_tensors.topk,
               topk_weights = test_tensors.topk_weights,
               w1 = w1,
               w2 = w2,
               w1_scale = w1_scale,
               w2_scale = w2_scale,
               a1_scale = test_tensors.rank_token_scales,
               block_shape = block_shape)
        # torch_moe = torch_moe_impl(a=test_tensors.rank_tokens,
        #                           w1=w1,
        #                           w2=w2,
        #                           topk_weight=test_tensors.topk_weights,
        #                           topk_ids=test_tensors.topk,
        #                           w1_scale=w1_scale,
        #                           w2_scale=w2_scale,
        #                           block_shape=block_shape)

        # Splice experts for this rank.
        num_local_experts = config.num_experts // pgi.world_size
        e_start = num_local_experts * pgi.rank
        e_end = e_start + num_local_experts
        w1_ep = w1[e_start:e_end]
        w2_ep = w2[e_start:e_end]
        w1_scale_ep = w1_scale[e_start:e_end]
        w2_scale_ep = w2_scale[e_start:e_end]

        deepep_moe = deep_ep_moe_impl(
            pg,
            pgi,
            dp_size,
            test_tensors,
            w1_ep,
            w2_ep,
            w1_scale_ep,
            w2_scale_ep,
            config.num_experts,
        )

    #max_diff = torch.max(torch.abs(deepep_moe.to(torch.float32)
    #                   - torch_moe.to(torch.float32)))
    #rel_diff = (torch.mean(
    #    torch.abs(deepep_moe.to(torch.float32) - torch_moe.to(torch.float32)))
    #            / torch.mean(torch.abs(torch_moe.to(torch.float32))))

    torch.cuda.synchronize()

    #torch_moe = torch_moe.to(torch.float32)
    #deepep_moe = deepep_moe.to(torch.float32)

    rel_diff = torch.abs(deepep_moe - torch_moe) / torch.abs(torch.min(deepep_moe, torch_moe))
    rel_diff = torch.where(rel_diff > 1.0, rel_diff, 0)


    torch.cuda.synchronize()

    if pgi.rank == 0 and False:
        torch.set_printoptions(profile="full")
        #print (f"R0 torch 126: {torch_moe[126]}", flush=True)
        #print (f"R0 deepep 126: {deepep_moe[126]}", flush=True)
        print (f"R0 torch 127: {torch_moe[127]}", flush=True)
        #print (f"R0 deepep 127: {deepep_moe[127]}", flush=True)
        #print (f"relative diff : {rel_diff}", flush=True)

        print (f"R0 torch 127 36: {torch_moe[127][36]}", flush=True)
        print (f"R0 deepep 127 36 : {deepep_moe[127][36]}", flush=True)

        print (f"R0 torch 127 79: {torch_moe[127][79]}", flush=True)
        print (f"R0 deepep 127 79 : {deepep_moe[127][79]}", flush=True)
        #print (f"relative diff 127 79 : {rel_diff[127][79]}", flush=True)
        torch.set_printoptions(profile="default")

    if pgi.rank == 0 and False:
        torch.set_printoptions(profile="full")
        #print (f"R1 torch 126: {torch_moe[126]}", flush=True)
        #print (f"R1 deepep 126: {deepep_moe[126]}", flush=True)
        print (f"R1 torch 128: {torch_moe[128]}", flush=True)
        print (f"R1 deepep 128: {deepep_moe[128]}", flush=True)
        print (f"R1 relative diff 128 : {rel_diff[128]}", flush=True)

        #print (f"R1 torch 127 36: {torch_moe[127][36]}", flush=True)
        #print (f"R1 deepep 127 36 : {deepep_moe[127][36]}", flush=True)

        #print (f"R1 torch 127 79: {torch_moe[127][79]}", flush=True)
        #print (f"R1 deepep 127 79 : {deepep_moe[127][79]}", flush=True)
        #print (f"relative diff 127 79 : {rel_diff[127][79]}", flush=True)
        torch.set_printoptions(profile="default")

    if pgi.rank == 0:
        torch.testing.assert_close(
            torch_moe,
            deepep_moe,
            atol=1e-3,
            rtol=1e-3,
        )


MNKs = [
    #(129, 128, 128),
    #(222, 128, 128),
    (129, 128, 256),
    (129, 1024, 2048),
    (222, 1024, 2048),
    (128, 1024, 2048),
    (128, 1024, 2048),
    (128, 1024, 2048),
    (128, 128, 128),
    #(8, 128, 128),
    #(8, 128, 512),
    #(8, 512, 512),
    #(3, 1024, 2048),
    #(32, 128, 1024),
    #(45, 512, 2048),
    #(64, 1024, 1024),
]


@pytest.mark.parametrize("mnk", MNKs)
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("world_dp_size", [(2, 1)])
@requires_deep_ep
@requires_deep_gemm
def test_deep_ep_moe(mnk: tuple[int, int, int], num_experts: int, topk: int,
                     world_dp_size: tuple[int, int]):

    m, n, k = mnk
    current_platform.seed_everything(7)

    if topk > num_experts:
        pytest.skip(f"Skipping test: topk={topk} > E={num_experts}")

    block_m = deep_gemm.get_m_alignment_for_contiguous_layout()
    block_size = [block_m, block_m]

    world_size, dp_size = world_dp_size
    config = TestConfig(
        topk=topk,
        m=m,
        k=k,
        n=n,
        num_experts=num_experts,
        block_size=block_size,
    )

    w1, w2, w1_scale, w2_scale = make_block_quant_fp8_weights(
        num_experts, n, k, block_size)
    #print (w1_scale)
    #print (w2_scale)

    #w1_scale.fill_(0.0004)
    #w2_scale.fill_(0.0004)

    torch.set_printoptions(profile="full")
    #print (w1_scale)
    #print (w2_scale)
    torch.set_printoptions(profile="default")


    assert w1_scale.is_contiguous()
    assert w2_scale.is_contiguous()
    parallel_launch(world_size, _deep_ep_moe, dp_size, config, w1, w2,
                    w1_scale, w2_scale)
