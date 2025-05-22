# SPDX-License-Identifier: Apache-2.0
"""
Test deepep low-latency dispatch-combine logic
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

from vllm import _custom_ops as ops
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import (
    DeepEPLLPrepareAndFinalize)
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedTritonExperts)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.platforms import current_platform

try:
    import deep_ep
    has_deep_ep = True
except ImportError:
    has_deep_ep = False

requires_deep_ep = pytest.mark.skipif(
    not has_deep_ep,
    reason="Requires deep_ep kernels",
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


def make_weights(
        e, n, k, dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return weights w1, w2, w1_scale, w2_scale
    """
    if dtype in [torch.float16, torch.bfloat16]:
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        return w1, w2, None, None

    # per-out-channel weight quantization
    assert dtype == torch.float8_e4m3fn
    w1 = torch.empty((e, 2 * n, k), device="cuda", dtype=torch.float16)
    w2 = torch.empty((e, k, n), device="cuda", dtype=torch.float16)

    n_b_scales = 2 * n
    k_b_scales = k
    w1_q = torch.empty_like(w1, dtype=dtype)
    w2_q = torch.empty_like(w2, dtype=dtype)
    w1_scale = torch.empty((e, n_b_scales, 1),
                           device="cuda",
                           dtype=torch.float32)
    w2_scale = torch.empty((e, k_b_scales, 1),
                           device="cuda",
                           dtype=torch.float32)
    for expert in range(e):
        w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(
            w1[expert], use_per_token_if_dynamic=True)
        w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(
            w2[expert], use_per_token_if_dynamic=True)
    return w1_q, w2_q, w1_scale, w2_scale


@dataclasses.dataclass
class TestConfig:
    dtype: torch.dtype
    topk: int
    m: int
    k: int
    n: int
    num_experts: int


@dataclasses.dataclass
class TestTensors:
    rank_tokens: torch.Tensor  # all ranks make this many tokens
    rank_token_scales: Optional[torch.Tensor]
    topk: torch.Tensor
    topk_weights: torch.Tensor
    config: TestConfig

    @staticmethod
    def make(config: TestConfig) -> "TestTensors":
        # TODO (varun) - check that float16 works ?
        assert config.dtype in [torch.bfloat16, torch.float8_e4m3fn]
        token_dtype = (torch.bfloat16 if config.dtype == torch.float8_e4m3fn
                       else config.dtype)
        rank_tokens = torch.randn(
            (config.m, config.k), device="cuda", dtype=token_dtype) / 10
        rank_token_scales = None
        if config.dtype == torch.float8_e4m3fn:
            _, rank_token_scales = ops.scaled_fp8_quant(
                rank_tokens, use_per_token_if_dynamic=True)

        topk = torch.randint(low=0,
                             high=config.num_experts,
                             size=(config.m, config.topk),
                             device="cuda").to(dtype=torch.int64)
        topk_weights = torch.randn(topk.shape,
                                   dtype=torch.float32,
                                   device="cuda")
        return TestTensors(rank_tokens=rank_tokens,
                           rank_token_scales=rank_token_scales,
                           topk=topk,
                           topk_weights=topk_weights,
                           config=config)


def deep_ep_moe_impl(pg: ProcessGroup, pgi: ProcessGroupInfo, dp_size: int,
                     test_tensors: TestTensors, w1: torch.Tensor,
                     w2: torch.Tensor, w1_scale: Optional[torch.Tensor],
                     w2_scale: Optional[torch.Tensor],
                     num_experts: int) -> torch.Tensor:

    MAX_TOKENS_PER_RANK = 64
    num_local_experts = w1.size(0)
    hidden_size = test_tensors.rank_tokens.size(1)
    num_ranks = pgi.world_size

    def make_a2a():

        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            MAX_TOKENS_PER_RANK, hidden_size, num_ranks, num_experts)
        buffer = deep_ep.Buffer(group=pg,
                                num_rdma_bytes=num_rdma_bytes,
                                low_latency_mode=True,
                                num_qps_per_rank=num_experts // num_ranks)

        q_dtype = (torch.float8_e4m3fn
                   if w1.dtype == torch.float8_e4m3fn else None)
        return DeepEPLLPrepareAndFinalize(
            buffer=buffer,
            world_size=pgi.world_size,
            rank=pgi.rank,
            dp_size=dp_size,
            rank_expert_offset=pgi.rank * num_local_experts,
            max_tokens_per_rank=MAX_TOKENS_PER_RANK,
            quant_dtype=q_dtype)

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

    is_quantized = w1.dtype == torch.float8_e4m3fn

    # Make modular kernel
    a2a = make_a2a()

    fused_experts = BatchedTritonExperts(max_num_tokens=MAX_TOKENS_PER_RANK,
                                         world_size=pgi.world_size,
                                         dp_size=dp_size,
                                         use_fp8_w8a8=is_quantized,
                                         use_int8_w8a8=False,
                                         use_int8_w8a16=False,
                                         use_int4_w4a16=False)

    mk = FusedMoEModularKernel(prepare_finalize=a2a,
                               fused_experts=fused_experts)

    out_hidden_states = torch.empty_like(test_tensors.rank_tokens)
    total_num_tokens = test_tensors.rank_tokens.size(0)

    def process_chunk(chunk_start, chunk_end, skip_result_store=False):
        rank_tokens_chunk = test_tensors.rank_tokens[chunk_start:chunk_end]
        topk_weights_chunk = test_tensors.topk_weights[chunk_start:chunk_end]
        topk_chunk = test_tensors.topk[chunk_start:chunk_end]
        rank_token_scales_chunk = test_tensors.rank_token_scales
        if rank_token_scales_chunk is not None and rank_token_scales_chunk.size(
                0) == total_num_tokens:
            # per act token
            rank_token_scales_chunk = rank_token_scales_chunk[
                chunk_start:chunk_end]

        out = mk.forward(hidden_states=rank_tokens_chunk,
                         w1=w1,
                         w2=w2,
                         topk_weights=topk_weights_chunk,
                         topk_ids=topk_chunk,
                         inplace=False,
                         activation="silu",
                         global_num_experts=num_experts,
                         expert_map=build_expert_map(),
                         w1_scale=w1_scale,
                         w2_scale=w2_scale,
                         w1_zp=None,
                         w2_zp=None,
                         a1_scale=rank_token_scales_chunk,
                         a2_scale=None,
                         apply_router_weight_on_input=False)

        if not skip_result_store:
            out_hidden_states[chunk_start:chunk_end, :].copy_(
                out, non_blocking=True)

    for chunk_start_ in range(0, total_num_tokens, MAX_TOKENS_PER_RANK):
        chunk_start = chunk_start_
        chunk_end = min(chunk_start + MAX_TOKENS_PER_RANK, total_num_tokens)
        # clamp start and end
        chunk_start = min(chunk_start, total_num_tokens - 1)
        chunk_end = min(chunk_end, total_num_tokens)

        process_chunk(chunk_start,
                      chunk_end,
                      skip_result_store=chunk_start_ >= total_num_tokens)

    return out_hidden_states


def torch_moe_impl(test_tensors: TestTensors, w1: torch.Tensor,
                   w2: torch.Tensor, w1_scale: Optional[torch.Tensor],
                   w2_scale: Optional[torch.Tensor]):
    is_quantized = w1.dtype == torch.float8_e4m3fn

    a = test_tensors.rank_tokens
    a_dtype = a.dtype
    if is_quantized:
        w1 = w1.to(dtype=torch.float32) * w1_scale
        w2 = w2.to(dtype=torch.float32) * w2_scale
        a = a.to(dtype=torch.float32)

    m, _ = test_tensors.rank_tokens.shape
    topk = test_tensors.topk.size(1)
    out = torch.zeros_like(a)

    for i in range(m):
        a_i = a[i]
        o_i = out[i]
        for j in range(topk):
            e = test_tensors.topk[i][j]
            e_w = test_tensors.topk_weights[i][j]
            w1_e = w1[e]
            w2_e = w2[e]
            o_i += (SiluAndMul()
                    (a_i @ w1_e.transpose(0, 1)) @ w2_e.transpose(0, 1)) * e_w

    if is_quantized:
        out = out.to(dtype=a_dtype)

    return out


def _deep_ep_moe(
    pgi: ProcessGroupInfo,
    dp_size: int,
    config: TestConfig,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
):
    is_quantized = w1.dtype == torch.float8_e4m3fn
    w1 = w1.to(device=torch.cuda.current_device())
    w2 = w2.to(device=torch.cuda.current_device())
    if is_quantized:
        w1_scale = w1_scale.to(device=torch.cuda.current_device())
        w2_scale = w2_scale.to(device=torch.cuda.current_device())

    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    test_tensors = TestTensors.make(config)

    torch_combined = torch_moe_impl(test_tensors, w1, w2, w1_scale, w2_scale)

    num_local_experts = config.num_experts // pgi.world_size
    e_start = num_local_experts * pgi.rank
    e_end = e_start + num_local_experts
    w1_ep = w1[e_start:e_end]
    w2_ep = w2[e_start:e_end]

    w1_scale_ep, w2_scale_ep = None, None
    if is_quantized:
        w1_scale_ep = w1_scale[e_start:e_end]
        w2_scale_ep = w2_scale[e_start:e_end]
    deepep_combined = deep_ep_moe_impl(pg, pgi, dp_size, test_tensors, w1_ep,
                                       w2_ep, w1_scale_ep, w2_scale_ep,
                                       config.num_experts)

    torch.testing.assert_close(torch_combined,
                               deepep_combined,
                               atol=6e-2,
                               rtol=6e-2)


MNKs = [
    (1, 128, 128),
    (2, 128, 512),
    (3, 1024, 2048),
    (32, 128, 1024),
    (45, 512, 2048),
    (64, 1024, 1024),
    (222, 1024, 2048),
]

DTYPES = [torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mnk", MNKs)
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("topk", [6])
@pytest.mark.parametrize("world_dp_size", [(2, 1)])
@requires_deep_ep
def test_deep_ep_moe(
    dtype: torch.dtype,
    mnk: tuple[int, int, int],
    num_experts: int,
    topk: int,
    world_dp_size: tuple[int, int],
):
    current_platform.seed_everything(7)
    world_size, dp_size = world_dp_size
    m, n, k = mnk

    if k not in [512, 1024, 2048, 2560, 4096, 5120, 7168]:
        pytest.skip(f"DeepEP low-latency kernels dont support hidden-size {k}")

    config = TestConfig(dtype=dtype,
                        topk=topk,
                        m=m,
                        k=k,
                        n=n,
                        num_experts=num_experts)

    w1, w2, w1_scale, w2_scale = make_weights(num_experts, n, k, dtype)

    parallel_launch(world_size, _deep_ep_moe, dp_size, config, w1, w2,
                    w1_scale, w2_scale)
