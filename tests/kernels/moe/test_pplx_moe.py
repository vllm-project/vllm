# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest tests/kernels/test_pplx_moe.py`.
"""
import dataclasses
import os
import traceback
from typing import Callable, Optional

import pytest
import torch

try:
    from pplx_kernels import AllToAll
    from pplx_kernels.nvshmem import (nvshmem_alloc_empty_unique_id,
                                      nvshmem_finalize, nvshmem_get_unique_id,
                                      nvshmem_init)
    has_pplx = True
except ImportError:
    has_pplx = False

from torch.multiprocessing import (
    spawn)  # pyright: ignore[reportPrivateImportUsage]
from typing_extensions import Concatenate, ParamSpec

import vllm.model_executor.layers.fused_moe  # noqa
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedDispatchCombine, BatchedExperts)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.model_executor.layers.fused_moe.pplx_dispatch_combine import (
    PplxDispatchCombine)
from vllm.platforms import current_platform

NUM_EXPERTS = [8, 64]
EP_SIZE = [1, 4]
TOP_KS = [2, 6]

vllm_config = VllmConfig()
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192

P = ParamSpec("P")

requires_pplx = pytest.mark.skipif(
    not has_pplx,
    reason="Requires PPLX kernels",
)


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


def parallel_launch_from_env(
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    """
    Launches a worker function in parallel across all processes in the current
    environment. The environment must have the following variables set:
    - WORLD_SIZE: The total number of processes.
    - WORLD_LOCAL_SIZE: The number of processes on the current node.
    - NODE_RANK: The rank of the current
    - MASTER_ADDR: The address of the master process.
    - MASTER_PORT: The port of the master process.
    """
    assert not kwargs
    world_size = int(os.environ["WORLD_SIZE"])
    world_local_size = int(os.environ["WORLD_LOCAL_SIZE"])
    node_rank = int(os.environ["NODE_RANK"])
    assert "MASTER_ADDR" in os.environ
    assert "MASTER_PORT" in os.environ
    spawn(
        _worker_parallel_launch,
        args=(
            world_size,
            world_local_size,
            node_rank,
            "env://",
            worker,
        ) + args,
        nprocs=world_local_size,
        join=True,
    )


def torch_dispatch(
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    max_num_tokens: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert topk_ids.dim() == 2
    assert topk_ids.shape[0] == a.shape[0]

    num_tokens, hidden_dim = a.shape
    topk = topk_ids.shape[1]

    tokens_per_expert = torch.bincount(topk_ids.view(-1),
                                       minlength=num_experts)

    assert tokens_per_expert.numel() == num_experts

    if max_num_tokens is None:
        max_num_tokens = int(tokens_per_expert.max().item())

    b_a = torch.zeros((num_experts, max_num_tokens, hidden_dim),
                      dtype=a.dtype,
                      device=a.device)

    token_counts = torch.zeros(num_experts, dtype=torch.int, device=a.device)

    for token in range(num_tokens):
        for j in range(topk):
            expert_id = topk_ids[token, j]
            idx = token_counts[expert_id]
            b_a[expert_id, idx:idx + 1, :] = a[token, :]
            token_counts[expert_id] = token_counts[expert_id] + 1

    return b_a, tokens_per_expert


def torch_combine(b_out, topk_weight, topk_ids):
    num_tokens = topk_ids.shape[0]
    num_experts = b_out.shape[0]
    K = b_out.shape[-1]
    out = torch.zeros((num_tokens, K), dtype=b_out.dtype, device=b_out.device)
    expert_counts = torch.zeros(num_experts,
                                dtype=torch.int,
                                device=b_out.device)
    for token in range(num_tokens):
        expert_ids = topk_ids[token]
        for i in range(expert_ids.numel()):
            expert_id = expert_ids[i]
            idx = expert_counts[expert_id]
            out[token, :] = out[token, :] + b_out[expert_id, idx:idx +
                                                  1, :] * topk_weight[token, i]
            expert_counts[expert_id] = expert_counts[expert_id] + 1

    return out


def torch_batched_moe(a, w1, w2, topk_weight, topk_ids):
    num_experts = w1.shape[0]
    b_a, tokens_per_expert = torch_dispatch(a, topk_ids, num_experts)
    assert b_a.dim() == 3
    num_tokens, topk = topk_ids.shape
    _, max_num_tokens, K = b_a.shape
    assert num_experts == b_a.shape[0] and w2.shape[1] == K
    out = torch.zeros((num_experts, max_num_tokens, K),
                      dtype=b_a.dtype,
                      device=b_a.device)
    tmp = torch.empty((max_num_tokens, w1.shape[1] // 2),
                      dtype=b_a.dtype,
                      device=b_a.device)
    for expert in range(num_experts):
        num = tokens_per_expert[expert]
        if num > 0:
            torch.ops._C.silu_and_mul(
                tmp[:num], b_a[expert, :num, :] @ w1[expert].transpose(0, 1))
            out[expert, :num, :] = tmp[:num] @ w2[expert].transpose(0, 1)

    return torch_combine(out, topk_weight, topk_ids)


def batched_moe(a, w1, w2, topk_weight, topk_ids):
    num_experts = w1.shape[0]

    fused_experts = FusedMoEModularKernel(
        BatchedDispatchCombine(a.shape[0], world_size=1, dp_size=1, rank=0),
        BatchedExperts(max_num_tokens=a.shape[0], dp_size=1, world_size=1))

    return fused_experts(a, w1, w2, topk_weight, topk_ids, num_experts)


# TODO: same as torch_moe but with fused_topk factored out.
def torch_moe2(a, w1, w2, topk_weight, topk_ids):
    M, K = a.shape
    topk = topk_ids.shape[1]
    a = a.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K)
    out = torch.zeros(M * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    num_experts = w1.shape[0]
    for i in range(num_experts):
        mask = (topk_ids == i).view(-1)
        if mask.sum():
            out[mask] = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)

    return (out.view(M, -1, w2.shape[1]) *
            topk_weight.view(M, -1, 1).to(out.dtype)).sum(dim=1)


@pytest.mark.parametrize("m", [1, 33, 64, 222])
@pytest.mark.parametrize("n", [128, 1024, 2048])
@pytest.mark.parametrize("k", [128, 512, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_moe_batched_experts(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    current_platform.seed_everything(7)

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    with set_current_vllm_config(vllm_config):
        topk_weight, topk_ids, _ = fused_topk(a, score, topk, False)
        baseline_output = torch_moe2(a, w1, w2, topk_weight, topk_ids)
        torch_output = torch_batched_moe(a, w1, w2, topk_weight, topk_ids)
        batched_output = batched_moe(a, w1, w2, topk_weight, topk_ids)

    torch.testing.assert_close(baseline_output,
                               torch_output,
                               atol=2e-2,
                               rtol=0)
    torch.set_printoptions(profile="full")
    torch.testing.assert_close(baseline_output,
                               batched_output,
                               atol=2e-2,
                               rtol=0)


def rank_chunk(num, r, w):
    rem = num % w
    return (num // w) + (1 if r < rem else 0)


def chunk_by_rank(t, r, w):
    chunk = rank_chunk(t.shape[0], r, w)
    return t[(r * chunk):(r + 1) * chunk]


def pplx_dispatch_combine(pgi, dp_size, a, topk_weight, topk_ids, num_experts):
    assert torch.cuda.current_device() == pgi.local_rank

    topk = topk_ids.shape[1]
    num_tokens, hidden_dim = a.shape
    block_size = 128
    device = pgi.device
    rank = pgi.rank
    world_size = pgi.world_size
    max_num_tokens = rank_chunk(num_tokens, 0, world_size)

    ata = AllToAll.internode(
        max_num_tokens=max_num_tokens,
        num_experts=num_experts,
        experts_per_token=topk,
        rank=rank,
        world_size=world_size,
        dp_size=dp_size,
        hidden_dim=hidden_dim,
        hidden_dim_bytes=hidden_dim * a.dtype.itemsize,
        hidden_dim_scale_bytes=(0 if a.dtype.itemsize != 1 else
                                ((hidden_dim + block_size - 1) // block_size *
                                 torch.float32.itemsize)),
    )

    topk_ids = topk_ids.to(dtype=torch.uint32)

    dispatch_combine = PplxDispatchCombine(
        ata,
        max_num_tokens,
        world_size,
        rank,
        dp_size,
        a.dtype,
    )

    a_chunk = chunk_by_rank(a, rank, world_size).to(device)
    chunk_topk_weight = chunk_by_rank(topk_weight, rank, world_size).to(device)
    chunk_topk_ids = chunk_by_rank(topk_ids, rank, world_size).to(device)

    b_a, b_a_scale, expert_num_tokens = dispatch_combine.dispatch(
        a_chunk,
        None,
        None,
        chunk_topk_weight,
        chunk_topk_ids,
        num_experts,
        None,
        False,
    )

    b_a = b_a * 1.5

    out = torch.full(
        (max_num_tokens, hidden_dim),
        torch.nan,
        dtype=a.dtype,
        device=device,
    )

    dispatch_combine.combine(
        out,
        b_a,
        chunk_topk_weight,
        chunk_topk_ids,
        False,
    )

    torch.cuda.synchronize()

    ata.destroy()

    num_tokens = a_chunk.shape[0]

    return out[:num_tokens]


def _pplx_dispatch_combine(
    pgi: ProcessGroupInfo,
    dp_size: int,
    a,
    score,
    topk,
    num_experts,
):
    uid = nvshmem_get_unique_id(
    ) if pgi.rank == 0 else nvshmem_alloc_empty_unique_id()
    torch.distributed.broadcast(uid, src=0)
    nvshmem_init(uid, pgi.rank, pgi.world_size)
    device = pgi.device

    topk_weight, topk_ids, _ = fused_topk(a, score, topk, False)
    k = a.shape[1]

    a_rep = torch.repeat_interleave(a, topk, dim=0).to(device)

    torch_output = (a_rep.view(-1, topk, k) * 1.5 *
                    topk_weight.view(-1, topk, 1).to(device)).sum(dim=1).to(
                        a.dtype)

    pplx_output = pplx_dispatch_combine(pgi, dp_size, a, topk_weight, topk_ids,
                                        num_experts)

    torch_output = chunk_by_rank(torch_output, pgi.rank,
                                 pgi.world_size).to(pplx_output.device)

    torch.testing.assert_close(pplx_output, torch_output, atol=2e-2, rtol=0)

    nvshmem_finalize()


# TODO: this test point does not work for M == 1
@pytest.mark.parametrize("m", [4, 32, 64, 222])
@pytest.mark.parametrize("n", [128, 1024, 2048])
@pytest.mark.parametrize("k", [128, 512, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("world_dp_size", [[2, 1]])
@requires_pplx
def test_pplx_dispatch_combine(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    world_dp_size: tuple[int, int],
):
    current_platform.seed_everything(7)
    world_size, dp_size = world_dp_size
    device = "cuda"
    a = torch.randn((m, k), device=device, dtype=dtype) / 10
    score = torch.randn((m, e), device=device, dtype=dtype)

    parallel_launch(world_size, _pplx_dispatch_combine, dp_size, a, score,
                    topk, e)


def pplx_moe(pgi, dp_size, a, w1, w2, topk_weight, topk_ids):
    assert torch.cuda.current_device() == pgi.local_rank

    hidden_dim = a.shape[1]
    num_experts = w1.shape[0]
    block_size = 128
    device = pgi.device
    rank = pgi.rank
    world_size = pgi.world_size
    topk = topk_ids.shape[1]
    max_num_tokens = rank_chunk(a.shape[0], 0, world_size)

    ata = AllToAll.internode(
        max_num_tokens=max_num_tokens,
        num_experts=num_experts,
        experts_per_token=topk,
        rank=rank,
        world_size=world_size,
        dp_size=dp_size,
        hidden_dim=hidden_dim,
        hidden_dim_bytes=hidden_dim * a.dtype.itemsize,
        hidden_dim_scale_bytes=(0 if a.dtype.itemsize != 1 else
                                ((hidden_dim + block_size - 1) // block_size *
                                 torch.float32.itemsize)),
    )

    topk_ids = topk_ids.to(dtype=torch.uint32)

    dispatch_combine = PplxDispatchCombine(
        ata,
        max_num_tokens,
        world_size,
        rank,
        dp_size,
    )

    experts = BatchedExperts(max_num_tokens=a.shape[0],
                             world_size=world_size,
                             dp_size=dp_size)

    fused_experts = FusedMoEModularKernel(
        dispatch_combine,
        experts,
    )

    # TODO: workers with the same dp_rank must use the exact same inputs.

    a_chunk = chunk_by_rank(a, rank, world_size).to(device)
    chunk_topk_weight = chunk_by_rank(topk_weight, rank, world_size).to(device)
    chunk_topk_ids = chunk_by_rank(topk_ids, rank, world_size).to(device)

    out = fused_experts(
        a_chunk,
        # Chunking weights like this only works for batched format
        chunk_by_rank(w1, rank, world_size).to(device),
        chunk_by_rank(w2, rank, world_size).to(device),
        chunk_topk_weight,
        chunk_topk_ids,
        global_num_experts=num_experts)

    torch.cuda.synchronize()

    ata.destroy()

    return out


def _batched_moe(pgi, dp_size, a, w1, w2, topk_weight, topk_ids):
    assert torch.cuda.current_device() == pgi.local_rank

    num_experts = w1.shape[0]
    device = pgi.device
    rank = pgi.rank
    world_size = pgi.world_size
    max_num_tokens = rank_chunk(a.shape[0], 0, world_size)

    dispatch_combine = BatchedDispatchCombine(
        max_num_tokens=max_num_tokens,
        world_size=world_size,
        dp_size=dp_size,
        rank=rank,
    )

    experts = BatchedExperts(max_num_tokens=a.shape[0],
                             world_size=1,
                             dp_size=1)

    fused_experts = FusedMoEModularKernel(
        dispatch_combine,
        experts,
    )

    # TODO: workers with the same dp_rank must use the exact same inputs.

    a_chunk = chunk_by_rank(a, rank, world_size).to(device)
    chunk_topk_weight = chunk_by_rank(topk_weight, rank, world_size).to(device)
    chunk_topk_ids = chunk_by_rank(topk_ids, rank, world_size).to(device)

    out = fused_experts(
        a_chunk,
        # Chunking weights like this only works for batched format
        chunk_by_rank(w1, rank, world_size).to(device),
        chunk_by_rank(w2, rank, world_size).to(device),
        chunk_topk_weight,
        chunk_topk_ids,
        global_num_experts=num_experts)

    return out


def _pplx_moe(
    pgi: ProcessGroupInfo,
    dp_size: int,
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    topk: int,
):
    uid = nvshmem_get_unique_id(
    ) if pgi.rank == 0 else nvshmem_alloc_empty_unique_id()
    torch.distributed.broadcast(uid, src=0)
    nvshmem_init(uid, pgi.rank, pgi.world_size)

    m, k = a.shape
    e, _, n = w2.shape

    with set_current_vllm_config(vllm_config):
        topk_weight, topk_ids, _ = fused_topk(a, score, topk, False)
        torch_output = torch_moe2(a, w1, w2, topk_weight, topk_ids)
        pplx_output = pplx_moe(pgi, dp_size, a, w1, w2, topk_weight, topk_ids)
        batched_output = _batched_moe(pgi, dp_size, a, w1, w2, topk_weight,
                                      topk_ids)

    torch_output = chunk_by_rank(torch_output, pgi.rank,
                                 pgi.world_size).to(pplx_output.device)

    torch.testing.assert_close(pplx_output, torch_output, atol=2e-2, rtol=0)
    torch.testing.assert_close(batched_output, torch_output, atol=2e-2, rtol=0)

    nvshmem_finalize()


@pytest.mark.parametrize("m", [1, 2, 3, 32, 45, 64, 222])
@pytest.mark.parametrize("n", [128, 1024, 2048])
@pytest.mark.parametrize("k", [128, 512, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("world_dp_size", [[2, 1]])
@requires_pplx
def test_pplx_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    world_dp_size: tuple[int, int],
):
    current_platform.seed_everything(7)
    world_size, dp_size = world_dp_size
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    parallel_launch(world_size, _pplx_moe, dp_size, a, w1, w2, score, topk)
