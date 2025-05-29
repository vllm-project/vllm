# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MOE layers.

Run `pytest tests/kernels/test_pplx_moe.py`.
"""
from typing import Optional

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

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import override_config
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedExperts, BatchedPrepareAndFinalize, BatchedTritonExperts)
from vllm.model_executor.layers.fused_moe.fused_moe import (fused_topk,
                                                            get_default_config)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.platforms import current_platform
from vllm.utils import round_up

from .deepep_utils import ProcessGroupInfo, parallel_launch

requires_pplx = pytest.mark.skipif(
    not has_pplx,
    reason="Requires PPLX kernels",
)

PPLX_PREPARE_COMBOS = [(4, 128, 128), (32, 1024, 512), (64, 1024, 512),
                       (222, 2048, 1024)]

PPLX_MOE_COMBOS = [
    (1, 128, 128),
    (2, 128, 512),
    (3, 1024, 2048),
    (32, 128, 1024),
    (45, 512, 2048),
    (64, 1024, 1024),
    (222, 1024, 2048),
]

NUM_EXPERTS = [8, 64]
EP_SIZE = [1, 4]
TOP_KS = [1, 2, 6]

vllm_config = VllmConfig()
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192


def torch_prepare(
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


def torch_finalize(b_out: torch.Tensor, topk_weight: torch.Tensor,
                   topk_ids: torch.Tensor) -> torch.Tensor:
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


def torch_batched_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    num_experts = w1.shape[0]
    b_a, tokens_per_expert = torch_prepare(a, topk_ids, num_experts)
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

    return torch_finalize(out, topk_weight, topk_ids)


def batched_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    num_experts = w1.shape[0]

    fused_experts = FusedMoEModularKernel(
        BatchedPrepareAndFinalize(max_num_tokens=a.shape[0],
                                  world_size=1,
                                  dp_size=1,
                                  rank=0),
        BatchedExperts(max_num_tokens=a.shape[0], dp_size=1, world_size=1))

    return fused_experts(a, w1, w2, topk_weight, topk_ids, num_experts)


def native_w8a8_block_matmul(A: torch.Tensor,
                             B: torch.Tensor,
                             As: torch.Tensor,
                             Bs: torch.Tensor,
                             block_size,
                             output_dtype=torch.bfloat16):
    """This function performs matrix multiplication with block-wise
    quantization using native torch.
    It is agnostic to the input data type and can be used for both int8 and
    fp8 data types.

    It takes two input tensors `A` and `B` (int8) with scales `As` and
    `Bs` (float32).
    The output is returned in the specified `output_dtype`.
    """
    A = A.to(torch.float32)
    B = B.to(torch.float32).contiguous()
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1], (
        f"{(A.shape[-1] + block_k - 1) // block_k} == {As.shape[-1]}")
    assert A.shape[:-1] == As.shape[:-1], f"{A.shape} == {As.shape}"

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N, )
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0]
    assert k_tiles == Bs.shape[1]

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=torch.float32, device=A.device)

    A_tiles = [
        A[:, i * block_k:min((i + 1) * block_k, K)] for i in range(k_tiles)
    ]
    B_tiles = [[
        B[
            j * block_n:min((j + 1) * block_n, N),
            i * block_k:min((i + 1) * block_k, K),
        ] for i in range(k_tiles)
    ] for j in range(n_tiles)]
    C_tiles = [
        C[:, j * block_n:min((j + 1) * block_n, N)] for j in range(n_tiles)
    ]
    As_tiles = [As[:, i:i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C


# Note: same as torch_moe but with fused_topk factored out.
def torch_moe2(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    M, K = a.shape
    topk = topk_ids.shape[1]

    a = a.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K)

    if use_fp8_w8a8:
        a, a_scale = per_token_group_quant_fp8(a, block_shape[1])
    else:
        a_scale = None

    out = torch.zeros(M * topk,
                      w2.shape[1],
                      dtype=torch.bfloat16,
                      device=a.device)
    num_experts = w1.shape[0]
    for i in range(num_experts):
        mask = (topk_ids == i).view(-1)
        if mask.sum():
            if not use_fp8_w8a8:
                tmp1 = a[mask] @ w1[i].transpose(0, 1)
                tmp2 = SiluAndMul()(tmp1)
                out[mask] = tmp2 @ w2[i].transpose(0, 1)
            else:
                tmp1 = native_w8a8_block_matmul(a[mask], w1[i], a_scale[mask],
                                                w1_scale[i], block_shape,
                                                torch.bfloat16)

                tmp2 = SiluAndMul()(tmp1)
                tmp2, b_scale = per_token_group_quant_fp8(tmp2, block_shape[1])

                out[mask] = native_w8a8_block_matmul(tmp2, w2[i], b_scale,
                                                     w2_scale[i], block_shape,
                                                     torch.bfloat16)

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
    torch.testing.assert_close(baseline_output,
                               batched_output,
                               atol=2e-2,
                               rtol=0)


def rank_chunk(num: int, r: int, w: int) -> int:
    rem = num % w
    return (num // w) + (1 if r < rem else 0)


def chunk_by_rank(t: torch.Tensor, r: int, w: int) -> torch.Tensor:
    chunk = rank_chunk(t.shape[0], r, w)
    return t[(r * chunk):(r + 1) * chunk]


def pplx_prepare_finalize(
    pgi: ProcessGroupInfo,
    dp_size: int,
    a: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    group_name: Optional[str],
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.pplx_prepare_finalize import (
        PplxPrepareAndFinalize)

    assert torch.cuda.current_device() == pgi.local_rank

    topk = topk_ids.shape[1]
    num_tokens, hidden_dim = a.shape
    block_size = 128
    device = pgi.device
    rank = pgi.rank
    world_size = pgi.world_size
    max_num_tokens = rank_chunk(num_tokens, 0, world_size)

    args = dict(
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

    if group_name is None:
        ata = AllToAll.internode(**args)
    else:
        args["group_name"] = group_name
        ata = AllToAll.intranode(**args)

    topk_ids = topk_ids.to(dtype=torch.uint32)

    prepare_finalize = PplxPrepareAndFinalize(
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

    b_a, b_a_scale, expert_num_tokens, _, _ = prepare_finalize.prepare(
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

    prepare_finalize.finalize(
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


def _pplx_prepare_finalize(
    pgi: ProcessGroupInfo,
    dp_size: int,
    a: torch.Tensor,
    score: torch.Tensor,
    topk: torch.Tensor,
    num_experts: int,
    use_internode: bool,
):
    if use_internode:
        uid = nvshmem_get_unique_id(
        ) if pgi.rank == 0 else nvshmem_alloc_empty_unique_id()
        torch.distributed.broadcast(uid, src=0)
        nvshmem_init(uid, pgi.rank, pgi.world_size)
        group_name = None
    else:
        group_ranks = list(range(pgi.world_size))
        cpu_group = torch.distributed.new_group(group_ranks, backend="gloo")
        group_name = cpu_group.group_name

    device = pgi.device

    topk_weight, topk_ids, _ = fused_topk(a, score, topk, False)
    k = a.shape[1]

    a_rep = torch.repeat_interleave(a, topk, dim=0).to(device)

    torch_output = (a_rep.view(-1, topk, k) * 1.5 *
                    topk_weight.view(-1, topk, 1).to(device)).sum(dim=1).to(
                        a.dtype)

    pplx_output = pplx_prepare_finalize(pgi, dp_size, a, topk_weight, topk_ids,
                                        num_experts, group_name)

    torch_output = chunk_by_rank(torch_output, pgi.rank,
                                 pgi.world_size).to(pplx_output.device)

    torch.testing.assert_close(pplx_output, torch_output, atol=2e-2, rtol=0)

    if use_internode:
        nvshmem_finalize()


# TODO (bnell): this test point does not work for odd M due to how the test is
# written, not due to limitations of the pplx kernels.  The pplx_moe
# test below is able to deal with odd M.
@pytest.mark.parametrize("mnk", PPLX_PREPARE_COMBOS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("world_dp_size", [[2, 1]])
@pytest.mark.parametrize("use_internode", [False])
@requires_pplx
def test_pplx_prepare_finalize(
    mnk: tuple[int, int, int],
    e: int,
    topk: int,
    dtype: torch.dtype,
    world_dp_size: tuple[int, int],
    use_internode: bool,
):
    current_platform.seed_everything(7)
    m, n, k = mnk
    world_size, dp_size = world_dp_size
    device = "cuda"
    a = torch.randn((m, k), device=device, dtype=dtype) / 10
    score = torch.randn((m, e), device=device, dtype=dtype)

    parallel_launch(world_size, _pplx_prepare_finalize, dp_size, a, score,
                    topk, e, use_internode)


def pplx_moe(
    group_name: Optional[str],
    rank: int,
    world_size: int,
    dp_size: int,
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    qtype: Optional[torch.dtype] = None,
    block_shape: Optional[list[int]] = None,
    use_compile: bool = True,
    use_cudagraphs: bool = True,
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.pplx_prepare_finalize import (
        PplxPrepareAndFinalize)

    device = torch.device("cuda", rank)
    hidden_dim = a.shape[1]
    num_experts = w1.shape[0]
    block_size = block_shape[1] if block_shape is not None else 128
    topk = topk_ids.shape[1]
    max_num_tokens = round_up(rank_chunk(a.shape[0], 0, world_size), 64)

    if qtype is not None:
        a_dtype = qtype
        # This is probably not right
        scale_bytes = round_up(((hidden_dim + block_size - 1) // block_size) * torch.float32.itemsize, 16)
    else:
        a_dtype = a.dtype
        scale_bytes = 0

    args = dict(
        max_num_tokens=max_num_tokens,
        num_experts=num_experts,
        experts_per_token=topk,
        rank=rank,
        world_size=world_size,
        dp_size=dp_size,
        hidden_dim=hidden_dim,
        hidden_dim_bytes=hidden_dim * a_dtype.itemsize,
        hidden_dim_scale_bytes=scale_bytes,
    )

    if group_name is None:
        ata = AllToAll.internode(**args)
    else:
        args["group_name"] = group_name
        ata = AllToAll.intranode(**args)

    topk_ids = topk_ids.to(dtype=torch.uint32)

    prepare_finalize = PplxPrepareAndFinalize(
        ata,
        max_num_tokens,
        world_size,
        rank,
        dp_size,
        quant_dtype=qtype,
        block_shape=block_shape,
    )

    experts = BatchedTritonExperts(max_num_tokens=max_num_tokens,
                                   world_size=world_size,
                                   dp_size=dp_size,
                                   use_fp8_w8a8=qtype==torch.float8_e4m3fn,
                                   block_shape=block_shape)

    fused_experts = FusedMoEModularKernel(
        prepare_finalize,
        experts,
    )

    # Note: workers with the same dp_rank must use the exact same inputs.
    a_chunk = chunk_by_rank(a, rank, world_size).to(device)
    chunk_topk_weight = chunk_by_rank(topk_weight, rank, world_size).to(device)
    chunk_topk_ids = chunk_by_rank(topk_ids, rank, world_size).to(device)

    # Chunking weights like this only works for batched format
    w1_chunk = chunk_by_rank(w1, rank, world_size).to(device)
    w2_chunk = chunk_by_rank(w2, rank, world_size).to(device)

    if w1_scale is not None:
        w1_scale_chunk = chunk_by_rank(w1_scale, rank, world_size).to(device)
        w2_scale_chunk = chunk_by_rank(w2_scale, rank, world_size).to(device)
    else:
        w1_scale_chunk = None
        w2_scale_chunk = None

    if False and use_compile:
        _fused_experts = torch.compile(fused_experts,
                                       backend='inductor',
                                       fullgraph=True)
    else:
        _fused_experts = fused_experts

    out = _fused_experts(a_chunk,
                         w1_chunk,
                         w2_chunk,
                         chunk_topk_weight,
                         chunk_topk_ids,
                         w1_scale=w1_scale_chunk,
                         w2_scale=w2_scale_chunk,
                         global_num_experts=num_experts)

    if False and use_cudagraphs: #XXXXXXXXXXXX
        out.fill_(0)
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = _fused_experts(a_chunk,
                                 w1_chunk,
                                 w2_chunk,
                                 chunk_topk_weight,
                                 chunk_topk_ids,
                                 w1_scale=w1_scale_chunk,
                                 w2_scale=w2_scale_chunk,
                                 global_num_experts=num_experts)

        torch.cuda.synchronize()
        graph.replay()

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

    prepare_finalize = BatchedPrepareAndFinalize(
        max_num_tokens=max_num_tokens,
        world_size=world_size,
        dp_size=dp_size,
        rank=rank,
    )

    experts = BatchedExperts(max_num_tokens=a.shape[0],
                             world_size=1,
                             dp_size=1)

    fused_experts = FusedMoEModularKernel(
        prepare_finalize,
        experts,
    )

    # Note: workers with the same dp_rank must use the exact same inputs.
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
    w1_s: Optional[torch.Tensor] = None,
    w2_s: Optional[torch.Tensor] = None,
    qtype: Optional[torch.dtype] = None,
    block_shape: Optional[list[int]] = None,
    use_internode: bool,
):
    if use_internode:
        uid = nvshmem_get_unique_id(
        ) if pgi.rank == 0 else nvshmem_alloc_empty_unique_id()
        torch.distributed.broadcast(uid, src=0)
        nvshmem_init(uid, pgi.rank, pgi.world_size)
        group_name = None
    else:
        group_ranks = list(range(pgi.world_size))
        cpu_group = torch.distributed.new_group(group_ranks, backend="gloo")
        group_name = cpu_group.group_name

    m, k = a.shape
    e, _, n = w2.shape

    moe_config = get_default_config(m, e, n, k, topk, a.dtype, False)

    use_fp8_w8a8 = qtype == torch.float8_e4m3fn

    device = torch.device("cuda", pgi.rank)
    a = a.to(device)
    w1 = w1.to(device)
    w2 = w2.to(device)
    w1_s = w1_s.to(device) if w1_s is not None else None
    w2_s = w2_s.to(device) if w2_s is not None else None

    with set_current_vllm_config(vllm_config), override_config(moe_config):
        topk_weight, topk_ids, _ = fused_topk(a, score, topk, False)
        torch_output = torch_moe2(a, w1, w2, topk_weight, topk_ids, w1_s, w2_s, use_fp8_w8a8, block_shape)
        pplx_output = pplx_moe(group_name, pgi.rank, pgi.world_size, dp_size, a, w1, w2,
                               topk_weight, topk_ids, w1_s, w2_s, qtype, block_shape)
        # TODO (bnell): fix + re-enable
        #batched_output = _batched_moe(pgi, dp_size, a, w1, w2, topk_weight,
        #                              topk_ids)

    torch_output = chunk_by_rank(torch_output, pgi.rank,
                                 pgi.world_size).to(pplx_output.device)

    torch.testing.assert_close(pplx_output, torch_output, atol=2e-2, rtol=0)
    #torch.testing.assert_close(batched_output, torch_output, atol=2e-2, rtol=0)

    if use_internode:
        nvshmem_finalize()


@pytest.mark.parametrize("mnk", PPLX_MOE_COMBOS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("world_dp_size", [[2, 1]])
@pytest.mark.parametrize("use_internode", [False])
@requires_pplx
def test_pplx_moe(
    mnk: tuple[int, int, int],
    e: int,
    topk: int,
    dtype: torch.dtype,
    world_dp_size: tuple[int, int],
    use_internode: bool,
):
    current_platform.seed_everything(7)
    m, n, k = mnk
    world_size, dp_size = world_dp_size
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=torch.bfloat16) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=torch.bfloat16) / 10
    score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)

    use_fp8_w8a8 = dtype == torch.float8_e4m3fn

    if use_fp8_w8a8:
        block_shape = [128, 128]
        quant_type = torch.float8_e4m3fn
        block_n, block_k = block_shape[0], block_shape[1]
        n_tiles_w1 = (2 * n + block_n - 1) // block_n
        n_tiles_w2 = (k + block_n - 1) // block_n
        k_tiles_w1 = (k + block_k - 1) // block_k
        k_tiles_w2 = (n + block_k - 1) // block_k

        finfo = torch.finfo(dtype)
        fp8_min = finfo.min
        fp8_max = finfo.max

        w1 = w1.clamp(min=fp8_min, max=fp8_max).to(dtype)
        w2 = w2.clamp(min=fp8_min, max=fp8_max).to(dtype)

        factor_for_scale = 1e-2
        w1_s = torch.rand(
            (e, n_tiles_w1, k_tiles_w1), dtype=torch.float32,
            device="cuda") * factor_for_scale
        w2_s = torch.rand(
            (e, n_tiles_w2, k_tiles_w2), dtype=torch.float32,
            device="cuda") * factor_for_scale
    else:
        block_shape = None
        quant_type = None
        w1_s = None
        w2_s = None

    parallel_launch(world_size, _pplx_moe, dp_size, a, w1, w2, score, topk, w1_s, w2_s, quant_type, block_shape, use_internode)
