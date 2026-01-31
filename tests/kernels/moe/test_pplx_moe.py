# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MOE layers.

Run `pytest tests/kernels/test_pplx_moe.py`.
"""

import copy
import itertools
import textwrap
import traceback
from collections.abc import Callable

import pytest
import torch

try:
    from pplx_kernels import AllToAll
    from pplx_kernels.nvshmem import (
        nvshmem_alloc_empty_unique_id,
        nvshmem_finalize,
        nvshmem_get_unique_id,
        nvshmem_init,
    )

    has_pplx = True
except ImportError:
    has_pplx = False

from tests.kernels.moe.modular_kernel_tools.parallel_utils import _set_vllm_config
from tests.kernels.moe.utils import (
    make_dummy_moe_config,
    make_shared_experts,
    make_test_weights,
    naive_batched_moe,
)
from tests.kernels.quant_utils import dequant
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk, override_config
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_batched_moe import BatchedTritonExperts
from vllm.model_executor.layers.fused_moe.fused_moe import get_default_config
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModular
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

from ...utils import multi_gpu_test
from .parallel_utils import ProcessGroupInfo, parallel_launch

requires_pplx = pytest.mark.skipif(
    not has_pplx,
    reason="Requires PPLX kernels",
)

BATCHED_MOE_MNK_FACTORS = [
    (1, 128, 128),
    (33, 2048, 128),
    (64, 128, 2048),
    (222, 128, 128),
    (222, 2048, 1024),
]

PPLX_COMBOS = [
    # TODO(bnell): figure out why this fails, seems to be test problem
    # (1, 128, 128),
    (2, 128, 512),
    (3, 1024, 2048),
    (4, 128, 128),
    (32, 1024, 512),
    (45, 512, 2048),
    (64, 1024, 512),
    (222, 2048, 1024),
    (256, 1408, 2048),
]

NUM_EXPERTS = [8, 64]
TOP_KS = [1, 2, 6]
DTYPES = [torch.float8_e4m3fn, torch.bfloat16]

vllm_config = VllmConfig()


def torch_prepare(
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    max_num_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert topk_ids.dim() == 2
    assert topk_ids.shape[0] == a.shape[0]

    num_tokens, hidden_dim = a.shape
    topk = topk_ids.shape[1]

    tokens_per_expert = torch.bincount(topk_ids.view(-1), minlength=num_experts)

    assert tokens_per_expert.numel() == num_experts

    if max_num_tokens is None:
        max_num_tokens = int(tokens_per_expert.max().item())

    b_a = torch.zeros(
        (num_experts, max_num_tokens, hidden_dim), dtype=a.dtype, device=a.device
    )

    token_counts = torch.zeros(num_experts, dtype=torch.int, device=a.device)

    for token in range(num_tokens):
        for j in range(topk):
            expert_id = topk_ids[token, j]
            idx = token_counts[expert_id]
            b_a[expert_id, idx : idx + 1, :] = a[token, :]
            token_counts[expert_id] = token_counts[expert_id] + 1

    return b_a, tokens_per_expert


def torch_finalize(
    b_out: torch.Tensor, topk_weight: torch.Tensor, topk_ids: torch.Tensor
) -> torch.Tensor:
    num_tokens = topk_ids.shape[0]
    num_experts = b_out.shape[0]
    K = b_out.shape[-1]
    out = torch.zeros((num_tokens, K), dtype=b_out.dtype, device=b_out.device)
    expert_counts = torch.zeros(num_experts, dtype=torch.int, device=b_out.device)
    for token in range(num_tokens):
        expert_ids = topk_ids[token]
        for i in range(expert_ids.numel()):
            expert_id = expert_ids[i]
            idx = expert_counts[expert_id]
            out[token, :] = (
                out[token, :]
                + b_out[expert_id, idx : idx + 1, :] * topk_weight[token, i]
            )
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
    out = torch.zeros(
        (num_experts, max_num_tokens, K), dtype=b_a.dtype, device=b_a.device
    )
    tmp = torch.empty(
        (max_num_tokens, w1.shape[1] // 2), dtype=b_a.dtype, device=b_a.device
    )
    for expert in range(num_experts):
        num = tokens_per_expert[expert]
        if num > 0:
            torch.ops._C.silu_and_mul(
                tmp[:num], b_a[expert, :num, :] @ w1[expert].transpose(0, 1)
            )
            out[expert, :num, :] = tmp[:num] @ w2[expert].transpose(0, 1)

    return torch_finalize(out, topk_weight, topk_ids)


@pytest.mark.parametrize("m,n,k", BATCHED_MOE_MNK_FACTORS)
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
    workspace_init,
):
    set_random_seed(7)

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    with set_current_vllm_config(vllm_config):
        topk_weight, topk_ids, _ = fused_topk(a, score, topk, False)
        baseline_output = torch_experts(
            a, w1, w2, topk_weight, topk_ids
        )  # only for baseline
        torch_output = torch_batched_moe(a, w1, w2, topk_weight, topk_ids)
        batched_output = naive_batched_moe(
            a, w1, w2, topk_weight, topk_ids
        )  # pick torch_experts or this

    torch.testing.assert_close(baseline_output, torch_output, atol=2e-2, rtol=0)
    torch.testing.assert_close(baseline_output, batched_output, atol=2e-2, rtol=0)


def create_pplx_prepare_finalize(
    num_tokens: int,
    hidden_dim: int,
    topk: int,
    num_experts: int,
    rank: int,
    dp_size: int,
    world_size: int,
    in_dtype: torch.dtype,
    quant_dtype: torch.dtype | None,
    block_shape: list[int] | None,
    per_act_token_quant: bool,
    group_name: str | None,
):
    from vllm.model_executor.layers.fused_moe.pplx_prepare_finalize import (
        PplxPrepareAndFinalize,
        pplx_hidden_dim_scale_bytes,
    )

    max_num_tokens = max(rank_chunk(num_tokens, 0, world_size), 1)
    num_local_experts = rank_chunk(num_experts, 0, world_size)

    hidden_dim_bytes, scale_bytes = pplx_hidden_dim_scale_bytes(
        max_num_tokens,
        hidden_dim,
        in_dtype,
        quant_dtype,
        per_act_token_quant=per_act_token_quant,
        block_shape=block_shape,
    )

    args = dict(
        max_num_tokens=max_num_tokens,
        num_experts=num_experts,
        experts_per_token=topk,
        rank=rank,
        world_size=world_size,
        dp_size=dp_size,
        hidden_dim=hidden_dim,
        hidden_dim_bytes=hidden_dim_bytes,
        hidden_dim_scale_bytes=scale_bytes,
    )

    if group_name is None:
        ata = AllToAll.internode(**args)
    else:
        args["group_name"] = group_name
        ata = AllToAll.intranode(**args)

    prepare_finalize = PplxPrepareAndFinalize(
        ata,
        max_num_tokens=max_num_tokens,
        num_local_experts=num_local_experts,
        num_dispatchers=world_size // dp_size,
    )

    return prepare_finalize, ata


def rank_chunk(num: int, r: int, w: int) -> int:
    rem = num % w
    return (num // w) + (1 if r < rem else 0)


def chunk_by_rank(t: torch.Tensor, r: int, w: int) -> torch.Tensor:
    chunk = rank_chunk(t.shape[0], r, w)
    return t[(r * chunk) : (r + 1) * chunk]


def maybe_chunk_by_rank(t: torch.Tensor | None, r: int, w: int) -> torch.Tensor | None:
    if t is not None:
        return chunk_by_rank(t, r, w)
    else:
        return t


def chunk_scales_by_rank(t: torch.Tensor | None, r: int, w: int) -> torch.Tensor | None:
    if t is not None and t.numel() > 1:
        chunk = rank_chunk(t.shape[0], r, w)
        return t[(r * chunk) : (r + 1) * chunk]
    else:
        return t


def chunk_scales(t: torch.Tensor | None, start: int, end: int) -> torch.Tensor | None:
    if t is not None and t.numel() > 1:
        return t[start:end]
    else:
        return t


def dummy_work(a: torch.Tensor) -> torch.Tensor:
    return a * 1.1


def pplx_prepare_finalize(
    pgi: ProcessGroupInfo,
    dp_size: int,
    a: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    quant_dtype: torch.dtype | None,
    block_shape: list[int] | None,
    per_act_token_quant: bool,
    group_name: str | None,
) -> torch.Tensor:
    assert torch.cuda.current_device() == pgi.local_rank

    topk = topk_ids.shape[1]
    num_tokens, hidden_dim = a.shape
    device = pgi.device
    rank = pgi.rank
    world_size = pgi.world_size

    topk_ids = topk_ids.to(dtype=torch.uint32)

    prepare_finalize, ata = create_pplx_prepare_finalize(
        num_tokens,
        hidden_dim,
        topk,
        num_experts,
        rank,
        dp_size,
        world_size,
        a.dtype,
        quant_dtype,
        block_shape,
        per_act_token_quant,
        group_name,
    )

    assert a.shape[0] == topk_ids.shape[0]

    a_chunk = chunk_by_rank(a, rank, world_size).to(device)
    chunk_topk_weight = chunk_by_rank(topk_weight, rank, world_size).to(device)
    chunk_topk_ids = chunk_by_rank(topk_ids, rank, world_size).to(device)

    assert a_chunk.shape[0] == chunk_topk_ids.shape[0]

    out = torch.full(
        a_chunk.shape,
        torch.nan,
        dtype=a.dtype,
        device=device,
    )

    if quant_dtype is not None and not per_act_token_quant and block_shape is None:
        a1_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        a2_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    else:
        a1_scale = None
        a2_scale = None

    b_a, b_a_scale, expert_num_tokens, _, _ = prepare_finalize.prepare(
        a_chunk,
        chunk_topk_weight,
        chunk_topk_ids,
        num_experts,
        None,
        False,
        FusedMoEQuantConfig.make(
            quant_dtype,
            per_act_token_quant=per_act_token_quant,
            per_out_ch_quant=False,
            block_shape=block_shape,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
        ),
    )

    b_a = dummy_work(dequant(b_a, b_a_scale, block_shape, per_act_token_quant, a.dtype))

    prepare_finalize.finalize(
        out,
        b_a,
        chunk_topk_weight,
        chunk_topk_ids,
        False,
        weight_and_reduce_impl=TopKWeightAndReduceDelegate(),
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
    quant_dtype: torch.dtype | None,
    block_shape: list[int] | None,
    per_act_token_quant: bool,
    use_internode: bool,
):
    try:
        if use_internode:
            uid = (
                nvshmem_get_unique_id()
                if pgi.rank == 0
                else nvshmem_alloc_empty_unique_id()
            )
            torch.distributed.broadcast(uid, src=0)
            nvshmem_init(uid, pgi.rank, pgi.world_size)
            group_name = None
        else:
            group_ranks = list(range(pgi.world_size))
            cpu_group = torch.distributed.new_group(group_ranks, backend="gloo")
            group_name = cpu_group.group_name

        topk_weight, topk_ids, _ = fused_topk(a, score, topk, False)
        m, k = a.shape

        a_rep = torch.repeat_interleave(dummy_work(a), topk, dim=0)

        torch_output = (
            a_rep.view(m, topk, k) * topk_weight.view(m, topk, 1).to(a_rep.dtype)
        ).sum(dim=1)

        pplx_output = pplx_prepare_finalize(
            pgi,
            dp_size,
            a,
            topk_weight,
            topk_ids,
            num_experts,
            quant_dtype,
            block_shape,
            per_act_token_quant,
            group_name,
        )

        torch_output = chunk_by_rank(torch_output, pgi.rank, pgi.world_size).to(
            pgi.device
        )

        torch.testing.assert_close(pplx_output, torch_output, atol=3e-2, rtol=3e-2)
    finally:
        if use_internode:
            nvshmem_finalize()


@pytest.mark.parametrize("mnk", PPLX_COMBOS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("world_dp_size", [[2, 1]])
@pytest.mark.parametrize("per_act_token_quant", [False, True])
@pytest.mark.parametrize("block_shape", [None, [128, 128]])
@pytest.mark.parametrize("use_internode", [False])
@pytest.mark.optional
@requires_pplx
@multi_gpu_test(num_gpus=2)
def test_pplx_prepare_finalize_slow(
    mnk: tuple[int, int, int],
    e: int,
    topk: int,
    dtype: torch.dtype,
    world_dp_size: tuple[int, int],
    per_act_token_quant: bool,
    block_shape: list[int] | None,
    use_internode: bool,
):
    if dtype == torch.float8_e4m3fn:
        use_fp8_w8a8 = True
        act_dtype = torch.bfloat16
        quant_dtype = dtype
    else:
        use_fp8_w8a8 = False
        act_dtype = dtype
        quant_dtype = None

    if not use_fp8_w8a8 and (per_act_token_quant or block_shape is not None):
        pytest.skip("Skip quantization test for non-quantized type")

    if per_act_token_quant and block_shape is not None:
        pytest.skip("Skip illegal quantization combination")

    set_random_seed(7)
    m, n, k = mnk
    world_size, dp_size = world_dp_size
    device = "cuda"

    a = torch.randn((m, k), device=device, dtype=act_dtype) / 10
    score = torch.randn((m, e), device=device, dtype=act_dtype)

    parallel_launch(
        world_size,
        _pplx_prepare_finalize,
        dp_size,
        a,
        score,
        topk,
        e,
        quant_dtype,
        block_shape,
        per_act_token_quant,
        use_internode,
    )


def pplx_moe(
    group_name: str | None,
    rank: int,
    world_size: int,
    dp_size: int,
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    quant_dtype: torch.dtype | None = None,
    per_act_token_quant=False,
    block_shape: list[int] | None = None,
    use_compile: bool = False,
    use_cudagraphs: bool = True,
    shared_experts: torch.nn.Module | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    num_tokens, hidden_dim = a.shape
    num_experts = w1.shape[0]
    topk = topk_ids.shape[1]
    max_num_tokens = round_up(rank_chunk(a.shape[0], 0, world_size), 16)

    prepare_finalize, ata = create_pplx_prepare_finalize(
        num_tokens,
        hidden_dim,
        topk,
        num_experts,
        rank,
        dp_size,
        world_size,
        a.dtype,
        quant_dtype,
        block_shape,
        per_act_token_quant,
        group_name,
    )

    topk_ids = topk_ids.to(dtype=torch.uint32)

    # Note: workers with the same dp_rank must use the exact same inputs.
    a_chunk = chunk_by_rank(a, rank, world_size)
    chunk_topk_weight = chunk_by_rank(topk_weight, rank, world_size)
    chunk_topk_ids = chunk_by_rank(topk_ids, rank, world_size)

    # Chunking weights like this only works for batched format
    w1_chunk = chunk_by_rank(w1, rank, world_size)
    w2_chunk = chunk_by_rank(w2, rank, world_size)
    w1_scale_chunk = maybe_chunk_by_rank(w1_scale, rank, world_size)
    w2_scale_chunk = maybe_chunk_by_rank(w2_scale, rank, world_size)
    a1_scale_chunk = chunk_scales_by_rank(a1_scale, rank, world_size)
    a2_scale_chunk = chunk_scales_by_rank(a2_scale, rank, world_size)

    quant_config = FusedMoEQuantConfig.make(
        quant_dtype,
        block_shape=block_shape,
        per_act_token_quant=per_act_token_quant,
        w1_scale=w1_scale_chunk,
        w2_scale=w2_scale_chunk,
        a1_scale=a1_scale_chunk,
        a2_scale=a2_scale_chunk,
    )

    experts = BatchedTritonExperts(
        max_num_tokens=max_num_tokens,
        num_dispatchers=prepare_finalize.num_dispatchers(),
        quant_config=quant_config,
        moe_config=make_dummy_moe_config(),
    )

    fused_experts = FusedMoEKernelModular(
        prepare_finalize,
        experts,
        shared_experts,
    )

    # Note: for now use_compile will error out if the problem size is
    # large enough to trigger chunking. I'm leaving the flag and
    # setup code in case we are able to revisit this later.
    if use_compile:
        _fused_experts = torch.compile(
            fused_experts, backend="inductor", fullgraph=True
        )
        torch._dynamo.mark_dynamic(a_chunk, 0)
        torch._dynamo.mark_dynamic(chunk_topk_weight, 0)
        torch._dynamo.mark_dynamic(chunk_topk_ids, 0)
    else:
        _fused_experts = fused_experts

    out = _fused_experts(
        a_chunk,
        w1_chunk,
        w2_chunk,
        chunk_topk_weight,
        chunk_topk_ids,
        global_num_experts=num_experts,
    )

    if use_cudagraphs:
        if isinstance(out, tuple):
            out[0].fill_(0)
            out[1].fill_(0)
        else:
            out.fill_(0)
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = _fused_experts(
                a_chunk,
                w1_chunk,
                w2_chunk,
                chunk_topk_weight,
                chunk_topk_ids,
                global_num_experts=num_experts,
            )

        torch.cuda.synchronize()
        graph.replay()

    torch.cuda.synchronize()

    ata.destroy()

    return out


def _pplx_moe(
    pgi: ProcessGroupInfo,
    dp_size: int,
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    topk: int,
    num_experts: int,
    w1_s: torch.Tensor | None = None,
    w2_s: torch.Tensor | None = None,
    quant_dtype: torch.dtype | None = None,
    per_act_token_quant: bool = False,
    block_shape: list[int] | None = None,
    use_internode: bool = False,
    shared_experts: torch.nn.Module | None = None,
):
    try:
        if use_internode:
            uid = (
                nvshmem_get_unique_id()
                if pgi.rank == 0
                else nvshmem_alloc_empty_unique_id()
            )
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

        device = torch.device("cuda", pgi.rank)
        rank = pgi.rank
        world_size = pgi.world_size

        a = a.to(device)
        w1 = w1.to(device)
        w2 = w2.to(device)
        w1_s = w1_s.to(device) if w1_s is not None else None
        w2_s = w2_s.to(device) if w2_s is not None else None

        if quant_dtype is not None and not per_act_token_quant and block_shape is None:
            a1_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
            a2_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        else:
            a1_scale = None
            a2_scale = None

        with set_current_vllm_config(vllm_config), override_config(moe_config):
            topk_weight, topk_ids, _ = fused_topk(a, score, topk, False)

            shared_output = shared_experts(a) if shared_experts is not None else None

            torch_output = torch_experts(
                a,
                w1,
                w2,
                topk_weight,
                topk_ids,
                w1_scale=w1_s,
                w2_scale=w2_s,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                quant_dtype=quant_dtype,
                per_act_token_quant=per_act_token_quant,
                block_shape=block_shape,
            )

            batched_output = naive_batched_moe(
                a,
                w1,
                w2,
                topk_weight,
                topk_ids,
                w1_scale=w1_s,
                w2_scale=w2_s,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                quant_dtype=quant_dtype,
                per_act_token_quant=per_act_token_quant,
                block_shape=block_shape,
            )

            pplx_outputs = pplx_moe(
                group_name,
                rank,
                world_size,
                dp_size,
                a,
                w1,
                w2,
                topk_weight,
                topk_ids,
                w1_scale=w1_s,
                w2_scale=w2_s,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                quant_dtype=quant_dtype,
                per_act_token_quant=per_act_token_quant,
                block_shape=block_shape,
                shared_experts=shared_experts,
            )

        if shared_experts is None:
            pplx_shared_output = None
            pplx_output = pplx_outputs
            assert isinstance(pplx_output, torch.Tensor)
        else:
            pplx_shared_output, pplx_output = pplx_outputs

        if shared_output is not None:
            assert pplx_shared_output is not None
            chunked_shared_output = chunk_by_rank(
                shared_output, pgi.rank, pgi.world_size
            ).to(pplx_shared_output.device)
        else:
            chunked_shared_output = None

        chunked_batch_output = chunk_by_rank(
            batched_output, pgi.rank, pgi.world_size
        ).to(pplx_output.device)

        torch.testing.assert_close(batched_output, torch_output, atol=3e-2, rtol=3e-2)

        torch.testing.assert_close(
            pplx_output, chunked_batch_output, atol=3e-2, rtol=3e-2
        )

        if shared_experts is not None:
            assert chunked_shared_output is not None
            assert pplx_shared_output is not None
            torch.testing.assert_close(
                pplx_shared_output, chunked_shared_output, atol=3e-2, rtol=3e-2
            )

    finally:
        if use_internode:
            nvshmem_finalize()


@pytest.mark.parametrize("mnk", PPLX_COMBOS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("world_dp_size", [[2, 1]])
@pytest.mark.parametrize("per_act_token_quant", [False, True])
@pytest.mark.parametrize("block_shape", [None, [128, 128]])
@pytest.mark.parametrize("use_internode", [False])
@pytest.mark.optional
@requires_pplx
@multi_gpu_test(num_gpus=2)
def test_pplx_moe_slow(
    mnk: tuple[int, int, int],
    e: int,
    topk: int,
    dtype: torch.dtype,
    world_dp_size: tuple[int, int],
    per_act_token_quant: bool,
    block_shape: list[int] | None,
    use_internode: bool,
):
    set_random_seed(7)
    m, n, k = mnk
    world_size, dp_size = world_dp_size

    if dtype == torch.float8_e4m3fn:
        use_fp8_w8a8 = True
        quant_dtype = dtype
    else:
        use_fp8_w8a8 = False
        quant_dtype = None

    if not use_fp8_w8a8 and (per_act_token_quant or block_shape is not None):
        pytest.skip("Skip quantization test for non-quantized type")

    if per_act_token_quant and block_shape is not None:
        pytest.skip("Skip illegal quantization combination")

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
    score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)

    (_, w1, w1_s, _), (_, w2, w2_s, _) = make_test_weights(
        e,
        n,
        k,
        quant_dtype=quant_dtype,
        block_shape=block_shape,
        per_out_ch_quant=per_act_token_quant,
    )

    parallel_launch(
        world_size,
        _pplx_moe,
        dp_size,
        a,
        w1,
        w2,
        score,
        topk,
        e,
        w1_s,
        w2_s,
        quant_dtype,
        per_act_token_quant,
        block_shape,
        use_internode,
    )


def _pplx_test_loop(
    pgi: ProcessGroupInfo,
    dp_size: int,
    use_internode: bool,
    use_shared_experts: bool,
    make_weights: bool,
    test_fn: Callable,
):
    device = torch.device(f"cuda:{pgi.local_rank}")
    init_workspace_manager(device)

    def format_result(msg, ex=None):
        if ex is not None:
            x = str(ex)
            newx = x.strip(" \n\t")[:16]
            if len(newx) < len(x):
                newx = newx + " ..."

            prefix = "E\t"
            print(f"{textwrap.indent(traceback.format_exc(), prefix)}")
            print(f"FAILED {msg} - {newx}\n")
        else:
            print(f"PASSED {msg}")

    if use_shared_experts:
        # Note: this config is only needed for the non-naive shared experts.
        new_vllm_config = copy.deepcopy(vllm_config)
        new_vllm_config.parallel_config.data_parallel_size = pgi.world_size
        new_vllm_config.parallel_config.enable_expert_parallel = True
        _set_vllm_config(new_vllm_config, pgi.world_size, pgi.rank, pgi.local_rank)

    set_random_seed(7)
    combos = itertools.product(
        PPLX_COMBOS, NUM_EXPERTS, TOP_KS, DTYPES, [False, True], [None, [128, 128]]
    )
    exceptions = []
    count = 0
    for mnk, e, topk, dtype, per_act_token_quant, block_shape in combos:
        count = count + 1
        m, n, k = mnk

        if dtype == torch.float8_e4m3fn:
            use_fp8_w8a8 = True
            quant_dtype = dtype
        else:
            use_fp8_w8a8 = False
            quant_dtype = None

        test_desc = (
            f"test_pplx_moe[mnk={mnk}, e={e}, topk={topk}, "
            f"dtype={dtype}, per_act_token={per_act_token_quant}, "
            f"block_shape={block_shape}, use_internode={use_internode}, "
            f"use_shared_experts={use_shared_experts}"
        )

        if not use_fp8_w8a8 and (per_act_token_quant or block_shape is not None):
            print(f"{test_desc} - Skip quantization test for non-quantized type.")
            continue

        if per_act_token_quant and block_shape is not None:
            print(f"{test_desc} - Skip illegal quantization combination.")
            continue

        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
        score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)

        args = dict()
        if make_weights:
            (_, w1, w1_s, _), (_, w2, w2_s, _) = make_test_weights(
                e,
                n,
                k,
                quant_dtype=quant_dtype,
                block_shape=block_shape,
                per_out_ch_quant=per_act_token_quant,
            )
            args["w1"] = w1
            args["w2"] = w2
            args["w1_s"] = w1_s
            args["w2_s"] = w2_s

        if use_shared_experts:
            args["shared_experts"] = make_shared_experts(
                n,
                k,
                in_dtype=a.dtype,
                quant_dtype=quant_dtype,
            )

        try:
            test_fn(
                pgi=pgi,
                dp_size=dp_size,
                a=a,
                score=score,
                topk=topk,
                num_experts=e,
                quant_dtype=quant_dtype,
                per_act_token_quant=per_act_token_quant,
                block_shape=block_shape,
                use_internode=use_internode,
                **args,
            )
            format_result(test_desc)
        except Exception as ex:
            format_result(test_desc, ex)
            exceptions.append(ex)

    if len(exceptions) > 0:
        raise RuntimeError(
            f"{len(exceptions)} of {count} tests failed in child process, "
            f"rank={pgi.rank}."
        )
    else:
        print(f"{count} of {count} tests passed in child process, rank={pgi.rank}.")


@pytest.mark.parametrize("world_dp_size", [[2, 1]])
@pytest.mark.parametrize("use_internode", [False])
@requires_pplx
@multi_gpu_test(num_gpus=2)
def test_pplx_prepare_finalize(
    world_dp_size: tuple[int, int],
    use_internode: bool,
):
    set_random_seed(7)
    world_size, dp_size = world_dp_size
    parallel_launch(
        world_size * dp_size,
        _pplx_test_loop,
        dp_size,
        use_internode,
        False,
        False,
        _pplx_prepare_finalize,
    )


@pytest.mark.parametrize("world_dp_size", [[2, 1]])
@pytest.mark.parametrize("use_internode", [False])
@pytest.mark.parametrize("use_shared_experts", [False, True])
@requires_pplx
@multi_gpu_test(num_gpus=2)
def test_pplx_moe(
    world_dp_size: tuple[int, int],
    use_internode: bool,
    use_shared_experts: bool,
):
    set_random_seed(7)
    world_size, dp_size = world_dp_size
    parallel_launch(
        world_size,
        _pplx_test_loop,
        dp_size,
        use_internode,
        use_shared_experts,
        True,
        _pplx_moe,
    )
