# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import pytest
import torch

from tests.kernels.utils import torch_experts
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import CutlassExpertsFp8
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.platforms import current_platform

from .deepep_utils import ProcessGroupInfo, parallel_launch

try:
    from pplx_kernels import AllToAll
    from pplx_kernels.nvshmem import (nvshmem_alloc_empty_unique_id,
                                      nvshmem_finalize, nvshmem_get_unique_id,
                                      nvshmem_init)
    has_pplx = True
except ImportError:
    has_pplx = False

requires_pplx = pytest.mark.skipif(
    not has_pplx,
    reason="Requires PPLX kernels",
)

NUM_EXPERTS = [40, 64]
TOP_KS = [6, 8]


def rank_chunk(num, r, w):
    rem = num % w
    return (num // w) + (1 if r < rem else 0)


def chunk_by_rank(t, r, w):
    num = t.shape[0]
    chunk = rank_chunk(num, r, w)
    rem = num % w
    if rem == 0 or r < rem:
        return t[(r * chunk):(r + 1) * chunk].contiguous()
    else:
        long_chunks = (num // w + 1) * rem
        short_chunks = (r - rem) * chunk
        start = long_chunks + short_chunks
        return t[start:start + chunk].contiguous()


def pplx_cutlass_moe(
    pgi: ProcessGroupInfo,
    dp_size: int,
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_scale: torch.Tensor,
    out_dtype,
    per_act_token: bool,
    per_out_ch: bool,
    group_name: Optional[str],
):
    from vllm.model_executor.layers.fused_moe.pplx_prepare_finalize import (
        PplxPrepareAndFinalize)
    assert torch.cuda.current_device() == pgi.local_rank

    num_tokens, hidden_dim = a.shape
    num_experts = w1.shape[0]
    block_size = hidden_dim  # TODO support more cases
    device = pgi.device
    rank = pgi.rank
    world_size = pgi.world_size
    rank_num_tokens = rank_chunk(num_tokens, rank, world_size)
    max_num_tokens = rank_chunk(num_tokens, 0, world_size)
    topk = topk_ids.shape[1]

    if block_size == hidden_dim:
        scale_elems = 4  # hack to circumvent pplx data format requirements
    else:
        scale_elems = (hidden_dim + block_size - 1) // block_size

    args = dict(
        max_num_tokens=max_num_tokens,
        num_experts=num_experts,
        experts_per_token=topk,
        rank=rank,
        world_size=pgi.world_size,
        dp_size=dp_size,
        hidden_dim=hidden_dim,
        hidden_dim_bytes=hidden_dim,  # because a.dtype.itemsize == 1
        hidden_dim_scale_bytes=scale_elems * torch.float32.itemsize,
    )

    if group_name is None:
        ata = AllToAll.internode(**args)
    else:
        args["group_name"] = group_name
        ata = AllToAll.intranode(**args)

    w1 = w1.to(device)
    w2 = w2.to(device)
    w1_scale = w1_scale.to(device)
    w2_scale = w2_scale.to(device)
    a1_scale = a1_scale.to(device)

    prepare_finalize = PplxPrepareAndFinalize(
        ata,
        max_num_tokens,
        pgi.world_size,
        rank,
        dp_size,
        quant_dtype=torch.float8_e4m3fn,
        per_act_token=per_act_token,
    )

    experts = CutlassExpertsFp8((num_experts + world_size - 1) // world_size,
                                out_dtype,
                                per_act_token,
                                per_out_ch,
                                use_batched_format=True)

    fused_cutlass_experts = FusedMoEModularKernel(
        prepare_finalize,
        experts,
    )

    a_chunk = chunk_by_rank(a, rank, world_size).to(device)
    chunk_topk_weight = chunk_by_rank(topk_weights, rank,
                                      world_size).to(device)
    chunk_topk_ids = chunk_by_rank(topk_ids, rank,
                                   world_size).to(torch.uint32).to(device)

    out = fused_cutlass_experts(
        a_chunk,
        chunk_by_rank(w1, rank, world_size),
        chunk_by_rank(w2, rank, world_size),
        chunk_topk_weight,
        chunk_topk_ids,
        global_num_experts=num_experts,
        expert_map=None,  #TODO
        w1_scale=chunk_by_rank(w1_scale, rank, world_size),
        w2_scale=chunk_by_rank(w2_scale, rank, world_size),
        a1_scale=chunk_by_rank(a1_scale, rank, world_size)
        if per_act_token else a1_scale[rank])

    torch.cuda.synchronize()

    ata.destroy()

    return out[:rank_num_tokens]


vllm_config = VllmConfig()
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192


def _pplx_moe(
    pgi: ProcessGroupInfo,
    dp_size: int,
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_scale: torch.Tensor,
    out_dtype,
    a_full: torch.Tensor,
    w1_full: torch.Tensor,
    w2_full: torch.Tensor,
    per_act_token: bool,
    per_out_ch: bool,
    use_internode: bool,
):
    if use_internode:
        uid = nvshmem_get_unique_id(
        ) if pgi.rank == 0 else nvshmem_alloc_empty_unique_id()
        torch.distributed.broadcast(uid, src=0)
        nvshmem_init(uid, pgi.rank, pgi.world_size)
    else:
        group_ranks = list(range(pgi.world_size))
        cpu_group = torch.distributed.new_group(group_ranks, backend="gloo")
        group_name = cpu_group.group_name

    with set_current_vllm_config(vllm_config):
        torch_output = torch_experts(a_full, w1_full, w2_full, topk_weights,
                                     topk_ids)
        pplx_output = pplx_cutlass_moe(pgi, dp_size, a, w1, w2, w1_scale,
                                       w2_scale, topk_weights, topk_ids,
                                       a1_scale, out_dtype, per_act_token,
                                       per_out_ch, group_name)

        torch_output = chunk_by_rank(torch_output, pgi.rank,
                                     pgi.world_size).to(pplx_output.device)

    # Uncomment if more debugging is needed
    # print("PPLX OUT:", pplx_output)
    # print("TORCH OUT:", torch_output)

    torch.testing.assert_close(pplx_output, torch_output, atol=0.05, rtol=0)

    if use_internode:
        nvshmem_finalize()


@pytest.mark.parametrize("m", [2, 224])
@pytest.mark.parametrize("n", [3072])
@pytest.mark.parametrize("k", [1536])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("world_dp_size", [[2, 1]])  #, [4, 2]])
@pytest.mark.parametrize("use_internode", [False])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()),
    reason="Grouped gemm is not supported on this GPU type.")
@requires_pplx
def test_cutlass_moe_pplx(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
    world_dp_size: tuple[int, int],
    use_internode: bool,
):
    current_platform.seed_everything(7)

    with set_current_vllm_config(vllm_config):

        dtype = torch.half

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10.0
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10.0
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10.0

        n_b_scales = 2 * n if per_out_ch else 1
        k_b_scales = k if per_out_ch else 1

        w1_q = torch.empty((e, 2 * n, k),
                           device="cuda",
                           dtype=torch.float8_e4m3fn)
        w2_q = torch.empty((e, k, n), device="cuda", dtype=torch.float8_e4m3fn)
        w1_scale = torch.empty((e, n_b_scales, 1),
                               device="cuda",
                               dtype=torch.float32)
        w2_scale = torch.empty((e, k_b_scales, 1),
                               device="cuda",
                               dtype=torch.float32)

        for expert in range(e):
            w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(
                w1[expert], use_per_token_if_dynamic=per_out_ch)
            w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(
                w2[expert], use_per_token_if_dynamic=per_out_ch)

        w1_d = torch.empty_like(w1)
        w2_d = torch.empty_like(w2)
        for expert in range(e):
            w1_d[expert] = (w1_q[expert].float() * w1_scale[expert]).half()
            w2_d[expert] = (w2_q[expert].float() * w2_scale[expert]).half()

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a,
                                               score,
                                               topk,
                                               renormalize=False)

        world_size, dp_size = world_dp_size
        a_scale1 = torch.randn(
            (m if per_act_token else 1, 1), device="cuda",
            dtype=torch.float32) / 10.0
        if not per_act_token:
            a_scale1 = a_scale1.repeat(world_size, 1)

        parallel_launch(world_size, _pplx_moe, dp_size, a, w1_q, w2_q,
                        w1_scale, w2_scale, topk_weights, topk_ids, a_scale1,
                        dtype, a, w1_d, w2_d, per_act_token, per_out_ch,
                        use_internode)
