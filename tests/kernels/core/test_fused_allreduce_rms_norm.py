# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests ``fused_allreduce_rms_norm`` on a MoE output whose top-k reduction is
still open (``MoEOutput`` holding an ``UnfinalizedMoEOutput``).

It must match the unfused path: finalize the routed output, add the shared
expert output, then ``RMSNorm(all_reduce(.), residual)``. Covers the FlashInfer
MoE-finalize all-reduce kernel (TP>1) and the eager fallback.
"""

import pytest
import torch
from torch.multiprocessing import spawn

from tests.utils import ensure_current_vllm_config, init_test_distributed_environment
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.common.ops import fused_allreduce_rms_norm as fused_ar_norm
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_random_seed

TOP_K = 8


@ensure_current_vllm_config()
def _worker_moe_finalize_ar_norm(
    local_rank, world_size, port, num_tokens, hidden_size, fused, seed
):
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        world_size, 1, local_rank, port, local_rank=local_rank
    )
    if fused:
        import flashinfer.comm as flashinfer_comm

        kernel = flashinfer_comm.trtllm_moe_finalize_allreduce_fusion
        calls = []

        def counted_kernel(**kwargs):
            calls.append(kwargs)
            kernel(**kwargs)

        flashinfer_comm.trtllm_moe_finalize_allreduce_fusion = counted_kernel
    else:
        fused_ar_norm.flashinfer_trtllm_fused_allreduce_norm = None
    dtype = torch.bfloat16

    set_random_seed(seed)
    norm = RMSNorm(hidden_size, eps=1e-5).cuda().to(dtype)
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.1)
    residual = 4 * torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # Per-rank partial MoE outputs: GEMM2 rows are permuted into a padded
    # buffer, and the permute map sends each (token, k) to a distinct row.
    torch.manual_seed(seed + 1000 + local_rank)
    num_rows = 2 * num_tokens * TOP_K + 16
    gemm2 = torch.randn(num_rows, hidden_size, dtype=dtype, device=device)
    permuted_idx = (
        torch.randperm(num_rows, device=device)[: num_tokens * TOP_K]
        .to(torch.int32)
        .view(num_tokens, TOP_K)
    )
    weights = torch.rand(num_tokens, TOP_K, device=device).to(dtype)
    shared = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # Reference: the unfused path (FP32 top-k sum, shared add, AR, RMSNorm).
    routed = (gemm2[permuted_idx.long()].float() * weights.float()[..., None]).sum(1)
    partial = shared + routed.to(dtype)
    ref_out, ref_res = norm(tensor_model_parallel_all_reduce(partial), residual.clone())

    moe_output = MoEOutput(
        routed=UnfinalizedMoEOutput(
            gemm2_permuted=gemm2,
            expert_weights=weights,
            expanded_idx_to_permuted_idx=permuted_idx,
        ),
        shared_output=shared,
    )
    out, res = fused_ar_norm.fused_allreduce_rms_norm(
        moe_output, residual.clone(), norm
    )
    torch.accelerator.synchronize()
    if fused:
        assert len(calls) == 1

    # The fused kernel sums the top-k and the ranks in BF16.
    torch.testing.assert_close(res, ref_res, atol=0.25, rtol=2e-2)
    torch.testing.assert_close(out, ref_out, atol=6e-2, rtol=2e-2)
    cleanup_dist_env_and_memory()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("num_tokens", [1, 4, 64])
@pytest.mark.parametrize("hidden_size", [2048, 6144])
@pytest.mark.parametrize("fused", [True, False])
def test_fused_allreduce_rms_norm_moe_output(
    world_size, num_tokens, hidden_size, fused
):
    if world_size == 1 and fused:
        pytest.skip("TP==1 has no fused path")
    if current_platform.device_count() < world_size:
        pytest.skip(f"Need >= {world_size} GPUs")
    spawn(
        _worker_moe_finalize_ar_norm,
        args=(world_size, str(get_open_port()), num_tokens, hidden_size, fused, 0),
        nprocs=world_size,
        join=True,
    )
