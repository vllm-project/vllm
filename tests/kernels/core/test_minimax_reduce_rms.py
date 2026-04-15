# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MiniMax QK RMS-norm: NCCL reference vs Lamport fused kernel."""

import pytest
import torch
import torch.nn as nn
from torch.multiprocessing import spawn

from tests.kernels.utils import opcheck
from tests.utils import ensure_current_vllm_config, init_test_distributed_environment
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.layers.mamba.linear_attn import MiniMaxText01RMSNormTP
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_random_seed


@ensure_current_vllm_config()
def _worker_forward_qk(
    local_rank,
    world_size,
    port,
    num_tokens,
    hidden_q_full,
    hidden_k_full,
    dtype,
    seed,
    eps,
):
    """Per-rank worker: compare NCCL allreduce path vs Lamport fused kernel."""

    if not hasattr(torch.ops._C, "minimax_allreduce_rms_qk"):
        cleanup_dist_env_and_memory()
        return
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        world_size, 1, local_rank, port, local_rank=local_rank
    )

    hq = hidden_q_full // world_size
    hk = hidden_k_full // world_size

    q_norm = MiniMaxText01RMSNormTP(hidden_q_full, eps=eps).cuda()
    k_norm = MiniMaxText01RMSNormTP(hidden_k_full, eps=eps).cuda()

    set_random_seed(seed)
    qw = torch.randn(hidden_q_full, dtype=dtype, device="cuda")
    kw = torch.randn(hidden_k_full, dtype=dtype, device="cuda")
    q_norm.weight = nn.Parameter(qw[local_rank * hq : (local_rank + 1) * hq])
    k_norm.weight = nn.Parameter(kw[local_rank * hk : (local_rank + 1) * hk])

    torch.manual_seed(seed + 1000 + local_rank)
    qkv = torch.randn(num_tokens, hq + hk + hk, dtype=dtype, device="cuda")

    q_ref, k_ref, v_ref = qkv.clone().split([hq, hk, hk], dim=-1)
    ref_q, ref_k = MiniMaxText01RMSNormTP.forward_qk(q_norm, k_norm, q_ref, k_ref)

    # Set up Lamport workspace.
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.model_executor.layers.mamba.lamport_workspace import (
        get_allreduce_workspace,
    )

    workspace = get_allreduce_workspace(
        rank=local_rank,
        world_size=world_size,
        max_tokens=num_tokens,
        process_group=get_tp_group().cpu_group,
    )

    opcheck(
        torch.ops._C.minimax_allreduce_rms_qk,
        (
            qkv.clone(),
            q_norm.weight,
            k_norm.weight,
            workspace,
            hq,
            hk,
            local_rank,
            world_size,
            eps,
        ),
    )
    fused_q, fused_k = torch.ops._C.minimax_allreduce_rms_qk(
        qkv.clone(),
        q_norm.weight,
        k_norm.weight,
        workspace,
        hq,
        hk,
        local_rank,
        world_size,
        eps,
    )
    _, _, fused_v = qkv.split([hq, hk, hk], dim=-1)
    torch.accelerator.synchronize()

    torch.testing.assert_close(
        fused_q,
        ref_q,
        atol=3e-2,
        rtol=3e-2,
    )
    torch.testing.assert_close(fused_k, ref_k, atol=3e-2, rtol=3e-2)

    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA required",
)
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("num_tokens", [1, 128, 333])
@pytest.mark.parametrize(
    "hidden_dims",
    [(6144, 1024)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("seed", [42])
def test_minimax_reduce_rms_qk(
    world_size,
    num_tokens,
    hidden_dims,
    dtype,
    eps,
    seed,
):
    num_gpus = current_platform.device_count()
    if num_gpus < world_size:
        pytest.skip(f"Need >= {world_size} GPUs, have {num_gpus}")
    hidden_q_full, hidden_k_full = hidden_dims
    port = str(get_open_port())
    spawn(
        _worker_forward_qk,
        args=(
            world_size,
            port,
            num_tokens,
            hidden_q_full,
            hidden_k_full,
            dtype,
            seed,
            eps,
        ),
        nprocs=world_size,
        join=True,
    )
