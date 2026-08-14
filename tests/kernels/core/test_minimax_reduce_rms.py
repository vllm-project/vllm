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
from vllm.model_executor.layers.minimax_rms_norm import (
    MiniMaxText01RMSNormTP,
    rms_norm_tp,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
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

    # Reference: eager all-reduce path. ``forward_qk`` no longer all-reduces
    # the variance (it is the tp==1 / already-reduced building block), so the
    # multi-rank reference must use the eager path that performs the global
    # variance all-reduce, matching the fused kernel below.
    ref_q, ref_k = rms_norm_tp._minimax_qk_norm_tp_eager(
        qkv.clone(),
        q_norm.weight,
        k_norm.weight,
        hq,
        hk,
        world_size,
        eps,
    )

    # Set up Lamport workspace.
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.model_executor.layers.minimax_rms_norm.lamport_workspace import (
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


@pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="CUDA and Triton required",
)
@pytest.mark.parametrize("num_tokens", [1, 7, 128, 333, 2049])
@pytest.mark.parametrize("hidden_dims", [(3072, 512), (768, 256), (3000, 500)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tp_world", [1, 4, 8])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("seed", [42])
def test_minimax_qk_norm_triton_fallback(
    monkeypatch, num_tokens, hidden_dims, dtype, tp_world, eps, seed
):
    """Single-GPU check: Triton fallback kernels vs the pure-torch reference.

    The all-reduce is a TP communication barrier, so it is monkeypatched to
    identity here; both the Triton path and the reference see the same
    (patched) reduction. This validates the kernel math and the folded
    ``/ tp_world`` scaling without needing multiple ranks -- ``hidden_dims``
    are the per-rank q/k segment widths.
    """
    monkeypatch.setattr(rms_norm_tp, "_all_reduce_variance", lambda v: v)

    q_size, kv_size = hidden_dims
    device = "cuda"
    torch.manual_seed(seed)
    qkv = torch.randn(num_tokens, q_size + 2 * kv_size, dtype=dtype, device=device)
    q_weight = torch.randn(q_size, dtype=dtype, device=device)
    k_weight = torch.randn(kv_size, dtype=dtype, device=device)

    q_triton, k_triton = rms_norm_tp._minimax_qk_norm_tp_fallback(
        qkv, q_weight, k_weight, q_size, kv_size, 0, tp_world, eps
    )
    q_ref, k_ref = rms_norm_tp._minimax_qk_norm_tp_eager(
        qkv, q_weight, k_weight, q_size, kv_size, tp_world, eps
    )

    torch.testing.assert_close(q_triton, q_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(k_triton, k_ref, atol=3e-2, rtol=3e-2)
