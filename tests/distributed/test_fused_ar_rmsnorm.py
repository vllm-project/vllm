# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-GPU tests for fused allreduce+RMSNorm communication op.

Run: pytest tests/distributed/test_fused_ar_rmsnorm.py
"""

import pytest
import ray
import torch

from vllm._aiter_ops import IS_AITER_FOUND
from vllm.platforms import current_platform

from ..utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)


@ray.remote(num_gpus=1, max_calls=1)
def fused_allreduce_rmsnorm_test_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    """Test that fused_allreduce_rmsnorm produces correct results.

    Compares the fused path (allreduce + add + rmsnorm in one call)
    against the split path (manual allreduce, then add, then rmsnorm).
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        tp_size, pp_size, rank, distributed_init_port
    )

    from vllm.distributed import get_tp_group

    tp_group = get_tp_group()

    hidden_size = 256
    num_tokens = 64
    eps = 1e-5

    torch.manual_seed(42 + rank)
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)

    for _ in range(3):
        input_ = torch.randn(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=device
        )
        residual = torch.randn(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=device
        )

        input_ref = input_.clone()
        residual_ref = residual.clone()

        normed, resid_out = tp_group._fused_allreduce_rmsnorm_out_place(
            input_, residual, weight, eps
        )

        ar_ref = tp_group.all_reduce(input_ref)
        combined = ar_ref + residual_ref
        variance = combined.pow(2).mean(-1, keepdim=True)
        normed_ref = combined * torch.rsqrt(variance + eps)
        normed_ref = normed_ref * weight

        torch.testing.assert_close(
            resid_out, combined, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            normed, normed_ref, atol=1e-2, rtol=1e-2
        )


@ray.remote(num_gpus=1, max_calls=1)
def fused_allreduce_rmsnorm_world_size_1_test_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    """Test world_size==1 fallback (add + rmsnorm, no allreduce)."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        tp_size, pp_size, rank, distributed_init_port
    )

    from vllm.distributed import get_tp_group

    tp_group = get_tp_group()
    assert tp_group.world_size == 1

    hidden_size = 128
    num_tokens = 16
    eps = 1e-5

    torch.manual_seed(42)
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)
    input_ = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    residual = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )

    normed, resid_out = tp_group.fused_allreduce_rmsnorm(
        input_, residual, weight, eps
    )

    combined = input_ + residual
    variance = combined.pow(2).mean(-1, keepdim=True)
    normed_ref = combined * torch.rsqrt(variance + eps)
    normed_ref = normed_ref * weight

    torch.testing.assert_close(resid_out, combined, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(normed, normed_ref, atol=1e-2, rtol=1e-2)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.skipif(
    not current_platform.is_rocm() or not IS_AITER_FOUND,
    reason="ROCm with AITER required",
)
def test_fused_allreduce_rmsnorm(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
):
    multi_process_parallel(
        monkeypatch,
        tp_size,
        1,
        fused_allreduce_rmsnorm_test_worker,
    )


@multi_gpu_test(num_gpus=1)
@pytest.mark.skipif(
    not current_platform.is_rocm() or not IS_AITER_FOUND,
    reason="ROCm with AITER required",
)
def test_fused_allreduce_rmsnorm_world_size_1(
    monkeypatch: pytest.MonkeyPatch,
):
    multi_process_parallel(
        monkeypatch,
        1,
        1,
        fused_allreduce_rmsnorm_world_size_1_test_worker,
    )
