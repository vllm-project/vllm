# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for eager AllReduce + RMSNorm fusion.

``fused_allreduce_rms_norm`` must match ``RMSNorm(all_reduce(partial), residual)``.
``fused_allreduce_rms_norm_out`` must match ``RMSNorm(all_reduce(partial))``.
Both the flashinfer/AITER fast path and the unfused fallback are covered by
comparing against the explicit all-reduce + RMSNorm path.
"""

import pytest
import torch
from torch.multiprocessing import spawn

from tests.utils import ensure_current_vllm_config, init_test_distributed_environment
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.common.ops.fused_allreduce_rms_norm import (
    fused_allreduce_rms_norm,
    fused_allreduce_rms_norm_out,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_random_seed


@ensure_current_vllm_config()
def _worker_fused_ar_rms(
    local_rank,
    world_size,
    port,
    num_tokens,
    hidden_size,
    dtype,
    seed,
    eps,
    with_residual,
):
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        world_size, 1, local_rank, port, local_rank=local_rank
    )

    set_random_seed(seed)
    norm = RMSNorm(hidden_size, eps=eps).cuda().to(dtype)
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.1)

    torch.manual_seed(seed + 1000 + local_rank)
    partial = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    reduced = tensor_model_parallel_all_reduce(partial.clone())
    if with_residual:
        torch.manual_seed(seed + 7)
        residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        ref_out, ref_res = norm(reduced, residual.clone())
        out, res = fused_allreduce_rms_norm(partial.clone(), residual.clone(), norm)
        torch.accelerator.synchronize()
        torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(res, ref_res, atol=2e-2, rtol=2e-2)
    else:
        ref_out = norm(reduced)
        out = fused_allreduce_rms_norm_out(partial.clone(), norm)
        torch.accelerator.synchronize()
        torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)

    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_cuda() and not current_platform.is_rocm(),
    reason="CUDA or ROCm required",
)
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("num_tokens", [1, 4, 64, 256])
@pytest.mark.parametrize("hidden_size", [512, 2048])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("with_residual", [True, False])
def test_fused_allreduce_rms_norm(
    world_size,
    num_tokens,
    hidden_size,
    dtype,
    eps,
    seed,
    with_residual,
):
    num_gpus = current_platform.device_count()
    if num_gpus < world_size:
        pytest.skip(f"Need >= {world_size} GPUs, have {num_gpus}")
    port = str(get_open_port())
    spawn(
        _worker_fused_ar_rms,
        args=(
            world_size,
            port,
            num_tokens,
            hidden_size,
            dtype,
            seed,
            eps,
            with_residual,
        ),
        nprocs=world_size,
        join=True,
    )
