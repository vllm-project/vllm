# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD projection/collective correctness corresponding to Kimi GEMM-RS/AR.

Kimi's AMD output projection uses RowParallelLinear. This exercises that GEMM
dispatch with vLLM's AR/RS collectives, not the NVIDIA fused-kernel algorithm.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.test_kimi_k3_gemm_rs_ar import _worker
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


class AmdProjectionCollective:
    def __init__(self, *, max_M, N, all_reduce, weights, world_size):
        self.all_reduce = all_reduce
        self.world_size = world_size
        self.layers = {}
        for k, weight in weights.items():
            layer = RowParallelLinear(
                input_size=k * world_size,
                output_size=N,
                bias=False,
                params_dtype=torch.bfloat16,
                reduce_results=all_reduce,
            ).to(weight.device)
            layer.weight.data.copy_(weight)
            self.layers[k] = layer

    def __call__(self, x, weight):
        result, _ = self.layers[x.shape[1]](x)
        if self.all_reduce:
            return result
        padding = -result.shape[0] % self.world_size
        if padding:
            result = F.pad(result, (0, 0, 0, padding))
        return tensor_model_parallel_reduce_scatter(result, dim=0)


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="AMD projection contract")
def test_kimi_k3_amd_projection_collectives(monkeypatch):
    """Preserve AR/RS tails, rank ordering, output lifetime, and graph replay."""
    if torch.accelerator.device_count() < 2:
        pytest.skip("Projection collectives require two GPUs")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_CUSTOM_AR", "1")
    try:
        torch.multiprocessing.spawn(_worker, args=(2, get_open_port(), True), nprocs=2)
    finally:
        cleanup_dist_env_and_memory()
