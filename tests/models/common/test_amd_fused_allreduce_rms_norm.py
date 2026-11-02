# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER fused AR+RMSNorm must equal RMSNorm(all_reduce(partial)).

One TP2 spawn covers the fused kernel (4 tokens) and the unfused fallback
(256 tokens). Hidden dim 3584 is outside AITER's old static-template set.
"""

import pytest
import torch
from torch.multiprocessing import spawn

from tests.utils import ensure_current_vllm_config, init_test_distributed_environment
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.common.amd.ops.fused_allreduce_rms_norm import (
    can_fuse_allreduce_rms_norm,
    fused_allreduce_rms_norm_out,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AITER fused AR+RMSNorm is only wired up on ROCm",
)

HIDDEN_SIZE = 3584
EPS = 1e-5
DTYPE = torch.bfloat16
WORLD_SIZE = 2


def _check_matches_unfused(norm: RMSNorm, partial: torch.Tensor) -> None:
    ref = norm(tensor_model_parallel_all_reduce(partial.clone()))
    out = fused_allreduce_rms_norm_out(partial.clone(), norm)
    torch.accelerator.synchronize()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@ensure_current_vllm_config()
def _worker_fused_ar_rms(local_rank, world_size, port, seed):
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        world_size, 1, local_rank, port, local_rank=local_rank
    )

    set_random_seed(seed)
    norm = RMSNorm(HIDDEN_SIZE, eps=EPS).to(device=device, dtype=DTYPE)
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.1)

    torch.manual_seed(seed + local_rank)
    fused = torch.randn(4, HIDDEN_SIZE, dtype=DTYPE, device=device)
    fallback = torch.randn(256, HIDDEN_SIZE, dtype=DTYPE, device=device)

    _check_matches_unfused(norm, fused)
    assert not can_fuse_allreduce_rms_norm(fallback)
    _check_matches_unfused(norm, fallback)

    cleanup_dist_env_and_memory()


def test_fused_allreduce_rms_norm_out():
    if current_platform.device_count() < WORLD_SIZE:
        pytest.skip(f"Need >= {WORLD_SIZE} GPUs")
    spawn(
        _worker_fused_ar_rms,
        args=(WORLD_SIZE, str(get_open_port()), 42),
        nprocs=WORLD_SIZE,
        join=True,
    )


def test_gate_is_closed_without_aiter_custom_ar(monkeypatch):
    from vllm import _aiter_ops

    monkeypatch.setattr(
        _aiter_ops.rocm_aiter_ops, "is_custom_all_reduce_enabled", lambda: False
    )
    hidden = torch.empty(4, HIDDEN_SIZE, dtype=DTYPE)
    assert not can_fuse_allreduce_rms_norm(hidden)
