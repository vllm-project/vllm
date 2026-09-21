# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The K3 AMD latent tail must equal RMSNorm(all_reduce(partial)).

The fused AITER path and the unfused fallback are both compared against an
explicit all-reduce + RMSNorm, so a silently skipped fusion or a wrong
residual shows up as a numeric mismatch rather than plausible-looking text.

``_zero_residual`` must also hand back the same buffer for a repeated shape:
a buffer that moved between CUDA graph captures would leave an earlier graph
replaying against freed memory.
"""

import pytest
import torch
from torch.multiprocessing import spawn

from tests.utils import ensure_current_vllm_config, init_test_distributed_environment
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.kimi_k3.amd.ops.fused_ar_rms import (
    _zero_residual,
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

LATENT_SIZE = 3584
EPS = 1e-5
DTYPE = torch.bfloat16


@ensure_current_vllm_config()
def _worker_fused_ar_rms(local_rank, world_size, port, num_tokens, seed):
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        world_size, 1, local_rank, port, local_rank=local_rank
    )

    set_random_seed(seed)
    norm = RMSNorm(LATENT_SIZE, eps=EPS).to(device=device, dtype=DTYPE)
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.1)

    torch.manual_seed(seed + local_rank)
    partial = torch.randn(num_tokens, LATENT_SIZE, dtype=DTYPE, device=device)

    ref = norm(tensor_model_parallel_all_reduce(partial.clone()))
    out = fused_allreduce_rms_norm_out(partial.clone(), norm)
    torch.accelerator.synchronize()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    cleanup_dist_env_and_memory()


@pytest.mark.parametrize("world_size", [1, 2])
# 4 tokens sits inside AITER's one-stage gate, 256 outside it, so a single
# run covers both the fused kernel and the unfused fallback.
@pytest.mark.parametrize("num_tokens", [4, 256])
def test_fused_allreduce_rms_norm_out(world_size, num_tokens):
    if current_platform.device_count() < world_size:
        pytest.skip(f"Need >= {world_size} GPUs")
    spawn(
        _worker_fused_ar_rms,
        args=(world_size, str(get_open_port()), num_tokens, 42),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(
    not current_platform.is_rocm() or current_platform.device_count() < 1,
    reason="needs a GPU",
)
def test_zero_residual_is_stable_per_shape():
    device = torch.device("cuda:0")
    small = torch.empty(4, LATENT_SIZE, dtype=DTYPE, device=device)
    large = torch.empty(256, LATENT_SIZE, dtype=DTYPE, device=device)

    first = _zero_residual(small)
    _zero_residual(large)

    assert _zero_residual(small).data_ptr() == first.data_ptr()
    assert not first.any()


def test_gate_is_closed_without_aiter_custom_ar(monkeypatch):
    from vllm import _aiter_ops

    monkeypatch.setattr(
        _aiter_ops.rocm_aiter_ops, "is_custom_all_reduce_enabled", lambda: False
    )
    hidden = torch.empty(4, LATENT_SIZE, dtype=DTYPE)
    assert not can_fuse_allreduce_rms_norm(hidden)
