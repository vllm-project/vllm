# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the manual AllReduce + GemmaRMSNorm fusion used by MiniMax M3.

``fused_allreduce_gemma_rms_norm`` must match the unfused model path, i.e.
``GemmaRMSNorm(all_reduce(partial), residual)``, both on the flashinfer fast
path (TP>1 with flashinfer + NVSwitch) and on the eager fallback (TP==1, or when
flashinfer is unavailable / the GPU has no NVSwitch).
"""

from types import SimpleNamespace

import pytest
import torch
from torch.multiprocessing import spawn

import vllm.model_executor.layers.fused_allreduce_gemma_rms_norm as fused_ar_norm
from tests.utils import ensure_current_vllm_config, init_test_distributed_environment
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    fused_allreduce_gemma_rms_norm,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_random_seed


@ensure_current_vllm_config()
def _worker_fused_ar_norm(
    local_rank,
    world_size,
    port,
    num_tokens,
    hidden_size,
    dtype,
    seed,
    eps,
):
    """Per-rank worker: compare the fused helper vs all_reduce + GemmaRMSNorm."""
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        world_size, 1, local_rank, port, local_rank=local_rank
    )

    # Norm weights are identical across ranks (replicated GemmaRMSNorm).
    set_random_seed(seed)
    norm = GemmaRMSNorm(hidden_size, eps=eps).cuda().to(dtype)
    with torch.no_grad():
        norm.weight.normal_(mean=0.0, std=0.1)

    # Residual is shared across ranks; the partial o_proj output differs per rank
    # (each rank holds a partial sum that all_reduce combines).
    torch.manual_seed(seed + 7)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    torch.manual_seed(seed + 1000 + local_rank)
    partial = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # Reference: the unfused model path.
    reduced = tensor_model_parallel_all_reduce(partial.clone())
    ref_out, ref_res = norm(reduced, residual.clone())

    # Fused helper (flashinfer fast path when available, else fallback).
    out, res = fused_allreduce_gemma_rms_norm(partial.clone(), residual.clone(), norm)
    torch.accelerator.synchronize()

    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(res, ref_res, atol=2e-2, rtol=2e-2)

    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA required",
)
# world_size=1 exercises the TP==1 identity branch on a single GPU; >1 exercises
# the all_reduce + GemmaRMSNorm equivalence (flashinfer kernel or fallback).
@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("num_tokens", [1, 128, 333])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("seed", [42])
def test_fused_allreduce_gemma_rms_norm(
    world_size,
    num_tokens,
    hidden_size,
    dtype,
    eps,
    seed,
):
    num_gpus = current_platform.device_count()
    if num_gpus < world_size:
        pytest.skip(f"Need >= {world_size} GPUs, have {num_gpus}")
    port = str(get_open_port())
    spawn(
        _worker_fused_ar_norm,
        args=(
            world_size,
            port,
            num_tokens,
            hidden_size,
            dtype,
            seed,
            eps,
        ),
        nprocs=world_size,
        join=True,
    )


class _FakeQuickReduce:
    def __init__(self, accept: bool):
        self.accept = accept
        self.calls = []

    def should_fused_allreduce_rmsnorm(self, inp, residual, weight):
        return self.accept

    def fused_allreduce_rmsnorm(self, inp, residual, weight, epsilon, gemma_norm):
        self.calls.append(gemma_norm)
        return inp + 1, residual + 1


@pytest.mark.parametrize("accept", [True, False])
def test_fused_allreduce_gemma_rms_norm_routes_prefill_to_quick_reduce(
    monkeypatch: pytest.MonkeyPatch, default_vllm_config, accept: bool
):
    """Sizes the one-stage custom all-reduce declines go to the QuickReduce
    fused kernel with ``gemma_norm`` when it accepts the operands, and to
    all-reduce + GemmaRMSNorm otherwise."""
    norm = GemmaRMSNorm(64, eps=1e-6).to(torch.bfloat16)
    hidden = torch.randn(8, 64, dtype=torch.bfloat16)
    residual = torch.randn(8, 64, dtype=torch.bfloat16)
    ref_out, ref_res = norm(hidden * 4, residual.clone())

    qr = _FakeQuickReduce(accept)
    tp_group = SimpleNamespace(device_communicator=SimpleNamespace(qr_comm=qr))
    rocm = SimpleNamespace(is_rocm=lambda: True)
    monkeypatch.setattr(
        fused_ar_norm, "get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(fused_ar_norm, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(fused_ar_norm, "_can_use_flashinfer", lambda *_: (False, 0))
    monkeypatch.setattr(fused_ar_norm, "_can_use_aiter_fused_ar_rms", lambda *_: False)
    monkeypatch.setattr(fused_ar_norm, "current_platform", rocm)
    monkeypatch.setattr(
        fused_ar_norm, "tensor_model_parallel_all_reduce", lambda x: x * 4
    )

    out, res = fused_ar_norm.fused_allreduce_gemma_rms_norm(hidden, residual, norm)

    if accept:
        assert qr.calls == [True]
        torch.testing.assert_close(out, hidden + 1)
        torch.testing.assert_close(res, residual + 1)
    else:
        assert qr.calls == []
        torch.testing.assert_close(out, ref_out)
        torch.testing.assert_close(res, ref_res)
