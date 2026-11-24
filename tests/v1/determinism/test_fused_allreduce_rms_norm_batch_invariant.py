# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test batch invariance for manual FlashInfer all-reduce + RMSNorm helpers."""

import pytest
import torch
from torch.multiprocessing import spawn

from tests.utils import ensure_current_vllm_config, init_test_distributed_environment
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    fused_allreduce_gemma_rms_norm,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


@ensure_current_vllm_config()
def _worker_batch_invariant_ar_norm(rank, world_size, port, gemma):
    torch.accelerator.set_device_index(rank)
    init_test_distributed_environment(world_size, 1, rank, port, local_rank=rank)
    norm_cls = GemmaRMSNorm if gemma else RMSNorm
    fused = fused_allreduce_gemma_rms_norm if gemma else fused_allreduce_rms_norm
    norm = norm_cls(6144).cuda().to(torch.bfloat16)
    gen = torch.Generator(device="cuda").manual_seed(42)
    residual = torch.randn(1, 6144, device="cuda", dtype=torch.bfloat16, generator=gen)
    gen.manual_seed(100 + rank)
    partial = torch.randn(1, 6144, device="cuda", dtype=torch.bfloat16, generator=gen)

    reduced = tensor_model_parallel_all_reduce(partial.clone())
    reference = norm(reduced, residual.clone())
    alone = fused(partial.clone(), residual.clone(), norm)
    # Cross the FlashInfer workspace limit, with the target at both ends.
    filler = torch.zeros(8190, 6144, device="cuda", dtype=torch.bfloat16)
    batch = fused(
        torch.cat((partial, filler, partial)),
        torch.cat((residual, filler, residual)),
        norm,
    )
    for expected, single, batched in zip(reference, alone, batch):
        torch.testing.assert_close(single, expected, rtol=0, atol=0)
        torch.testing.assert_close(batched[:1], single, rtol=0, atol=0)
        torch.testing.assert_close(batched[-1:], single, rtol=0, atol=0)
    cleanup_dist_env_and_memory()


@pytest.mark.parametrize("gemma", [False, True], ids=["rms", "gemma"])
def test_fused_allreduce_norm_batch_invariant(monkeypatch, gemma):
    """Manual norm fusion must obey BI across fused/unfused batch sizes."""
    if not current_platform.is_cuda() or current_platform.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    spawn(
        _worker_batch_invariant_ar_norm,
        args=(2, str(get_open_port()), gemma),
        nprocs=2,
        join=True,
    )
