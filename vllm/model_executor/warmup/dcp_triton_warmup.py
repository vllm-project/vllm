# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DCP kernels and communicators that only long prefills reach.

Decode and cudagraph capture take the direct symmetric-memory DCP combine, so
the NCCL all-to-all fallback (used once a batch exceeds the direct workspace)
and its pack/unpack Triton kernels first run on a long prefill. The DCP
device-group NCCL communicator is also created lazily on that first collective.
"""

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    triton_scalar_specialization_rep,
)
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def _combine_lse_dtypes(
    impl: object, model_dtype: torch.dtype
) -> tuple[torch.dtype, ...]:
    """LSE dtypes the runtime combine can see for a layer with ``impl``.

    Prefill backends return fp32 LSE. Triton MLA decode returns LSE in the
    query dtype, and MLAAttention.forward_impl passes it to
    dcp_manager.combine unconverted, so those layers reach the pack kernel
    with the model dtype as well.
    """
    if isinstance(impl, TritonMLAImpl):
        return (torch.float32, model_dtype)
    return (torch.float32,)


def _num_seqs_reps(max_num_seqs: int) -> list[int]:
    """One batch size per (Triton int class, next_power_of_2) pack variant."""
    reps: dict[tuple[int, int], int] = {}
    for n in range(1, max_num_seqs + 1):
        reps.setdefault((triton_scalar_specialization_rep(n), next_power_of_2(n)), n)
    return sorted(reps.values())


@torch.inference_mode()
def _warm_dcp_a2a_combine(worker: "Worker") -> None:
    """Run the NCCL all-to-all combine once per layer config and batch class.

    Collective: every DCP rank must reach this with the same layers.
    """
    runner = worker.model_runner
    max_num_seqs = worker.scheduler_config.max_num_seqs
    seen: set[tuple[int, int, torch.dtype, torch.dtype, bool]] = set()
    for module in worker.get_model().modules():
        if not isinstance(module, MLAAttention) or module.dcp_manager is None:
            continue
        manager = module.dcp_manager
        if not manager.use_a2a:
            continue
        num_heads = module.num_heads * manager.group.world_size
        dtype = runner.model_config.dtype
        for lse_dtype in _combine_lse_dtypes(module.impl, dtype):
            key = (
                num_heads,
                module.kv_lora_rank,
                dtype,
                lse_dtype,
                module.impl.lse_base_on_e,
            )
            if key in seen:
                continue
            seen.add(key)
            # One token past the direct workspace forces the NCCL fallback.
            num_tokens = manager.max_num_tokens + 1
            out = torch.zeros(
                (num_tokens, num_heads, module.kv_lora_rank),
                dtype=dtype,
                device=runner.device,
            )
            lse = torch.zeros(
                (num_tokens, num_heads), dtype=lse_dtype, device=runner.device
            )
            for num_seqs in _num_seqs_reps(min(max_num_seqs, num_tokens)):
                seq_lens = torch.ones(num_seqs, dtype=torch.int32, device=runner.device)
                query_start_loc = torch.zeros(
                    num_seqs + 1, dtype=torch.int32, device=runner.device
                )
                query_start_loc[-1] = num_tokens
                manager.combine(
                    out, lse, seq_lens=seq_lens, query_start_loc=query_start_loc
                )
    torch.accelerator.synchronize()


def dcp_triton_warmup(worker: "Worker") -> None:
    parallel_config = worker.vllm_config.parallel_config
    if parallel_config.decode_context_parallel_size <= 1:
        return
    if worker.model_runner.is_pooling_model:
        return
    _warm_dcp_a2a_combine(worker)
