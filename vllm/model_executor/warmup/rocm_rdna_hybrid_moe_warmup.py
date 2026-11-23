# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup warmup for the Triton prefill GEMMs of the RDNA hybrid W4A16 MoE."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import MoERunner
from vllm.model_executor.layers.fused_moe.experts.rdna_hybrid_w4a16_moe import (
    RDNAHybridW4A16Experts,
)

if TYPE_CHECKING:
    import torch

    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _uses_rdna_hybrid_w4a16_moe(model: torch.nn.Module) -> bool:
    for module in model.modules():
        if not isinstance(module, MoERunner):
            continue
        moe_kernel = getattr(module._quant_method, "moe_kernel", None)
        if moe_kernel is not None and isinstance(
            moe_kernel.impl.fused_experts, RDNAHybridW4A16Experts
        ):
            return True
    return False


def rocm_rdna_hybrid_moe_warmup(worker: Worker) -> None:
    """Compile the Triton prefill GEMMs of the RDNA hybrid W4A16 MoE backend.

    The Triton GEMMs (GEMM2, and GEMM1 when the WMMA kernel doesn't apply)
    compile once per divisibility specialization of their routed-row count,
    and the startup runs at `max_num_batched_tokens` cover only one of them.
    Warm up one odd and one even token count so the first prefills don't
    compile.
    """
    if not _uses_rdna_hybrid_w4a16_moe(worker.get_model()):
        return

    max_m = worker.scheduler_config.max_num_batched_tokens
    top = max_m - 1
    sizes = [
        m for m in (top, top - 1) if m > RDNAHybridW4A16Experts.MAX_SKINNY_BATCH_SIZE
    ]
    start = time.perf_counter()
    for num_tokens in sizes:
        worker.model_runner._dummy_run(num_tokens, skip_eplb=True)
    logger.info(
        "Warmed up RDNA hybrid W4A16 MoE Triton GEMMs at %s tokens in %.1fs.",
        sizes,
        time.perf_counter() - start,
    )
