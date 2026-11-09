# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.model_executor.warmup.step5_dsa_warmup import (
    run_step5_scored_prefill_warmup,
)

from .model import Step5Attention

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def step5_csa_warmup(worker: "Worker") -> None:
    """JIT the CSA scored-prefill kernels before the first long prompt."""
    context = worker.vllm_config.compilation_config.static_forward_context
    for layer in context.values():
        if isinstance(layer, Step5Attention) and layer.indexer is not None:
            run_step5_scored_prefill_warmup(worker, layer.indexer.topk)
            return
