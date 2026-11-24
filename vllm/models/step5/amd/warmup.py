# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from ..common.warmup import run_step5_scored_prefill_warmup

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    from .model import Step5Attention


def step5_csa_warmup(layers: list["Step5Attention"], worker: "Worker") -> None:
    """JIT the CSA scored-prefill kernels before the first long prompt."""
    for layer in layers:
        if layer.indexer is not None:
            run_step5_scored_prefill_warmup(worker, layer.indexer.topk)
            return
