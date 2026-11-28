# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.model_executor.warmup.kernel_warmup import KernelWarmup

from ..common.warmup import run_step5_scored_prefill_warmup
from .model import Step5Attention

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


class Step5KernelWarmup(KernelWarmup):
    def warmup_model_runs(self, worker: "Worker") -> None:
        super().warmup_model_runs(worker)
        step5_csa_warmup(self.find_modules(Step5Attention), worker)


def step5_csa_warmup(layers: list["Step5Attention"], worker: "Worker") -> None:
    """JIT the CSA scored-prefill kernels before the first long prompt."""
    for layer in layers:
        if layer.indexer is not None:
            run_step5_scored_prefill_warmup(worker, layer.indexer.topk)
            return
