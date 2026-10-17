# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.warmup.attention_warmup import (
    mixed_batch_attention_warmup,
)
from vllm.models.minimax_m3.nvidia.model import MiniMaxM3SparseAttention
from vllm.platforms import current_platform
from vllm.tracing import instrument

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


@instrument(span_name="MiniMax M3 MSA warmup")
def minimax_m3_msa_warmup(worker: "Worker") -> None:
    sparse_module = next(
        (
            module
            for module in worker.get_model().modules()
            if isinstance(module, MiniMaxM3SparseAttention)
        ),
        None,
    )
    if sparse_module is None:
        return
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        return

    logger.info("Warming up MiniMax M3 MSA kernels.")

    # Cover sparse prefill through the normal model path.
    mixed_batch_attention_warmup(worker)
