# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, cast

from vllm.logger import init_logger
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def mixed_batch_attention_warmup(worker: "Worker", num_tokens: int = 16) -> None:
    """Run a mixed prefill+decode batch to warm up both attention paths."""
    num_tokens = min(
        num_tokens,
        worker.scheduler_config.max_num_batched_tokens,
        worker.model_config.max_model_len,
    )
    if not worker.use_v2_model_runner:
        worker.model_runner._dummy_run(
            num_tokens=num_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )
        return

    if not run_mixed_prefill_decode_warmup(
        cast("GPUModelRunner", worker.model_runner),
        worker.execute_model,
        worker.sample_tokens,
        num_tokens,
    ):
        logger.info_once(
            "Skipping mixed prefill+decode attention warmup; attention kernels "
            "will be loaded on the first request."
        )
