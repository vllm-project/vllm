# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up TurboQuant Triton attention compile keys."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def turboquant_triton_warmup(worker: "Worker") -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    vllm_config = runner.vllm_config
    if not vllm_config.cache_config.cache_dtype.startswith("turboquant_"):
        return

    from vllm.v1.attention.ops.triton_decode_attention import (
        _DECODE_STAGE2_KERNEL,
    )
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _TQ_DECODE_STAGE1_KERNEL,
        _TQ_FULL_DEQUANT_KERNEL,
    )

    _TQ_DECODE_STAGE1_KERNEL.warmup(vllm_config)
    _TQ_FULL_DEQUANT_KERNEL.warmup(vllm_config)
    _DECODE_STAGE2_KERNEL.warmup(vllm_config)
