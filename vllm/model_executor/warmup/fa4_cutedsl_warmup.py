# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up FA4 CuTeDSL MLA prefill compile keys."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def fa4_cutedsl_warmup(worker: Worker) -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    vllm_config = runner.vllm_config
    if not vllm_config.model_config.use_mla:
        return

    try:
        backend_cls = get_mla_prefill_backend(vllm_config)
    except ValueError:
        # fall back to top-k MQA prefill path.
        return
    if backend_cls.get_name() != "FLASH_ATTN":
        return

    from vllm.v1.attention.backends.mla.prefill import flash_attn

    flash_attn.FA4_MLA_PREFILL_KERNEL.warmup(vllm_config)
