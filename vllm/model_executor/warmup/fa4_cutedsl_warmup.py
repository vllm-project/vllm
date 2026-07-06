# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up FA4 CuTeDSL MLA prefill compile keys."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def fa4_cutedsl_warmup(worker: "Worker") -> None:
    from vllm.v1.attention.backends.mla.prefill import flash_attn

    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    vllm_config = runner.vllm_config
    backend_cls = get_mla_prefill_backend(vllm_config)
    if backend_cls.get_name() != "FLASH_ATTN":
        return

    flash_attn.FA4_MLA_PREFILL_KERNEL.warmup(vllm_config)
