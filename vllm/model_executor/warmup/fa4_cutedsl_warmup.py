# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up FA4 CuTeDSL kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def _warm_fa4_mla_prefill(worker: Worker) -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    vllm_config = runner.vllm_config
    if not vllm_config.model_config.use_mla:
        return

    backend_cls = get_mla_prefill_backend(vllm_config)
    if backend_cls.get_name() != "FLASH_ATTN":
        return

    from vllm.v1.attention.backends.mla.prefill import flash_attn

    flash_attn.FA4_MLA_PREFILL_KERNEL.warmup(vllm_config)


def _warm_inkling_fa4_rel_attention(worker: Worker) -> None:
    seen_compile_keys: set[Any] = set()
    for module in worker.get_model().modules():
        warmup_kernel = getattr(module, "fa4_rel_attention_kernel", None)
        if warmup_kernel is not None:
            warmup_kernel.warmup(seen_compile_keys=seen_compile_keys)


def fa4_cutedsl_warmup(worker: Worker) -> None:
    _warm_fa4_mla_prefill(worker)
    _warm_inkling_fa4_rel_attention(worker)
