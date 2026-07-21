# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up FA4 CuTeDSL kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    from vllm.models.inkling.configs import InklingModelConfig
    from vllm.models.inkling.nvidia.ops.fa4_warmup import (
        INKLING_FA4_REL_ATTENTION_KERNEL,
    )

    vllm_config = worker.vllm_config
    hf_config = vllm_config.model_config.hf_config
    if not isinstance(hf_config, InklingModelConfig):
        return

    INKLING_FA4_REL_ATTENTION_KERNEL.warmup(vllm_config)


def fa4_cutedsl_warmup(worker: Worker) -> None:
    _warm_fa4_mla_prefill(worker)
    _warm_inkling_fa4_rel_attention(worker)
