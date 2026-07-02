# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up sparse-MLA Triton metadata kernels."""

from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_DEEPSEEK_V4_SPARSE_MLA_BACKENDS = frozenset(
    {
        "FLASHMLA_SPARSE_DSV4",
        "FLASHINFER_MLA_SPARSE_DSV4",
        "ROCM_FLASHMLA_SPARSE_DSV4",
        "DEEPSEEK_SPARSE_SWA",
    }
)
_GENERIC_SPARSE_MLA_BACKENDS = frozenset(
    {
        "FLASHMLA_SPARSE",
        "FLASHINFER_MLA_SPARSE",
        "FLASHINFER_MLA_SPARSE_SM120",
    }
)


def _attention_backend_name(backend: object) -> str | None:
    get_name = getattr(backend, "get_name", None)
    if get_name is None:
        return None
    try:
        return get_name()
    except NotImplementedError:
        return None


def _has_attention_backend(
    runner: "GPUModelRunner",
    backend_names: frozenset[str],
) -> bool:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name in backend_names:
                return True
    return False


def _compile_sparse_swa_prefill_metadata_kernel(
    vllm_config: "VllmConfig",
) -> None:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _COMPUTE_PREFILL_METADATA_KERNEL,
    )

    for compile_key in _COMPUTE_PREFILL_METADATA_KERNEL.get_warmup_keys(
        vllm_config
    ):
        _COMPUTE_PREFILL_METADATA_KERNEL.compile(compile_key)


def _compile_prefill_chunk_metadata_kernel(
    vllm_config: "VllmConfig",
) -> None:
    from vllm.v1.attention.backends.mla.indexer import (
        _BUILD_PREFILL_CHUNK_METADATA_KERNEL,
    )

    for compile_key in _BUILD_PREFILL_CHUNK_METADATA_KERNEL.get_warmup_keys(
        vllm_config
    ):
        _BUILD_PREFILL_CHUNK_METADATA_KERNEL.compile(compile_key)


def _compile_combine_topk_swa_indices_kernel(
    vllm_config: "VllmConfig",
) -> None:
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        _COMBINE_TOPK_SWA_INDICES_KERNEL,
    )

    for compile_key in _COMBINE_TOPK_SWA_INDICES_KERNEL.get_warmup_keys(
        vllm_config
    ):
        _COMBINE_TOPK_SWA_INDICES_KERNEL.compile(compile_key)


def _compile_sparse_mla_triton_kernels(
    runner: "GPUModelRunner",
) -> None:
    _compile_sparse_swa_prefill_metadata_kernel(runner.vllm_config)
    _compile_prefill_chunk_metadata_kernel(runner.vllm_config)


def sparse_mla_triton_warmup(worker: "Worker") -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    max_num_prefills = min(worker.scheduler_config.max_num_seqs, max_tokens)
    if max_tokens <= 0 or max_num_prefills <= 0:
        return

    try:
        has_dsv4_sparse_mla_backend = _has_attention_backend(
            runner, _DEEPSEEK_V4_SPARSE_MLA_BACKENDS
        )
        if has_dsv4_sparse_mla_backend or _has_attention_backend(
            runner, _GENERIC_SPARSE_MLA_BACKENDS
        ):
            _compile_sparse_mla_triton_kernels(runner)
        if has_dsv4_sparse_mla_backend:
            _compile_combine_topk_swa_indices_kernel(runner.vllm_config)
    except Exception:
        logger.warning("Skipping sparse MLA Triton warmup.", exc_info=True)
