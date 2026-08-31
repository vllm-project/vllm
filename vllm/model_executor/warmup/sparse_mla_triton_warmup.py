# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up sparse-MLA Triton metadata kernels."""

from typing import TYPE_CHECKING

import torch

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
_PREFILL_CHUNK_METADATA_ONLY_BACKENDS = frozenset(
    {"B12X_MLA_SPARSE", "DEEPSEEK_V32_INDEXER"}
)


def _attention_backend_name(backend: object) -> str | None:
    get_name = getattr(backend, "get_name", None)
    if get_name is None:
        return None
    try:
        return get_name()
    except NotImplementedError:
        return None


def _configured_attention_backend_name(
    vllm_config: "VllmConfig",
) -> str | None:
    attention_config = getattr(vllm_config, "attention_config", None)
    backend = getattr(attention_config, "backend", None)
    if isinstance(backend, str):
        return backend
    name = getattr(backend, "name", None)
    return name if isinstance(name, str) else None


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

    _COMPUTE_PREFILL_METADATA_KERNEL.warmup(vllm_config)


def _compile_prefill_chunk_metadata_kernel(
    vllm_config: "VllmConfig",
) -> None:
    from vllm.v1.attention.backends.mla.indexer import (
        _BUILD_PREFILL_CHUNK_METADATA_KERNEL,
    )

    _BUILD_PREFILL_CHUNK_METADATA_KERNEL.warmup(vllm_config)


def _execute_prefill_chunk_metadata_kernel(worker: "Worker") -> int:
    """Populate Triton's in-process cache with runtime pointer variants."""
    from vllm.v1.attention.backends.mla.indexer import (
        _BUILD_PREFILL_CHUNK_METADATA_KERNEL,
    )

    parameter = next(worker.get_model().parameters(), None)
    if parameter is None:
        return 0
    device = parameter.device
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cu_compressed_seq_lens = torch.tensor(
        [0, 1], dtype=torch.int32, device=device
    )
    token_to_seq = torch.empty((1,), dtype=torch.int32, device=device)
    cu_compressed_seq_len_ks = torch.empty(
        (1,), dtype=torch.int32, device=device
    )
    cu_compressed_seq_len_ke = torch.empty(
        (1,), dtype=torch.int32, device=device
    )
    compress_ratios = {
        key.COMPRESS_RATIO
        for key in _BUILD_PREFILL_CHUNK_METADATA_KERNEL.get_warmup_keys(
            worker.vllm_config
        )
    }
    warmed = 0
    for compress_ratio in sorted(compress_ratios):
        aligned = torch.tensor(
            [compress_ratio], dtype=torch.int32, device=device
        )
        unaligned_storage = torch.tensor(
            [-1, compress_ratio], dtype=torch.int32, device=device
        )
        for uncompressed_seq_lens in (aligned, unaligned_storage[1:]):
            _BUILD_PREFILL_CHUNK_METADATA_KERNEL(
                query_start_loc,
                uncompressed_seq_lens,
                cu_compressed_seq_lens,
                cu_compressed_seq_lens,
                token_to_seq,
                cu_compressed_seq_len_ks,
                cu_compressed_seq_len_ke,
                0,
                1,
                0,
                1,
                1,
                num_reqs=1,
                COMPRESS_RATIO=compress_ratio,
            )
            warmed += 1
    if warmed:
        torch.accelerator.synchronize()
    return warmed


def _compile_combine_topk_swa_indices_kernel(
    vllm_config: "VllmConfig",
) -> None:
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        _COMBINE_TOPK_SWA_INDICES_KERNEL,
    )

    _COMBINE_TOPK_SWA_INDICES_KERNEL.warmup(vllm_config)


def sparse_mla_triton_warmup(worker: "Worker") -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    max_num_prefills = min(worker.scheduler_config.max_num_seqs, max_tokens)
    if max_tokens <= 0 or max_num_prefills <= 0:
        return

    vllm_config = runner.vllm_config
    warmed_prefill_chunk = False
    try:
        if _has_attention_backend(runner, _DEEPSEEK_V4_SPARSE_MLA_BACKENDS):
            _compile_sparse_swa_prefill_metadata_kernel(vllm_config)
            _compile_prefill_chunk_metadata_kernel(vllm_config)
            warmed_prefill_chunk = True
            _compile_combine_topk_swa_indices_kernel(vllm_config)
        elif _has_attention_backend(runner, _GENERIC_SPARSE_MLA_BACKENDS):
            _compile_sparse_swa_prefill_metadata_kernel(vllm_config)
            _compile_prefill_chunk_metadata_kernel(vllm_config)
            warmed_prefill_chunk = True
        elif (
            _configured_attention_backend_name(vllm_config)
            in _PREFILL_CHUNK_METADATA_ONLY_BACKENDS
            or _has_attention_backend(
                runner, _PREFILL_CHUNK_METADATA_ONLY_BACKENDS
            )
        ):
            _compile_prefill_chunk_metadata_kernel(vllm_config)
            warmed_prefill_chunk = True
        if warmed_prefill_chunk:
            warmed = _execute_prefill_chunk_metadata_kernel(worker)
            logger.info(
                "Warmed up %d sparse MLA prefill metadata runtime variants.",
                warmed,
            )

    except Exception:
        logger.warning("Skipping sparse MLA Triton warmup.", exc_info=True)
