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
_INDEXER_PREFILL_CHUNK_METADATA_BACKENDS = frozenset({"DEEPSEEK_V32_INDEXER"})


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



def sparse_mla_triton_warmup(worker: "Worker") -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    max_num_prefills = min(worker.scheduler_config.max_num_seqs, max_tokens)
    if max_tokens <= 0 or max_num_prefills <= 0:
        return

    has_dsv4_backend = _has_attention_backend(
        runner, _DEEPSEEK_V4_SPARSE_MLA_BACKENDS
    )
    has_generic_backend = _has_attention_backend(
        runner, _GENERIC_SPARSE_MLA_BACKENDS
    )
    has_indexer_backend = _has_attention_backend(
        runner, _INDEXER_PREFILL_CHUNK_METADATA_BACKENDS
    )
    if not (has_dsv4_backend or has_generic_backend or has_indexer_backend):
        return

    vllm_config = runner.vllm_config
    try:
        from vllm.v1.attention.backends.mla.compressor_utils import (
            _COMPRESSED_SLOT_MAPPING_KERNEL,
        )
        from vllm.v1.attention.backends.mla.indexer import (
            _BUILD_PREFILL_CHUNK_METADATA_KERNEL,
            _PREPARE_UNIFORM_DECODE_KERNEL,
        )
        from vllm.v1.attention.backends.mla.sparse_swa import (
            _COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL,
            _COMPUTE_PREFILL_METADATA_KERNEL,
            _COMPUTE_SWA_INDICES_AND_LENS_KERNEL,
        )
        from vllm.v1.attention.backends.mla.sparse_utils import (
            _CONVERT_REQ_INDEX_TO_GLOBAL_INDEX_KERNEL,
        )

        _COMPRESSED_SLOT_MAPPING_KERNEL.warmup(vllm_config)
        _CONVERT_REQ_INDEX_TO_GLOBAL_INDEX_KERNEL.warmup(vllm_config)
        _PREPARE_UNIFORM_DECODE_KERNEL.warmup(vllm_config)
        _BUILD_PREFILL_CHUNK_METADATA_KERNEL.warmup(vllm_config)

        if has_dsv4_backend or has_generic_backend:
            _COMPUTE_PREFILL_METADATA_KERNEL.warmup(vllm_config)

        if has_dsv4_backend:
            from vllm.models.deepseek_v4.common.ops.cache_utils import (
                _COMBINE_TOPK_SWA_INDICES_KERNEL,
            )
            from vllm.models.deepseek_v4.sparse_mla import (
                _BUILD_C128A_TOPK_METADATA_KERNEL,
            )

            _COMPUTE_SWA_INDICES_AND_LENS_KERNEL.warmup(vllm_config)
            _COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL.warmup(vllm_config)
            _COMBINE_TOPK_SWA_INDICES_KERNEL.warmup(vllm_config)
            _BUILD_C128A_TOPK_METADATA_KERNEL.warmup(vllm_config)

    except Exception:
        logger.warning("Skipping sparse MLA Triton warmup.", exc_info=True)
