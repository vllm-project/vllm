# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up sparse-MLA Triton metadata kernels."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_scalar_specialization_rep,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2GPUModelRunner
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


def _compile_combine_topk_swa_indices_kernel(
    vllm_config: "VllmConfig",
) -> None:
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        _COMBINE_TOPK_SWA_INDICES_KERNEL,
    )

    _COMBINE_TOPK_SWA_INDICES_KERNEL.warmup(vllm_config)


def _warmup_hisparse_index_conversion(runner: "V2GPUModelRunner") -> None:
    from vllm.v1.attention.backends.mla.sparse_utils import (
        _convert_req_index_to_global_index_kernel,
        _remap_tiling,
    )

    topk_tokens = runner.vllm_config.model_config.hf_config.index_topk
    req_ids = TritonWarmupTensor(torch.int32)
    topk_indices = TritonWarmupTensor(torch.int32, shape=(1, topk_tokens))
    out = TritonWarmupTensor(torch.int32, shape=(1, topk_tokens))
    valid_counts = TritonWarmupTensor(torch.int32)
    prefill_request_ids = TritonWarmupTensor(torch.int32)
    prefill_workspace_starts = TritonWarmupTensor(torch.int32)

    attention_strides: dict[int, set[int]] = {}
    for layer in runner.vllm_config.compilation_config.static_forward_context.values():
        impl = getattr(layer, "impl", None)
        cache = getattr(impl, "hisparse_cache", None)
        if cache is None:
            continue
        hot = cache.runtime.hot
        attention_strides.setdefault(hot.block_size, set()).add(
            hot.attention_block_stride
        )

    for block_table, block_size in zip(
        runner.block_tables.input_block_tables,
        runner.kernel_block_sizes,
        strict=True,
    ):
        max_num_blocks_per_req = block_table.shape[1]
        block_table_desc = TritonWarmupTensor(
            torch.int32, shape=(1, max_num_blocks_per_req)
        )
        block_strides = attention_strides.get(block_size, set()) | {block_size}
        for block_stride in block_strides:
            for has_prefill, count_valid in (
                (False, False),
                (False, True),
                (True, True),
            ):
                single_tile, block_n, tiles_per_row, num_warps = _remap_tiling(
                    topk_tokens, 128, count_valid
                )
                _convert_req_index_to_global_index_kernel.warmup(
                    req_ids,
                    block_table_desc,
                    topk_indices,
                    out,
                    valid_counts if count_valid else None,
                    prefill_request_ids if has_prefill else None,
                    prefill_workspace_starts if has_prefill else None,
                    max_num_blocks_per_req,
                    block_size,
                    block_stride,
                    block_n,
                    has_prefill,
                    count_valid,
                    single_tile,
                    False,
                    1,
                    0,
                    1,
                    triton_scalar_specialization_rep(block_table_desc.stride()[0]),
                    triton_scalar_specialization_rep(block_table_desc.stride()[1]),
                    triton_scalar_specialization_rep(topk_indices.stride()[0]),
                    triton_scalar_specialization_rep(topk_indices.stride()[1]),
                    triton_scalar_specialization_rep(out.stride()[0]),
                    triton_scalar_specialization_rep(out.stride()[1]),
                    num_warps=num_warps,
                    grid=(1, tiles_per_row),
                )


def sparse_mla_triton_warmup(worker: "Worker") -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    max_num_prefills = min(worker.scheduler_config.max_num_seqs, max_tokens)
    if max_tokens <= 0 or max_num_prefills <= 0:
        return

    vllm_config = runner.vllm_config
    if vllm_config.attention_config.hisparse_config is not None:
        _warmup_hisparse_index_conversion(cast("V2GPUModelRunner", runner))
    try:
        if _has_attention_backend(runner, _DEEPSEEK_V4_SPARSE_MLA_BACKENDS):
            _compile_sparse_swa_prefill_metadata_kernel(vllm_config)
            _compile_prefill_chunk_metadata_kernel(vllm_config)
            _compile_combine_topk_swa_indices_kernel(vllm_config)
        elif _has_attention_backend(runner, _GENERIC_SPARSE_MLA_BACKENDS):
            _compile_sparse_swa_prefill_metadata_kernel(vllm_config)
            _compile_prefill_chunk_metadata_kernel(vllm_config)
        elif _has_attention_backend(runner, _INDEXER_PREFILL_CHUNK_METADATA_BACKENDS):
            _compile_prefill_chunk_metadata_kernel(vllm_config)

    except Exception:
        logger.warning("Skipping sparse MLA Triton warmup.", exc_info=True)
