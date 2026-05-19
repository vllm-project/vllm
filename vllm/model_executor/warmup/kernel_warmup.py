# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def kernel_warmup(worker: "Worker"):
    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    deepseek_v4_flashinfer_sparse_mla_warmup(worker)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )


def deepseek_v4_flashinfer_sparse_mla_warmup(worker: "Worker") -> None:
    """Warm the DSV4 FlashInfer sparse-index builder variants.

    CUDA graph capture exercises mixed batches, but Triton can still see the
    first real prefill wave as a separate specialization for the per-layer C4A
    and C128A index shapes. Compile those tiny index-builder launches during
    engine warmup so they do not appear as inference-time bubbles.
    """
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _compute_prefill_metadata_kernel,
    )
    from vllm.v1.attention.ops.deepseek_v4_ops.cache_utils import (
        build_flashinfer_mixed_sparse_indices,
    )

    hf_config = worker.vllm_config.model_config.hf_config
    compress_ratios = {
        int(ratio) for ratio in getattr(hf_config, "compress_ratios", ())
    }
    if not compress_ratios:
        return

    window_size = int(getattr(hf_config, "sliding_window", 0))
    if window_size <= 0:
        return

    logger.info("Warming up DeepSeek V4 FlashInfer sparse MLA index kernels.")
    device = worker.model_runner.device
    index_topk = int(getattr(hf_config, "index_topk", 0))
    max_model_len = worker.vllm_config.model_config.max_model_len
    max_num_seqs = max(1, worker.scheduler_config.max_num_seqs)

    def _prefill_batch_sizes() -> list[int]:
        sizes: list[int] = []
        size = 1
        while size < max_num_seqs:
            sizes.append(size)
            size *= 2
        sizes.append(max_num_seqs)
        return sizes

    max_prefill_reqs = max(_prefill_batch_sizes())
    seq_lens = torch.ones((max_prefill_reqs,), device=device, dtype=torch.int32)
    query_start_loc = torch.arange(
        max_prefill_reqs + 1, device=device, dtype=torch.int32
    )
    prefill_query_start_loc = torch.empty(
        max_prefill_reqs + 1, device=device, dtype=torch.int32
    )
    prefill_gather_lens = torch.empty(
        max_prefill_reqs, device=device, dtype=torch.int32
    )
    for num_prefills in _prefill_batch_sizes():
        _compute_prefill_metadata_kernel[(1,)](
            prefill_query_start_loc[: num_prefills + 1],
            prefill_gather_lens[:num_prefills],
            seq_lens[:num_prefills],
            query_start_loc[: num_prefills + 1],
            num_prefills,
            0,
            window_size,
            BLOCK_SIZE=1 << num_prefills.bit_length(),
        )

    for compress_ratio in sorted(compress_ratios):
        if compress_ratio == 4:
            topk = index_topk
            decode_compressed_indices_are_local = True
            has_decode_compressed_lens = False
        elif compress_ratio == 128:
            topk = (max_model_len + compress_ratio - 1) // compress_ratio
            topk = ((topk + 127) // 128) * 128
            decode_compressed_indices_are_local = False
            has_decode_compressed_lens = True
        else:
            continue

        if topk <= 0:
            continue

        decode_swa_indices = torch.zeros(
            (1, window_size), device=device, dtype=torch.int32
        )
        decode_compressed_indices = torch.zeros(
            (1, topk), device=device, dtype=torch.int32
        )
        prefill_topk_indices = torch.zeros((1, topk), device=device, dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([1, 2], device=device, dtype=torch.int32)
        token_to_req_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
        swa_block_table = torch.zeros((2, 1), device=device, dtype=torch.int32)
        compressed_block_table = torch.zeros((2, 1), device=device, dtype=torch.int32)
        decode_compressed_topk_lens = (
            torch.ones((1,), device=device, dtype=torch.int32)
            if has_decode_compressed_lens
            else None
        )
        decode_is_valid_token = (
            torch.ones((1,), device=device, dtype=torch.bool)
            if decode_compressed_indices_are_local
            else None
        )

        build_flashinfer_mixed_sparse_indices(
            decode_swa_indices,
            decode_compressed_indices,
            decode_compressed_topk_lens,
            prefill_topk_indices,
            query_start_loc,
            seq_lens,
            token_to_req_indices,
            swa_block_table,
            256,
            compressed_block_table,
            max(1, 256 // compress_ratio),
            window_size,
            compress_ratio,
            topk,
            decode_compressed_indices_are_local=decode_compressed_indices_are_local,
            decode_is_valid_token=decode_is_valid_token,
        )
    torch.cuda.synchronize()


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.
    """
    import vllm.utils.flashinfer as fi_utils

    with torch.inference_mode(), fi_utils.autotune():
        # Certain FlashInfer kernels (e.g. nvfp4 routed moe) are
        # incompatible with autotuning. This state is used to skip
        # those kernels during the autotuning process.
        fi_utils._is_fi_autotuning = True

        # We skip EPLB here since we don't want to record dummy metrics
        # When autotuning with number of tokens m, flashinfer will autotune
        # operations for all number of tokens up to m.
        # So we only need to run with the max number of tokens.
        runner._dummy_run(
            runner.scheduler_config.max_num_batched_tokens,
            skip_eplb=True,
            is_profile=True,
        )

        fi_utils._is_fi_autotuning = False
