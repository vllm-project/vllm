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


# cohere start: cherry-pick upstream draft PR #43879 (precompile slot-mapping/KV-zero kernels)
def _warmup_zero_kv_blocks(worker: "Worker") -> None:
    """Warm up the Triton KV-zeroing kernel via the real model-runner path."""
    runner = worker.model_runner
    if not hasattr(runner, "_kv_block_zeroer"):
        return

    try:
        with torch.inference_mode():
            # Warm up with the smallest valid block id list.
            runner._zero_block_ids([0])
    except Exception:
        logger.debug("Skipping KV zero warmup.", exc_info=True)


def _warmup_slot_mapping(worker: "Worker") -> None:
    """Warm up the Triton slot-mapping kernel through BlockTable wrappers."""
    runner = worker.model_runner
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return

    block_tables = getattr(input_batch, "block_table", None)
    if block_tables is None:
        return

    device = runner.device
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    positions = torch.tensor([0], dtype=torch.int64, device=device)

    try:
        with torch.inference_mode():
            block_tables.clear()
            for block_table in block_tables.block_tables:
                block_table.add_row([0], row_idx=0)
            block_tables.commit_block_table(num_reqs=1)
            block_tables.compute_slot_mapping(
                num_reqs=1,
                query_start_loc=query_start_loc,
                positions=positions,
            )
            block_tables.clear()
    except Exception:
        logger.debug("Skipping slot-mapping warmup.", exc_info=True)


# cohere end


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

    # cohere start: cherry-pick upstream draft PR #43879 (precompile slot-mapping/KV-zero kernels)
    _warmup_zero_kv_blocks(worker)
    _warmup_slot_mapping(worker)
    # cohere end


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
