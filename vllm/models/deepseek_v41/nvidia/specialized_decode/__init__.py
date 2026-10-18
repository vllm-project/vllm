# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 specialized decode on NVIDIA GPUs: a decode executor over the
stock backbone, enabled by
``additional_config={"decode_backend": "latency_specialized"}``.

The executor runs FULL-graph decode steps; every other step runs the stock
forward. It issues cross-rank collectives, so the deployment check below is
collective over the TP group and every per-step decision reads host state that
all TP ranks share.
"""

import importlib.metadata
from dataclasses import dataclass
from functools import cache

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from ..ops.mhc import supports_mhc_all_reduce, supports_mhc_overlap

logger = init_logger(__name__)

# Rows of the executor's fused all-reduce + mHC mailbox: the largest FULL
# decode graph at 64 requests x 6 DSpark tokens.
FUSED_MHC_MAX_ROWS = 384
# Rows whose next mHC coefficients run on the side stream; larger steps compute
# them inline after the fused kernel.
MHC_OVERLAP_MAX_ROWS = 48
# Rows of the low-batch GEMMs: one DSpark verification step of up to 8 rows
# (6 at one request).
LOW_BATCH_MAX_ROWS = 8
# The FlashInfer (requirements/cuda.txt) whose internals the publishing wo_b
# GEMM, the MXFP8 layer input and the pinned MXFP8 tactics were checked
# against, and the first whose sparse-MLA decode takes caller-owned KV counters.
FLASHINFER_VERSION = "0.7.1"


@dataclass(frozen=True)
class DecodeConfig:
    # Tune the executor's FlashInfer GEMMs at build
    # (kernel_config.enable_flashinfer_autotune is not False).
    autotune: bool
    # Compile the mHC pre-norm kernels of the executor's row counts.
    jit_warmup: bool


@cache
def uses_checked_flashinfer(feature: str) -> bool:
    """Whether the installed FlashInfer is the version whose internals
    ``feature`` uses; logs once per feature when it is not."""
    try:
        version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version == FLASHINFER_VERSION:
        return True
    logger.warning(
        "DeepSeek-V4.1 specialized decode: %s needs flashinfer-python %s "
        "(found %s); using the stock path instead.",
        feature,
        FLASHINFER_VERSION,
        version,
    )
    return False


def unsupported(vllm_config: VllmConfig) -> list[str]:
    """Every requirement of the specialized decode this deployment misses."""
    parallel = vllm_config.parallel_config
    attention = vllm_config.attention_config
    spec = vllm_config.speculative_config
    compilation = vllm_config.compilation_config
    cudagraph_mode = compilation.cudagraph_mode
    requirements = {
        "a GB300 (SM103) GPU": current_platform.is_device_capability(103),
        "Model Runner V2": vllm_config.use_v2_model_runner,
        "tensor parallel size 4 without pipeline, data, expert or context "
        "parallelism or micro-batching": (
            parallel.tensor_parallel_size == 4
            and parallel.pipeline_parallel_size == 1
            and parallel.data_parallel_size == 1
            and not parallel.enable_expert_parallel
            and parallel.decode_context_parallel_size == 1
            and parallel.prefill_context_parallel_size == 1
            and not parallel.use_ubatching
        ),
        "DSpark speculative decoding without adaptive verification": (
            spec is not None
            and spec.method == "dspark"
            and not spec.enable_adaptive_verification
        ),
        "the FLASHINFER_MLA_SPARSE_DSV41 attention backend, an fp8 KV cache "
        "and the MXFP4 indexer with sparse logits": (
            attention.backend == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV41
            and vllm_config.cache_config.cache_dtype in ("fp8", "fp8_e4m3")
            and attention.indexer_kv_dtype == "mxfp4"
            and attention.indexer_sparse_logits
        ),
        "FULL CUDA graphs for decode batches only (FULL_AND_PIECEWISE or "
        "FULL_DECODE_ONLY)": (
            cudagraph_mode is not None
            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.mixed_mode() != CUDAGraphMode.FULL
        ),
        # The decode loop runs in forward; a compiled forward would trace it.
        "no torch.compile of the model": (
            compilation.mode != CompilationMode.STOCK_TORCH_COMPILE
        ),
        # The routed-expert GEMMs and the publishing wo_b bind weight addresses.
        "no sleep mode": not vllm_config.model_config.enable_sleep_mode,
        "no LoRA, KV transfer or batch invariance": (
            vllm_config.lora_config is None
            and vllm_config.kv_transfer_config is None
            and not envs.VLLM_BATCH_INVARIANT
        ),
        "DeepGEMM and NVLink multicast for the fused all-reduce + mHC": (
            supports_mhc_overlap(vllm_config) and supports_mhc_all_reduce(vllm_config)
        ),
    }
    return [need for need, ok in requirements.items() if not ok]


def specialized_decode_config(vllm_config: VllmConfig) -> DecodeConfig | None:
    """The specialized decode settings, or None for the stock decode.

    Collective over the TP group when requested: every rank raises if any rank
    cannot run the specialized decode.
    """
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend == "auto":
        return None
    if backend != "latency_specialized":
        raise ValueError("Supported decode_backend values are auto/latency_specialized")
    missing = unsupported(vllm_config)
    supported = torch.tensor([0 if missing else 1], dtype=torch.int32)
    # The NVLink multicast check is rank-local (the custom all-reduce catches a
    # failed setup on each rank), and ranks that disagree would hang.
    dist.all_reduce(supported, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
    if not supported.item():
        reason = "; ".join(missing) or "another tensor parallel rank to support it"
        raise ValueError(f"decode_backend=latency_specialized needs {reason}.")
    logger.info_once(
        "DeepSeek-V4.1 specialized decode on, for FULL-graph decode steps of up "
        "to %d rows.",
        FUSED_MHC_MAX_ROWS,
    )
    kernel = vllm_config.kernel_config
    return DecodeConfig(
        autotune=kernel.enable_flashinfer_autotune is not False,
        jit_warmup=bool(kernel.enable_jit_warmup),
    )
