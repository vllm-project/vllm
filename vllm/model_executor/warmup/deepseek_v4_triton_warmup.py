# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek V4 model-local Triton kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _is_deepseek_v4_config(worker: "Worker") -> bool:
    model_config = getattr(worker.vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) == "deepseek_v4"


def _warm_prepare_megamoe_inputs(worker: "Worker") -> None:
    kernel_config = getattr(worker.vllm_config, "kernel_config", None)
    if getattr(kernel_config, "moe_backend", None) != "deep_gemm_mega_moe":
        return

    from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import (
        _PREPARE_MEGAMOE_INPUTS_KERNEL,
    )

    _PREPARE_MEGAMOE_INPUTS_KERNEL.warmup(worker.vllm_config)


def _warm_mtp_rmsnorm_kernels(worker: "Worker") -> None:
    from vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm import (
        _FUSED_MTP_INPUT_RMSNORM_KERNEL,
        _MTP_SHARED_HEAD_RMSNORM_KERNEL,
    )

    _FUSED_MTP_INPUT_RMSNORM_KERNEL.warmup(worker.vllm_config)
    _MTP_SHARED_HEAD_RMSNORM_KERNEL.warmup(worker.vllm_config)


def _warm_attention_rmsnorm_kernel(worker: "Worker") -> None:
    from vllm.models.deepseek_v4.common.ops.fused_qk_rmsnorm import (
        _FUSED_Q_KV_RMSNORM_KERNEL,
    )

    _FUSED_Q_KV_RMSNORM_KERNEL.warmup(worker.vllm_config)


def _warm_compressor_state_kernel(worker: "Worker") -> None:
    from vllm.models.deepseek_v4.common.ops.save_partial_states import (
        _SAVE_PARTIAL_STATES_KERNEL,
    )

    _SAVE_PARTIAL_STATES_KERNEL.warmup(worker.vllm_config)


def _warm_compressor_store_kernels(worker: "Worker") -> None:
    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
        _FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL,
    )

    if has_cutedsl():
        from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
            _SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL,
        )

        _SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL.warmup(worker.vllm_config)
    _FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL.warmup(
        worker.vllm_config
    )


def _warm_router_auxiliary_kernels(worker: "Worker") -> None:
    from vllm.model_executor.layers.fused_moe.router.base_router import (
        _EPLB_MAP_AND_RECORD_KERNEL,
    )

    _EPLB_MAP_AND_RECORD_KERNEL.warmup(worker.vllm_config)

    kernel_config = getattr(worker.vllm_config, "kernel_config", None)
    if not bool(getattr(kernel_config, "enable_bf16x3_router_gemm", False)):
        return
    if not has_cutedsl():
        return
    if not current_platform.is_device_capability_family(100):
        return

    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl import (
        _BF16X3_ROUTER_GEMM_KERNEL,
        _BF16X3_SPLITK_REDUCE_KERNEL,
    )

    _BF16X3_ROUTER_GEMM_KERNEL.warmup(worker.vllm_config)
    _BF16X3_SPLITK_REDUCE_KERNEL.warmup(worker.vllm_config)


def _warm_generic_moe_kernels(worker: "Worker") -> None:
    kernel_config = getattr(worker.vllm_config, "kernel_config", None)
    if getattr(kernel_config, "moe_backend", None) == "deep_gemm_mega_moe":
        return

    from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
        _BATCHED_TRITON_KERNEL,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        _COMPUTE_IDENTITY_KERNEL,
        _FUSED_MOE_TRITON_KERNEL,
    )
    from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import (
        _MOE_FUSED_MUL_SUM_KERNEL,
    )
    from vllm.model_executor.layers.fused_moe.utils import (
        _COUNT_EXPERT_NUM_TOKENS_KERNEL,
        _PACK_TOPK_IDS_WEIGHTS_KERNEL,
        _SWIGLU_LIMIT_PAD_AWARE_KERNEL,
    )

    _FUSED_MOE_TRITON_KERNEL.warmup(worker.vllm_config)
    _BATCHED_TRITON_KERNEL.warmup(worker.vllm_config)
    _COMPUTE_IDENTITY_KERNEL.warmup(worker.vllm_config)

    if getattr(kernel_config, "moe_backend", None) == "emulation":
        from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
            _FUSED_MOE_NVFP4_EMULATION_KERNEL,
        )

        _FUSED_MOE_NVFP4_EMULATION_KERNEL.warmup(worker.vllm_config)

    if getattr(worker.vllm_config, "lora_config", None) is not None:
        from vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe import (
            _TRTLLM_LORA_FINALIZE_KERNEL,
            _TRTLLM_LORA_UNPERMUTE_ACTIVATION_KERNEL,
        )

        _TRTLLM_LORA_UNPERMUTE_ACTIVATION_KERNEL.warmup(worker.vllm_config)
        _TRTLLM_LORA_FINALIZE_KERNEL.warmup(worker.vllm_config)

    parallel_config = getattr(worker.vllm_config, "parallel_config", None)
    if getattr(parallel_config, "all2all_backend", None) == "deepep_v2":
        from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import (
            _GLOBALIZE_RECV_TOPK_IDX_KERNEL,
        )

        _GLOBALIZE_RECV_TOPK_IDX_KERNEL.warmup(worker.vllm_config)
    _MOE_FUSED_MUL_SUM_KERNEL.warmup(worker.vllm_config)
    _COUNT_EXPERT_NUM_TOKENS_KERNEL.warmup(worker.vllm_config)
    _PACK_TOPK_IDS_WEIGHTS_KERNEL.warmup(worker.vllm_config)
    _SWIGLU_LIMIT_PAD_AWARE_KERNEL.warmup(worker.vllm_config)


def _warm_deep_gemm_moe_helper_kernels(worker: "Worker") -> None:
    kernel_config = getattr(worker.vllm_config, "kernel_config", None)
    if getattr(kernel_config, "moe_backend", None) != "deep_gemm_mega_moe":
        return

    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
        _DEEPGEMM_EP_GATHER_KERNEL,
        _DEEPGEMM_EP_SCATTER_KERNEL,
    )

    _DEEPGEMM_EP_SCATTER_KERNEL.warmup(worker.vllm_config)
    _DEEPGEMM_EP_GATHER_KERNEL.warmup(worker.vllm_config)


def _warm_router_topk_kernel(worker: "Worker") -> None:
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import (
        _DSV4_TOPK_KERNEL,
    )

    _DSV4_TOPK_KERNEL.warmup(worker.vllm_config)


def _warm_sparse_utility_kernels(worker: "Worker") -> None:
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        _BUILD_FLASHINFER_MIXED_SPARSE_INDICES_KERNEL,
        _COMPUTE_GLOBAL_TOPK_INDICES_AND_LENS_KERNEL,
        _DEQUANTIZE_AND_GATHER_K_CACHE_KERNEL,
    )

    if has_cutedsl():
        from vllm.models.deepseek_v4.nvidia.ops.dequant_gather_k_cutedsl import (
            _DEQUANT_GATHER_K_CACHE_CUTEDSL_KERNEL,
        )

        _DEQUANT_GATHER_K_CACHE_CUTEDSL_KERNEL.warmup(worker.vllm_config)
    _DEQUANTIZE_AND_GATHER_K_CACHE_KERNEL.warmup(worker.vllm_config)
    _COMPUTE_GLOBAL_TOPK_INDICES_AND_LENS_KERNEL.warmup(worker.vllm_config)
    _BUILD_FLASHINFER_MIXED_SPARSE_INDICES_KERNEL.warmup(worker.vllm_config)


def _warm_output_projection_kernel(worker: "Worker") -> None:
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        _FUSED_INV_ROPE_FP8_QUANT_KERNEL,
    )

    _FUSED_INV_ROPE_FP8_QUANT_KERNEL.warmup(worker.vllm_config)


def _warm_sparse_indexer_helper_kernels(worker: "Worker") -> None:
    from vllm.v1.attention.ops.common import (
        _PACK_SEQ_TRITON_KERNEL,
        _UNPACK_SEQ_TRITON_KERNEL,
    )

    _PACK_SEQ_TRITON_KERNEL.warmup(worker.vllm_config)
    _UNPACK_SEQ_TRITON_KERNEL.warmup(worker.vllm_config)

    parallel_config = getattr(worker.vllm_config, "parallel_config", None)
    if int(getattr(parallel_config, "decode_context_parallel_size", 1) or 1) <= 1:
        return
    if not has_cutedsl():
        return

    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        _PACK_DCP_TOPK_CANDIDATES_KERNEL,
        _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL,
    )

    _PACK_DCP_TOPK_CANDIDATES_KERNEL.warmup(worker.vllm_config)
    _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL.warmup(worker.vllm_config)


def _warm_indexer_q_kernel(worker: "Worker") -> None:
    if has_cutedsl():
        from vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl import (
            _INDEXER_Q_CUTEDSL_KERNEL,
        )

        _INDEXER_Q_CUTEDSL_KERNEL.warmup(worker.vllm_config)
        return

    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
        _FUSED_INDEXER_Q_ROPE_MXFP4_TRITON_KERNEL,
        _FUSED_INDEXER_Q_ROPE_QUANT_TRITON_KERNEL,
    )

    attention_config = getattr(worker.vllm_config, "attention_config", None)
    if bool(getattr(attention_config, "use_fp4_indexer_cache", False)):
        _FUSED_INDEXER_Q_ROPE_MXFP4_TRITON_KERNEL.warmup(worker.vllm_config)
    else:
        _FUSED_INDEXER_Q_ROPE_QUANT_TRITON_KERNEL.warmup(worker.vllm_config)


def deepseek_v4_triton_warmup(worker: "Worker") -> None:
    if worker.model_runner.is_pooling_model:
        return
    if not current_platform.is_cuda():
        return
    if not _is_deepseek_v4_config(worker):
        return

    try:
        _warm_attention_rmsnorm_kernel(worker)
        _warm_compressor_state_kernel(worker)
        _warm_compressor_store_kernels(worker)
        _warm_sparse_utility_kernels(worker)
        _warm_output_projection_kernel(worker)
        _warm_indexer_q_kernel(worker)
        _warm_sparse_indexer_helper_kernels(worker)
        _warm_router_auxiliary_kernels(worker)
        _warm_generic_moe_kernels(worker)
        _warm_deep_gemm_moe_helper_kernels(worker)
        _warm_router_topk_kernel(worker)
        _warm_prepare_megamoe_inputs(worker)
        _warm_mtp_rmsnorm_kernels(worker)
    except Exception:
        logger.warning("Skipping DeepSeek V4 Triton warmup.", exc_info=True)
