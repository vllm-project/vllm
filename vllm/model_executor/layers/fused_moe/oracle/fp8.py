# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
    get_flashinfer_moe_backend,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer_moe
from vllm.utils.import_utils import has_deep_gemm

logger = init_logger(__name__)


class Fp8MoeBackend(Enum):
    NONE = 0
    FLASHINFER_TRTLLM = 1
    FLASHINFER_CUTLASS = 2
    DEEPGEMM = 3
    MARLIN = 4
    TRITON = 5
    AITER = 6


def get_fp8_moe_backend(
    block_quant: bool,
    tp_size: int,
    with_lora_support: bool,
) -> Fp8MoeBackend | None:
    """
    Select the primary FP8 MoE backend
    Note: Shape-specific fallbacks may still occur at runtime.
    """
    # TODO(rob): update so that each mk expresses supported features.
    # TODO(rob): update so that we have priority order for each.

    if current_platform.is_xpu():
        return None
    if with_lora_support:
        return Fp8MoeBackend.TRITON

    def _make_log_backend(backend_name: str):
        return f"Using {backend_name} backend for FP8 MoE"

    # Prefer FlashInfer backends on supported GPUs; allow SM90 and SM100.
    if (
        current_platform.is_cuda()
        and (
            current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability(90)
        )
        and envs.VLLM_USE_FLASHINFER_MOE_FP8
        and has_flashinfer_moe()
    ):
        backend = get_flashinfer_moe_backend()
        if backend == FlashinferMoeBackend.TENSORRT_LLM:
            logger.info_once(_make_log_backend("FlashInfer TRTLLM"))
            return Fp8MoeBackend.FLASHINFER_TRTLLM
        else:
            if block_quant and current_platform.is_device_capability_family(100):
                raise ValueError(
                    "FlashInfer FP8 MoE throughput backend does not "
                    "support block quantization. Please use "
                    "VLLM_FLASHINFER_MOE_BACKEND=latency "
                    "instead."
                )
            logger.info_once(_make_log_backend("FlashInfer CUTLASS"))
            return Fp8MoeBackend.FLASHINFER_CUTLASS

    # weight-only path for older GPUs without native FP8
    use_marlin = (
        not current_platform.has_device_capability(89)
        or envs.VLLM_TEST_FORCE_FP8_MARLIN
    )
    if current_platform.is_rocm():
        use_marlin = False
    if use_marlin:
        logger.info_once(_make_log_backend("Marlin"))
        return Fp8MoeBackend.MARLIN

    # Determine if we should use DeepGEMM with block-quantized weights:
    # - If explicitly set by user, respect their choice
    # - If not explicitly set (default), disable when TP size is >= 8
    moe_use_deep_gemm = envs.VLLM_MOE_USE_DEEP_GEMM
    if not envs.is_set("VLLM_MOE_USE_DEEP_GEMM") and tp_size >= 8:
        moe_use_deep_gemm = False
        logger.info_once(
            "DeepGEMM MoE is disabled by default when TP size is >= 8. "
            "Set VLLM_MOE_USE_DEEP_GEMM=1 to enable it.",
            scope="local",
        )

    if envs.VLLM_USE_DEEP_GEMM and moe_use_deep_gemm and block_quant:
        if not has_deep_gemm():
            logger.warning_once(
                "DeepGEMM backend requested but not available.", scope="local"
            )
        elif is_deep_gemm_supported():
            logger.info_once(_make_log_backend("DeepGEMM"), scope="local")
            return Fp8MoeBackend.DEEPGEMM

    if envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_MOE:
        logger.info_once(_make_log_backend("ROCm AITER"), scope="local")
        return Fp8MoeBackend.AITER

    # default to Triton
    logger.info_once(_make_log_backend("Triton"))
    return Fp8MoeBackend.TRITON
