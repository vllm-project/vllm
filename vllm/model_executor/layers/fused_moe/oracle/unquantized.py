# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch
from torch.nn import Module

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.flashinfer_trtllm_moe import (
    is_supported_config_trtllm_bf16,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    swap_w13_to_w31,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer, has_flashinfer_cutlass_fused_moe

logger = init_logger(__name__)


class UnquantizedMoeBackend(Enum):
    FLASHINFER_TRTLLM = "FlashInfer TRTLLM"
    FLASHINFER_CUTLASS = "FlashInfer CUTLASS"
    SONIC = "Sonic MoE"
    AITER = "ROCm AITER"
    TRITON = "TRITON"
    CPU = "CPU"
    XPU = "XPU"
    TPU = "TPU"
    OOT = "OOT"


# NOTE(zyongye): Unsupported backend means backend
# that is not conform with Modular kernel format.
# We will directly call the kernel for those backend
UNSUPPORTED_BACKEND = [
    UnquantizedMoeBackend.FLASHINFER_TRTLLM,
    UnquantizedMoeBackend.CPU,
    UnquantizedMoeBackend.TPU,
    UnquantizedMoeBackend.OOT,
]


def select_unquantized_moe_backend(
    moe_config: FusedMoEConfig,
    use_ep: bool,
    use_dp: bool,
    is_act_and_mul: bool,
    has_bias: bool,
) -> UnquantizedMoeBackend:
    """
    Select the primary unquantized MoE backend
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    def _make_log_backend(backend: UnquantizedMoeBackend):
        return f"Using {backend.value} backend for Unquantized MoE"

    backend = UnquantizedMoeBackend.TRITON
    rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
    sonic_requested = envs.VLLM_USE_SONIC_MOE
    sonic_supported = False
    if sonic_requested:
        from vllm.model_executor.layers.fused_moe.sonic_moe import (
            is_sonic_moe_supported,
        )

        sonic_supported = is_sonic_moe_supported()
    sonic_enabled = (
        sonic_supported
        and sonic_requested
        and is_act_and_mul
        and not has_bias
        and not use_ep
        and not moe_config.moe_parallel_config.is_sequence_parallel
        and moe_config.experts_per_token <= 16
        and moe_config.in_dtype in (torch.float16, torch.bfloat16)
        and moe_config.activation in ("silu", "silu_and_mul")
    )
    if sonic_requested and sonic_supported and not sonic_enabled:
        if use_ep:
            logger.debug_once(
                "Sonic MoE disabled because expert parallelism is enabled."
            )
        elif has_bias:
            logger.debug_once("Sonic MoE disabled because MoE biases are enabled.")
        elif not is_act_and_mul:
            logger.debug_once("Sonic MoE disabled because is_act_and_mul is False.")
        elif moe_config.moe_parallel_config.is_sequence_parallel:
            logger.debug_once(
                "Sonic MoE disabled because sequence parallelism is enabled."
            )
        elif moe_config.experts_per_token > 16:
            logger.debug_once("Sonic MoE disabled because topk > 16.")
        elif moe_config.in_dtype not in (torch.float16, torch.bfloat16):
            logger.debug_once(
                "Sonic MoE disabled because input dtype is unsupported: %s",
                moe_config.in_dtype,
            )
        elif moe_config.activation not in ("silu", "silu_and_mul"):
            logger.debug_once(
                "Sonic MoE disabled because activation is unsupported: %s",
                moe_config.activation,
            )

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if moe_config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    # Check if FlashInfer TRTLLM BF16 MoE is supported
    trtllm_supported, _ = is_supported_config_trtllm_bf16(
        moe_config=moe_config,
        activation_format=activation_format,
    )
    flashinfer_trtllm_moe_enabled = (
        has_flashinfer()
        and envs.VLLM_USE_FLASHINFER_MOE_FP16
        and trtllm_supported
        and envs.VLLM_FLASHINFER_MOE_BACKEND == "latency"
    )
    # FlashInfer CUTLASS MoE is only supported on Hopper and later GPUS
    flashinfer_cutlass_moe_enabled = (
        has_flashinfer_cutlass_fused_moe()
        and envs.VLLM_USE_FLASHINFER_MOE_FP16
        and use_ep
        and (not use_dp)
        and current_platform.has_device_capability(90)
    )
    if current_platform.is_rocm():
        if rocm_aiter_moe_enabled:
            backend = UnquantizedMoeBackend.AITER
        else:
            backend = UnquantizedMoeBackend.TRITON
    if current_platform.is_cuda():
        if flashinfer_trtllm_moe_enabled:
            backend = UnquantizedMoeBackend.FLASHINFER_TRTLLM
        elif flashinfer_cutlass_moe_enabled:
            backend = UnquantizedMoeBackend.FLASHINFER_CUTLASS
            if trtllm_supported:
                logger.info_once(
                    "FlashInfer TRTLLM MoE is available but not enabled, "
                    "consider setting VLLM_FLASHINFER_MOE_BACKEND=latency "
                    "to enable it for better performance.",
                    scope="local",
                )
        elif sonic_enabled:
            backend = UnquantizedMoeBackend.SONIC
        else:
            if not envs.VLLM_USE_FLASHINFER_MOE_FP16 and trtllm_supported:
                logger.info_once(
                    "FlashInfer TRTLLM MoE is available but not enabled, "
                    "consider setting VLLM_USE_FLASHINFER_MOE_FP16=1 "
                    "and VLLM_FLASHINFER_MOE_BACKEND=latency "
                    "to enable it for better performance.",
                    scope="local",
                )
            elif use_ep and (not use_dp):
                logger.info_once(
                    "FlashInfer MoE is available for EP"
                    " but not enabled, consider setting"
                    " VLLM_USE_FLASHINFER_MOE_FP16=1 to enable it.",
                    scope="local",
                )
            elif use_dp:
                logger.info_once(
                    "FlashInfer CUTLASS MoE is currently not available for DP.",
                    scope="local",
                )
            backend = UnquantizedMoeBackend.TRITON
        if sonic_enabled and backend in (
            UnquantizedMoeBackend.FLASHINFER_TRTLLM,
            UnquantizedMoeBackend.FLASHINFER_CUTLASS,
        ):
            logger.info_once(
                "VLLM_USE_SONIC_MOE=1 is set, but FlashInfer MoE is enabled and was "
                "selected. To force Sonic for experiments, disable FlashInfer MoE "
                "(e.g. VLLM_USE_FLASHINFER_MOE_FP16=0).",
                scope="local",
            )
    if current_platform.is_xpu():
        backend = UnquantizedMoeBackend.XPU
    if current_platform.is_cpu():
        backend = UnquantizedMoeBackend.CPU
    if current_platform.is_tpu():
        backend = UnquantizedMoeBackend.TPU
    if current_platform.is_out_of_tree():
        backend = UnquantizedMoeBackend.OOT

    logger.info_once(_make_log_backend(backend), scope="local")
    return backend


def convert_to_unquantized_kernel_format(
    unquantized_backend: UnquantizedMoeBackend,
    layer: Module,
    w13_weight: torch.Tensor | None = None,
    w2_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if unquantized_backend == UnquantizedMoeBackend.AITER:
        w13_weight, w2_weight = rocm_aiter_ops.shuffle_weights(
            layer.w13_weight.data, layer.w2_weight.data
        )

    elif unquantized_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
        # Swap halves to arrange as [w3; w1] (kernel expectation)
        w13_weight = swap_w13_to_w31(layer.w13_weight.data)
    elif unquantized_backend == UnquantizedMoeBackend.SONIC:
        from vllm.model_executor.layers.fused_moe.sonic_moe import (
            permute_weights_for_sonic,
        )

        w13_weight = permute_weights_for_sonic(layer.w13_weight.data)

    return w13_weight, w2_weight


def make_unquantized_moe_kernel(
    backend: UnquantizedMoeBackend,
    quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
) -> mk.FusedMoEModularKernel | None:
    if backend in UNSUPPORTED_BACKEND:
        return None

    if backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
        from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
            FlashInferExperts,
        )

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            FlashInferExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
            inplace=False,
        )

    elif backend == UnquantizedMoeBackend.AITER:
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            AiterExperts,
        )

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            AiterExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
            inplace=not moe_config.disable_inplace,
        )
    elif backend == UnquantizedMoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe import TritonExperts

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            TritonExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
            inplace=not moe_config.disable_inplace,
        )
    elif backend == UnquantizedMoeBackend.SONIC:
        from vllm.model_executor.layers.fused_moe.sonic_moe import SonicMoeExperts

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            SonicMoeExperts(
                moe_config=moe_config,
                quant_config=quant_config,
                weights_prepermuted=True,
            ),
            inplace=False,
        )
    elif backend == UnquantizedMoeBackend.XPU:
        from vllm.model_executor.layers.fused_moe import XPUExperts

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            XPUExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
            inplace=not moe_config.disable_inplace,
        )
    return kernel
