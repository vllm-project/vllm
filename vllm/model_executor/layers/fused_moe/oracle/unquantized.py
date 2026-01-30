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
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    swap_w13_to_w31,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

logger = init_logger(__name__)


class UnquantizedMoeBackend(Enum):
    FLASHINFER_CUTLASS = "FlashInfer CUTLASS"
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
    UnquantizedMoeBackend.CPU,
    UnquantizedMoeBackend.XPU,
    UnquantizedMoeBackend.TPU,
    UnquantizedMoeBackend.OOT,
]


def select_unquantized_moe_backend(
    use_ep: bool,
    use_dp: bool,
) -> UnquantizedMoeBackend:
    """
    Select the primary FP8 MoE backend
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    def _make_log_backend(backend: UnquantizedMoeBackend):
        return f"Using {backend.value} backend for Unquantized MoE"

    rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()

    # FlashInfer CUTLASS MoE is only supported on Hopper and later GPUS
    flashinfer_cutlass_moe_enabled = (
        has_flashinfer_cutlass_fused_moe()
        and envs.VLLM_USE_FLASHINFER_MOE_FP16
        and use_ep
        and (not use_dp)
        and current_platform.get_device_capability()[0] >= 9
    )
    if current_platform.is_rocm():
        if rocm_aiter_moe_enabled:
            backend = UnquantizedMoeBackend.AITER
        else:
            backend = UnquantizedMoeBackend.TRITON
    if current_platform.is_cuda():
        if flashinfer_cutlass_moe_enabled:
            backend = UnquantizedMoeBackend.FLASHINFER_CUTLASS
        else:
            if use_ep and (not use_dp):
                logger.info_once(
                    "FlashInfer CUTLASS MoE is available for EP"
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

    return w13_weight, w2_weight


def make_unquantized_moe_kernel(
    backend: UnquantizedMoeBackend,
    quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
) -> tuple[mk.FusedMoEModularKernel | None, bool]:
    use_inplace = True

    if backend in UNSUPPORTED_BACKEND:
        return None, use_inplace

    if backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
        from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
            FlashInferExperts,
        )

        kernel = mk.FusedMoEKernel.make_mk(
            MoEPrepareAndFinalizeNoEP(),
            FlashInferExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )
        use_inplace = False
    elif backend == UnquantizedMoeBackend.AITER:
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            AiterExperts,
        )

        kernel = mk.FusedMoEKernel.make_mk(
            MoEPrepareAndFinalizeNoEP(),
            AiterExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )
    elif backend == UnquantizedMoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe import TritonExperts

        kernel = mk.FusedMoEKernel.make_mk(
            MoEPrepareAndFinalizeNoEP(),
            TritonExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )
    return kernel, use_inplace
