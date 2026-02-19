# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_triton_kernels

logger = init_logger(__name__)


class Mxfp4MoeBackend(Enum):
    NONE = "None"
    SM100_FI_MXFP4_MXFP8_TRTLLM = "SM100_FI_MXFP4_MXFP8_TRTLLM"
    SM100_FI_MXFP4_MXFP8_TRTLLM_MONOLITHIC = "SM100_FI_MXFP4_MXFP8_TRTLLM_MONOLITHIC"
    SM100_FI_MXFP4_MXFP8_CUTLASS = "SM100_FI_MXFP4_MXFP8_CUTLASS"
    SM100_FI_MXFP4_BF16 = "SM100_FI_MXFP4_BF16"
    SM100_FI_MXFP4_BF16_MONOLOTHIC = "SM100_FI_MXFP4_BF16_MONILITHIC"
    SM90_FI_MXFP4_BF16 = "SM90_FI_MXFP4_BF16"
    BATCHED_MARLIN = "BATCHED_MARLIN"
    MARLIN = "MARLIN"
    TRITON = "TRITON"
    TRITON_MONOLITHIC = "TRITON_MONOLITHIC"
    TRITON_UNFUSED = "TRITON_UNFUSED"
    XPU = "XPU"


def backend_to_kernel_cls(
    backend: Mxfp4MoeBackend,
) -> type[mk.FusedMoEPermuteExpertsUnpermute]: ...


def select_mxfp4_moe_backend(
    config: FusedMoEConfig,
) -> tuple[Mxfp4MoeBackend, type[mk.FusedMoEPermuteExpertsUnpermute] | None]:
    """
    Select the primary MXFP4 MoE backend.
    Note: Shape-specific fallbacks may still occur at runtime.
    """
    k_cls: type[mk.FusedMoEPermuteExpertsUnpermute] | None = None

    # If FlashInfer is not available, try either Marlin or Triton
    triton_kernels_supported = (
        has_triton_kernels()
        # NOTE: triton_kernels are only confirmed to work on SM90 and SM100
        # SM110 fails with this error: https://github.com/vllm-project/vllm/issues/29317
        # SM120 needs this fix: https://github.com/triton-lang/triton/pull/8498
        and (9, 0) <= current_platform.get_device_capability() < (11, 0)
    )

    if config.is_lora_enabled:
        if not current_platform.is_cuda():
            raise NotImplementedError("Mxfp4 LoRA only supported on CUDA Platform.")

        if envs.VLLM_MXFP4_USE_MARLIN is False and triton_kernels_supported:
            logger.info_once("Using Triton backend for mxfp4 lora")
            return Mxfp4MoeBackend.TRITON_UNFUSED, backend_to_kernel_cls(
                Mxfp4MoeBackend.TRITON_UNFUSED
            )

        logger.info_once("Using Marlin backend for mxfp4 lora")
        return Mxfp4MoeBackend.MARLIN, backend_to_kernel_cls(Mxfp4MoeBackend.MARLIN)

    # NOTE: the kernels are selected in the following order.
    AVAILABLE_BACKENDS = [
        Mxfp4MoeBackend.SM90_FI_MXFP4_BF16,
        Mxfp4MoeBackend.SM100_FI_MXFP4_BF16,
        Mxfp4MoeBackend.SM100_FI_MXFP4_MXFP8_TRTLLM,
        Mxfp4MoeBackend.SM100_FI_MXFP4_MXFP8_CUTLASS,
        Mxfp4MoeBackend.MARLIN,
        Mxfp4MoeBackend.BATCHED_MARLIN,
        Mxfp4MoeBackend.TRITON,
        Mxfp4MoeBackend.TRITON_MONOLITHIC,
        Mxfp4MoeBackend.TRITON_UNFUSED,
        Mxfp4MoeBackend.XPU,
    ]

    # NOTE(zyongye): See similar comments in fp8.py
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: Mxfp4MoeBackend):
        available_backend_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using {backend.value} Mxfp4 MoE backend out "
            f"of potential backends: {available_backend_strs}."
        )

    def _make_log_unsupported(backend: Mxfp4MoeBackend, reason: str | None) -> str:
        if reason:
            return (
                f"Mxfp4 MoE backend {backend.value} does not support the "
                f"deployment configuration since {reason}."
            )
        else:
            return (
                f"Mxfp4 MoE backend '{backend.value}' does not support the "
                "deployment configuration."
            )

    def _return_or_raise(
        backend: Mxfp4MoeBackend,
        config: FusedMoEConfig,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[Mxfp4MoeBackend, type[mk.FusedMoEPermuteExpertsUnpermute]]:
        k_cls = backend_to_kernel_cls(backend)
        supported, reason = k_cls.is_supported_config(
            k_cls, config, None, None, activation_format
        )
        if supported:
            logger.info_once(_make_log_backend(backend), scope="local")
            return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    # Handle explicit FlashInfer MXFP4 BF16 configuration.
    if envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16:
        if current_platform.is_device_capability(90):
            backend = Mxfp4MoeBackend.SM90_FI_MXFP4_BF16
            return _return_or_raise(backend, config, activation_format)
        elif current_platform.is_device_capability_family(100):
            backend = Mxfp4MoeBackend.SM100_FI_MXFP4_BF16
            return _return_or_raise(backend, config, activation_format)

    # Handle explicit FlashInfer MXFP4 MXFP8 TRTLLM configuration.
    if envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8:
        backend = Mxfp4MoeBackend.SM100_FI_MXFP4_MXFP8_TRTLLM
        return _return_or_raise(backend, config, activation_format)

    # Handle explicit FlashInfer MXFP4 MXFP8 CUTLASS configuration.
    if envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS:
        backend = Mxfp4MoeBackend.SM100_FI_MXFP4_MXFP8_CUTLASS
        return _return_or_raise(backend, config, activation_format)

    # Handle explicit Marlin MXFP4 configuration.
    if envs.is_set("VLLM_MXFP4_USE_MARLIN"):
        if not envs.VLLM_MXFP4_USE_MARLIN:
            AVAILABLE_BACKENDS.remove(Mxfp4MoeBackend.MARLIN)
            AVAILABLE_BACKENDS.remove(Mxfp4MoeBackend.BATCHED_MARLIN)
        else:
            backend = Mxfp4MoeBackend.MARLIN
            return _return_or_raise(backend, config, activation_format)

    # Select kernels in order of backend.
    for backend in AVAILABLE_BACKENDS:
        k_cls = backend_to_kernel_cls(backend)
        supported, reason = k_cls.is_supported_config(
            k_cls,
            config,
            None,
            None,
            activation_format,
        )

        if supported:
            logger.info_once(_make_log_backend(backend), scope="local")
            return backend, k_cls
        else:
            logger.debug_once(_make_log_unsupported(backend, reason), scope="local")

    if current_platform.is_cuda() or current_platform.is_rocm():
        raise NotImplementedError(
            "No MXFP4 MoE backend supports the deployment configuration."
        )

    return Mxfp4MoeBackend.NONE, None


def convert_to_mxfp4_moe_kernel_format(): ...


def make_mxfp4_moe_quant_config(): ...


def make_mxfp4_moe_kernel(): ...
