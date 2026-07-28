# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    backend_to_kernel_cls,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

_SUPPORTED_BACKENDS = (
    Fp8MoeBackend.FLASHINFER_TRTLLM,
    Fp8MoeBackend.DEEPGEMM,
    Fp8MoeBackend.MARLIN,
    Fp8MoeBackend.XPU,
    # AITER FlyDSL (gfx950): auto-picked by select_mxfp8_moe_backend when
    # is_supported_config passes (gfx950 + flydsl installed + not EP). On other
    # devices / no flydsl / EP it is skipped and native is used.
    Fp8MoeBackend.AITER_MXFP8,
    Fp8MoeBackend.HUMMING,
)

_BACKEND_NAME_MAP: dict[str, Fp8MoeBackend] = {
    "flashinfer_trtllm": Fp8MoeBackend.FLASHINFER_TRTLLM,
    "deep_gemm": Fp8MoeBackend.DEEPGEMM,
    "marlin": Fp8MoeBackend.MARLIN,
    "xpu": Fp8MoeBackend.XPU,
    "aiter": Fp8MoeBackend.AITER_MXFP8,
    "triton": Fp8MoeBackend.TRITON_MXFP8,
    "humming": Fp8MoeBackend.HUMMING,
}


def _mxfp8_backend_to_kernel_cls(
    backend: Fp8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    """Resolve the MXFP8 expert classes for a backend.

    DeepGEMM resolves directly to ``DeepGemmExperts`` (not the
    ``TritonOrDeepGemmExperts`` wrapper, whose Triton fallback cannot handle the
    MXFP8 1x32 scheme); all other backends defer to the FP8 resolver.
    """
    if backend == Fp8MoeBackend.DEEPGEMM:
        from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
            DeepGemmExperts,
        )

        return [DeepGemmExperts]
    if backend == Fp8MoeBackend.AITER_MXFP8:
        from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_moe import (
            AiterMxfp8Experts,
        )

        return [AiterMxfp8Experts]
    if backend == Fp8MoeBackend.TRITON_MXFP8:
        # Explicit ``--moe-backend triton``: the Triton mxfp8 path, i.e.
        # dot_scaled on MX-capable HW (gfx950) and BF16 emulation otherwise.
        # Mirrors the ROCm auto-fallback in ``_select_rocm_mxfp8_backend``.
        if current_platform.supports_mx():
            from vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe import (
                Mxfp8NativeTritonExperts,
            )

            return [Mxfp8NativeTritonExperts]
        from vllm.model_executor.layers.fused_moe.experts.mxfp8_emulation_moe import (
            Mxfp8EmulationTritonExperts,
        )

        return [Mxfp8EmulationTritonExperts]
    return backend_to_kernel_cls(backend)


def _select_kernel_cls(
    backend: Fp8MoeBackend,
    config: FusedMoEConfig,
) -> type[mk.FusedMoEExperts]:
    """Select the first supported expert class for the MXFP8 config."""
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )
    last_reason: str | None = None
    for cls in _mxfp8_backend_to_kernel_cls(backend):
        supported, reason = cls.is_supported_config(
            cls,
            config,
            kMxfp8Static,
            kMxfp8Dynamic,
            activation_format,
        )
        if supported:
            return cls
        last_reason = reason
    raise ValueError(
        f"No supported MXFP8 expert class for {backend.value}: {last_reason}"
    )


def _select_rocm_mxfp8_backend() -> tuple[Fp8MoeBackend, type[mk.FusedMoEExperts]]:
    """ROCm fallback when no auto-selected MXFP8 backend is available.

    The aiter FlyDSL backend (``AITER_MXFP8``) is auto-picked earlier by
    ``select_mxfp8_moe_backend`` via ``_SUPPORTED_BACKENDS`` when usable, or
    explicitly via ``--moe-backend aiter``; this fallback handles the rest
    (native dot_scaled on gfx950, else BF16 emulation).
    """

    if current_platform.supports_mx():
        from vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe import (
            Mxfp8NativeTritonExperts,
        )

        logger.info_once("Using native CDNA4 (gfx950) MXFP8 dot_scaled MoE backend.")
        return Fp8MoeBackend.TRITON_MXFP8, Mxfp8NativeTritonExperts

    from vllm.model_executor.layers.fused_moe.experts.mxfp8_emulation_moe import (
        Mxfp8EmulationTritonExperts,
    )

    logger.info_once(
        "No native MXFP8 MoE backend available on this device; "
        "MXFP8 weights will be dequantized to BF16 once at load time and the "
        "MoE will run in BF16 (no per-step dequant)."
    )
    return Fp8MoeBackend.EMULATION, Mxfp8EmulationTritonExperts


def select_mxfp8_moe_backend(
    config: FusedMoEConfig,
) -> tuple[Fp8MoeBackend, type[mk.FusedMoEExperts]]:
    """Select the MXFP8 MoE backend and the best expert class.

    Returns:
        A tuple of (fp8_backend, experts_cls).
    """

    runner_backend = config.moe_backend
    if runner_backend != "auto":
        backend = _BACKEND_NAME_MAP.get(runner_backend)
        if backend is None:
            raise ValueError(
                f"moe_backend='{runner_backend}' is not supported for "
                f"MXFP8 MoE. Expected one of "
                f"{list(_BACKEND_NAME_MAP.keys())}."
            )
        logger.info_once(
            "Using '%s' MxFp8 MoE backend (user-requested).",
            backend.value,
        )
        return backend, _select_kernel_cls(backend, config)

    # Auto-select: pick the first supported backend.
    for backend in _SUPPORTED_BACKENDS:
        try:
            experts_cls = _select_kernel_cls(backend, config)
        except ValueError:
            continue
        logger.info_once("Using '%s' MxFp8 MoE backend.", backend.value)
        return backend, experts_cls

    # simplify the logic for rocm, refactor later when more backends are supported
    if current_platform.is_rocm():
        return _select_rocm_mxfp8_backend()

    raise ValueError("No MXFP8 MoE backends available.")
