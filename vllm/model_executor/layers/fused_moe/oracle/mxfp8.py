# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    backend_to_kernel_cls,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.utils.humming import prioritize_humming
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
    kMxfp8Static,
)

logger = init_logger(__name__)

# Ordered by priority.
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
    Fp8MoeBackend.TRITON_MXFP8,
    Fp8MoeBackend.EMULATION,
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
        from vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe import (
            Mxfp8NativeTritonExperts,
        )

        return [Mxfp8NativeTritonExperts]
    if backend == Fp8MoeBackend.EMULATION:
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


def convert_to_mxfp8_moe_kernel_format(
    backend: Fp8MoeBackend,
    layer: "RoutedExperts",
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_input_scale: torch.Tensor | None = None,
    w2_input_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform MXFP8 weights and per-block scales to the backend-specific format."""
    if backend == Fp8MoeBackend.HUMMING:
        from vllm.model_executor.layers.quantization.utils.humming import (
            convert_to_humming_moe_kernel_format,
        )
        from vllm.model_executor.layers.quantization.utils.layer_utils import (
            replace_parameter,
        )

        for prefix, weight, scale, input_scale in (
            ("w13", w13, w13_scale, w13_input_scale),
            ("w2", w2, w2_scale, w2_input_scale),
        ):
            if hasattr(layer, f"{prefix}_weight"):
                replace_parameter(layer, f"{prefix}_weight", weight)
            else:
                layer.register_parameter(
                    f"{prefix}_weight",
                    torch.nn.Parameter(weight, requires_grad=False),
                )

            if hasattr(layer, f"{prefix}_weight_scale"):
                replace_parameter(layer, f"{prefix}_weight_scale", scale)
            else:
                layer.register_parameter(
                    f"{prefix}_weight_scale",
                    torch.nn.Parameter(scale, requires_grad=False),
                )

            if input_scale is not None:
                if hasattr(layer, f"{prefix}_input_scale"):
                    replace_parameter(layer, f"{prefix}_input_scale", input_scale)
                else:
                    layer.register_parameter(
                        f"{prefix}_input_scale",
                        torch.nn.Parameter(input_scale, requires_grad=False),
                    )

        convert_to_humming_moe_kernel_format(
            layer, quant_config={"quant_method": "modelopt", "quant_algo": "mxfp8"}
        )
        return (
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )

    # Defer remaining backends (DeepGEMM, Marlin, Aiter) to fp8 conversion logic
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        convert_to_fp8_moe_kernel_format,
    )

    return convert_to_fp8_moe_kernel_format(
        fp8_backend=backend,
        layer=layer,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_input_scale=w13_input_scale,
        w2_input_scale=w2_input_scale,
    )


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
    backends = prioritize_humming(list(_SUPPORTED_BACKENDS))
    for backend in backends:
        try:
            experts_cls = _select_kernel_cls(backend, config)
        except ValueError:
            continue
        logger.info_once("Using '%s' MxFp8 MoE backend.", backend.value)
        return backend, experts_cls

    # TODO: add debug log with reason.
    raise ValueError("No MXFP8 MoE backends available.")
