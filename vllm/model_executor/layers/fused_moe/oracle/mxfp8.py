# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle.base import MoEKernelOracle
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    backend_to_kernel_cls,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
    kMxfp8Static,
)

logger = init_logger(__name__)

_SUPPORTED_BACKENDS: frozenset[Fp8MoeBackend] = frozenset(
    {
        Fp8MoeBackend.FLASHINFER_TRTLLM,
    }
)

_BACKEND_NAME_MAP: dict[str, Fp8MoeBackend] = {
    "flashinfer_trtllm": Fp8MoeBackend.FLASHINFER_TRTLLM,
}


class Mxfp8MoEKernelOracle(MoEKernelOracle[Fp8MoeBackend]):
    """Oracle for MXFP8 MoE kernel selection."""

    @property
    def quant_type_name(self) -> str:
        return "MxFp8"

    def backend_to_kernel_cls(
        self,
        backend: Fp8MoeBackend,
    ) -> list[type[mk.FusedMoEExperts]]:
        return backend_to_kernel_cls(backend)

    def map_backend(self, runner_backend: str) -> Fp8MoeBackend:
        backend = _BACKEND_NAME_MAP.get(runner_backend)
        if backend is None:
            raise ValueError(
                f"moe_backend='{runner_backend}' is not supported for "
                f"MXFP8 MoE. Expected one of "
                f"{list(_BACKEND_NAME_MAP.keys())}."
            )
        return backend

    def _select_kernel_cls(
        self,
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
        for cls in self.backend_to_kernel_cls(backend):
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

    def select_backend(
        self,
        config: FusedMoEConfig,
    ) -> tuple[Fp8MoeBackend, type[mk.FusedMoEExperts]]:
        """Select the MXFP8 MoE backend and the best expert class.

        Returns:
            A tuple of (fp8_backend, experts_cls).
        """
        if config.is_lora_enabled:
            raise NotImplementedError("LoRA is not supported for MXFP8 MoE.")

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
            return backend, self._select_kernel_cls(backend, config)

        # Auto-select: pick the first supported backend.
        for backend in _SUPPORTED_BACKENDS:
            logger.info_once("Using '%s' MxFp8 MoE backend.", backend.value)
            return backend, self._select_kernel_cls(backend, config)

        raise ValueError("No MXFP8 MoE backends available.")


_oracle = Mxfp8MoEKernelOracle()


def _select_kernel_cls(
    backend: Fp8MoeBackend,
    config: FusedMoEConfig,
) -> type[mk.FusedMoEExperts]:
    """Select the first supported expert class for the MXFP8 config."""
    return _oracle._select_kernel_cls(backend, config)


def select_mxfp8_moe_backend(
    config: FusedMoEConfig,
) -> tuple[Fp8MoeBackend, type[mk.FusedMoEExperts]]:
    """Select the MXFP8 MoE backend and the best expert class.

    Returns:
        A tuple of (fp8_backend, experts_cls).
    """
    return _oracle.select_backend(config)
