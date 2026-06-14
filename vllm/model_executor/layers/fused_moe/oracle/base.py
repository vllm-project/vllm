# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for MoE kernel selection oracles.

Each quantization scheme (unquantized, fp8, nvfp4, mxfp4, mxfp8, int8,
w4a8, w4a8_int8, int_wna16) has an oracle that selects the best MoE kernel
backend for a given deployment configuration.

This module provides :class:`MoEKernelOracle`, an abstract base class that
standardizes the 4 core operations every oracle must implement, plus shared
helper methods that were previously copy-pasted across oracle modules.

Subclassing guide
-----------------
1. Define a backend enum (e.g. ``Fp8MoeBackend``).
2. Subclass ``MoEKernelOracle[YourBackendEnum]``.
3. Implement the four abstract methods.
4. Keep the existing module-level wrapper functions that delegate to the
   oracle instance for full backward compatibility.

The module-level wrappers are intentionally preserved — zero changes are
required from external callers.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from torch.nn import Module

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)

logger = init_logger(__name__)

#: Type variable bound to a backend enum (e.g. UnquantizedMoeBackend).
BackendT = TypeVar("BackendT")


class MoEKernelOracle(ABC, Generic[BackendT]):
    """Abstract base class for MoE kernel selection oracles.

    Each quantization scheme inherits from this class and implements the
    four core operations: backend selection, weight format conversion,
    quant config construction, and kernel construction.

    Type Parameters
    ---------------
    BackendT : Enum
        The oracle-specific backend enum (e.g. ``Fp8MoeBackend``).
    """

    # ------------------------------------------------------------------
    # Abstract methods — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def select_backend(
        self,
        moe_config: FusedMoEConfig,
    ) -> tuple[BackendT, type[mk.FusedMoEExperts] | None]:
        """Select the best MoE kernel backend for the given config.

        Returns
        -------
        tuple[BackendT, type[mk.FusedMoEExperts] | None]
            The selected backend and its corresponding expert kernel class.
            ``None`` for platforms without a modular-kernel implementation
            (CPU / TPU / OOT).
        """
        ...

    @abstractmethod
    def convert_to_kernel_format(
        self,
        backend: BackendT,
        layer: Module,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert weights to the format expected by the selected kernel.

        This may include shuffling, swizzling, or block-layout conversion.
        """
        ...

    @abstractmethod
    def make_quant_config(
        self,
        moe_config: FusedMoEConfig,
        backend: BackendT,
    ) -> FusedMoEQuantConfig:
        """Build the quantization config for the selected backend."""
        ...

    @abstractmethod
    def make_kernel(
        self,
        quant_config: FusedMoEQuantConfig,
        moe_config: FusedMoEConfig,
        backend: BackendT,
        experts_cls: type[mk.FusedMoEExperts],
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEKernel:
        """Construct the fused MoE kernel from its components."""
        ...

    # ------------------------------------------------------------------
    # Helpers — shared across all oracles
    # ------------------------------------------------------------------

    @staticmethod
    def _oracle_name() -> str:
        """Human-readable name for log messages.

        Override in subclasses to return e.g. ``"FP8"`` or ``"Unquantized"``.
        """
        return "MoE"

    @classmethod
    def _make_log_backend(
        cls,
        backend: BackendT,
        available_backends: list[BackendT] | None = None,
    ) -> str:
        """Log message emitted when a backend is selected."""
        name = cls._oracle_name()
        if available_backends:
            available_strs = [b.value for b in available_backends]
            return (
                f"Using {backend.value} {name} MoE backend out "
                f"of potential backends: {available_strs}."
            )
        return f"Using {backend.value} {name} MoE backend."

    @classmethod
    def _make_log_unsupported(
        cls,
        backend: BackendT,
        reason: str | None,
    ) -> str:
        """Log message emitted when a backend is unsupported."""
        name = cls._oracle_name()
        if reason:
            return (
                f"{name} MoE backend {backend.value} does not support the "
                f"deployment configuration since {reason}."
            )
        return (
            f"{name} MoE backend '{backend.value}' does not support the "
            "deployment configuration."
        )

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def backend_to_kernel_cls(
        backend: BackendT,
    ) -> type[mk.FusedMoEExperts] | list[type[mk.FusedMoEExperts]]:
        """Map a backend enum value to its expert kernel class(es).

        Some oracles return a single class; others (fp8, int_wna16) return
        a list of candidate classes that are tried in order.
        """
        ...

    @staticmethod
    @abstractmethod
    def map_backend(runner_backend: MoEBackend) -> BackendT:
        """Map a user-facing ``MoEBackend`` string to the oracle's enum."""
        ...
