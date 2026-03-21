# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend

BackendT = TypeVar("BackendT", bound=Enum)


class MoEKernelOracle(ABC, Generic[BackendT]):
    """Abstract base class for MoE kernel selection oracles.

    Each oracle selects the right MoE kernel for a given quantization
    type (e.g. FP8, NvFP4, MXFP4, MXFP8, unquantized). Subclasses
    implement backend-specific selection, weight conversion, quant
    config creation, and kernel construction.

    Standard oracle operations (subclasses override as needed):
        - ``select_backend``  – choose the best kernel backend
        - ``convert_to_kernel_format`` – shuffle weights for a backend
        - ``make_quant_config`` – build a ``FusedMoEQuantConfig``
        - ``make_kernel`` – construct the ``FusedMoEKernel``

    Note: Method signatures intentionally vary across subclasses because
    each quantization type requires different weight/scale parameters.
    This ABC provides structural guarantees (every oracle implements the
    same set of operations) rather than call-site polymorphism.
    Optional methods (convert_to_kernel_format, make_quant_config,
    make_kernel) raise NotImplementedError by default for oracles that
    delegate these operations (e.g. MXFP8 reuses FP8's kernel logic).
    """

    @abstractmethod
    def backend_to_kernel_cls(
        self,
        backend: BackendT,
    ) -> list[type[mk.FusedMoEExperts]]:
        """Map a backend enum value to its expert class(es)."""
        ...

    @abstractmethod
    def map_backend(self, runner_backend: MoEBackend) -> BackendT:
        """Map a user-specified MoEBackend string to the typed backend enum."""
        ...

    @abstractmethod
    def select_backend(self, *args, **kwargs):
        """Select the best backend for the given configuration.

        Signature varies per oracle – see concrete subclass for details.
        """
        ...

    def convert_to_kernel_format(self, *args, **kwargs):
        """Convert loaded weights into backend-specific kernel format.

        Not all oracles need this (e.g. MXFP8). Default raises
        NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement convert_to_kernel_format."
        )

    def make_quant_config(self, *args, **kwargs):
        """Create FusedMoEQuantConfig for the selected backend.

        Not all oracles need this (e.g. MXFP8, unquantized). Default
        raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement make_quant_config."
        )

    def make_kernel(self, *args, **kwargs):
        """Construct the FusedMoEKernel for the selected backend.

        Not all oracles need this (e.g. MXFP8). Default raises
        NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement make_kernel."
        )
