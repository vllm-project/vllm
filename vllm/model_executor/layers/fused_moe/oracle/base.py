# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from torch.nn import Module

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)


class MoEKernelOracle(ABC):
    """
    Abstract base class for MoE Oracles.
    An oracle encapsulates the logic for selecting the most optimal
    kernel backend and preparing weights for a specific quantization format.
    """

    @classmethod
    @abstractmethod
    def get_priority_backends(
        cls,
        moe_config: FusedMoEConfig,
        *args, **kwargs
    ) -> list[Enum]:
        """Return a list of supported backends in priority order."""
        pass

    @classmethod
    @abstractmethod
    def backend_to_kernel_cls(
        cls,
        backend: Enum,
    ) -> list[type[mk.FusedMoEExperts]] | type[mk.FusedMoEExperts]:
        """Map a backend to its corresponding FusedMoEExperts class."""
        pass

    @classmethod
    @abstractmethod
    def map_backend(cls, runner_backend: MoEBackend) -> Enum:
        """Map a string identifier from configuration to the Enum backend."""
        pass

    @classmethod
    @abstractmethod
    def select_backend(
        cls,
        config: FusedMoEConfig,
        *args, **kwargs
    ) -> tuple[Enum, type[mk.FusedMoEExperts] | None]:
        """Select the most optimal backend based on configuration."""
        pass

    @classmethod
    @abstractmethod
    def convert_to_kernel_format(
        cls,
        backend: Enum,
        layer: Module,
        *args, **kwargs
    ) -> Any:
        """Convert weights into the format required by the selected kernel."""
        pass

    @classmethod
    @abstractmethod
    def make_quant_config(
        cls,
        backend: Enum,
        *args, **kwargs
    ) -> FusedMoEQuantConfig | None:
        """Construct a FusedMoEQuantConfig instance for the selected backend."""
        pass

    @classmethod
    @abstractmethod
    def make_kernel(
        cls,
        moe_quant_config: FusedMoEQuantConfig | None,
        moe_config: FusedMoEConfig,
        backend: Enum,
        experts_cls: type[mk.FusedMoEExperts],
        *args, **kwargs
    ) -> mk.FusedMoEKernel:
        """Instantiate the FusedMoEKernel for the selected backend."""
        pass
