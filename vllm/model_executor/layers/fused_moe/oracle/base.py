# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Abstract base class for MoE kernel oracles.

Each MoE oracle (unquantized / fp8 / nvfp4 / mxfp4 / mxfp8 / int8 /
int_wna16) is responsible for selecting the right MoE kernel backend for a
given (model, hardware, deployment-config) tuple. The current
implementation expresses this responsibility as module-level functions
that follow an informal convention.

This module declares the abstract contract; concrete oracles inherit from
`MoEKernelOracle` and provide the platform-specific behaviour.

This is the first PR in the series suggested by @robertgshaw2-redhat in
PR #37776 (see issue #37753). It intentionally only introduces the ABC;
follow-up PRs migrate each oracle to inherit from it. The single concrete
subclass shipped here (`UnquantizedMoEKernelOracle`) delegates to the
existing module-level functions to keep behaviour bit-identical with
pre-class code.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar

import torch
from torch.nn import Module

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)

BackendT = TypeVar("BackendT", bound=Enum)


class MoEKernelOracle(ABC, Generic[BackendT]):
    """Abstract base for MoE kernel-selection oracles.

    Concrete oracles MUST implement: `backend_enum_cls`,
    `get_priority_backends`, `backend_to_kernel_cls`, `map_backend`,
    `select_backend`, `make_kernel`.

    Concrete oracles MAY override: `convert_to_kernel_format`,
    `make_quant_config`. The base class provides default implementations
    that are appropriate for oracles which do not need them
    (e.g. `make_quant_config` raises on the unquantized oracle).
    """

    @abstractmethod
    def backend_enum_cls(self) -> type[BackendT]:
        """Return the concrete `Enum` class enumerating this oracle's
        backends (e.g. `UnquantizedMoeBackend`, `Fp8MoeBackend`)."""

    @abstractmethod
    def get_priority_backends(self, moe_config: FusedMoEConfig) -> list[BackendT]:
        """Return platform-appropriate backends in priority order for
        this `moe_config`."""

    @abstractmethod
    def backend_to_kernel_cls(self, backend: BackendT) -> type[mk.FusedMoEExperts]:
        """Map a backend enum value to its concrete `FusedMoEExperts`
        subclass."""

    @abstractmethod
    def map_backend(self, runner_backend: MoEBackend) -> BackendT:
        """Map a user-facing `MoEBackend` (from the runner config) to
        this oracle's enum."""

    @abstractmethod
    def select_backend(
        self, moe_config: FusedMoEConfig
    ) -> tuple[BackendT, type[mk.FusedMoEExperts] | None]:
        """Primary entry point: choose the best supported backend for
        the given `moe_config`."""

    @abstractmethod
    def make_kernel(
        self,
        quant_config: FusedMoEQuantConfig,
        moe_config: FusedMoEConfig,
        backend: BackendT,
        experts_cls: type[mk.FusedMoEExperts],
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEKernel:
        """Construct the `FusedMoEKernel` (Prepare/Finalize + Experts
        combinator) for the chosen backend."""

    def convert_to_kernel_format(
        self,
        backend: BackendT,
        layer: Module,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shuffle weights into the layout expected by `backend`.

        Default implementation returns the inputs unchanged. Oracles
        whose backends need weight permutation should override this
        (e.g. `UnquantizedMoEKernelOracle` handles AITER and FlashInfer
        layouts).
        """
        return w13_weight, w2_weight

    def make_quant_config(self, *args, **kwargs) -> FusedMoEQuantConfig:
        """Build a `FusedMoEQuantConfig` for this oracle.

        Quantised oracles (fp8, nvfp4, mxfp4, ...) override this with
        the appropriate signature for their quantisation scheme.
        Unquantised oracles inherit the default, which raises because
        there is no quantisation-specific config to build.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement make_quant_config; "
            "this oracle has no quantisation-specific config to build."
        )
