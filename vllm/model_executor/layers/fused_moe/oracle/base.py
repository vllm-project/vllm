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
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

BackendT = TypeVar("BackendT", bound=Enum)


def make_fused_moe_experts(
    prepare_finalize: mk.FusedMoEPrepareAndFinalize,
    experts_cls: type[mk.FusedMoEExperts],
    moe_config: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig,
    **kwargs: Any,
) -> mk.FusedMoEExperts:
    """Validate the activation contract before constructing the experts."""
    experts_activation_format = experts_cls.activation_format()
    prepare_finalize_activation_format = prepare_finalize.activation_format
    if experts_activation_format != prepare_finalize_activation_format:
        raise ValueError(
            f"{experts_cls.__name__} expects {experts_activation_format.name} "
            "activations, but "
            f"{prepare_finalize.__class__.__name__} produces "
            f"{prepare_finalize_activation_format.name} activations."
        )

    if experts_activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        kwargs["max_num_tokens"] = max_num_tokens
        kwargs["num_dispatchers"] = prepare_finalize.num_dispatchers()

    return experts_cls(
        moe_config=moe_config,
        quant_config=quant_config,
        **kwargs,
    )


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
    def backend_to_kernel_cls(
        self, backend: BackendT
    ) -> list[type[mk.FusedMoEExperts]]:
        """Map a backend enum value to its concrete `FusedMoEExperts`
        subclasses, in selection priority order."""

    @abstractmethod
    def map_backend(self, runner_backend: MoEBackend) -> BackendT:
        """Map a user-facing `MoEBackend` (from the runner config) to
        this oracle's enum."""

    @abstractmethod
    def select_backend(
        self,
        moe_config: FusedMoEConfig,
        weight_key: "QuantKey | None" = None,
        activation_key: "QuantKey | None" = None,
    ) -> tuple[BackendT, type[mk.FusedMoEExperts] | None]:
        """Primary entry point: choose the best supported backend for
        the given `moe_config`.

        `weight_key` / `activation_key` carry the quantization scheme of
        the weights and activations and are consumed by quantized oracles
        (fp8, nvfp4, int8, ...) to disambiguate backends. The unquantized
        oracle ignores them. Subclasses with additional selection inputs
        (e.g. int_wna16 needs `weight_bits`, fp8 needs
        `allow_vllm_cutlass`) widen the signature in their override; a
        per-oracle config object is the longer-term target tracked in
        the #37753 follow-up PRs.
        """

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
        moe_config: FusedMoEConfig,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shuffle weights into the layout expected by `backend`.

        Default implementation returns the inputs unchanged. Oracles
        whose backends need weight permutation should override this
        (e.g. `UnquantizedMoEKernelOracle` handles AITER and FlashInfer
        layouts).

        `moe_config` carries MoE-layer state (e.g. `is_act_and_mul`)
        that the conversion needs without coupling the oracle to a
        `Module` reference. Quantized oracles whose conversion
        additionally needs scales / zero-points / block shapes will
        override with a wider signature (and ultimately a per-oracle
        config object — tracked in the #37753 follow-up PRs).
        """
        return w13_weight, w2_weight

    def make_quant_config(self, *args, **kwargs) -> FusedMoEQuantConfig:
        """Build a `FusedMoEQuantConfig` for this oracle.

        Quantized oracles (fp8, nvfp4, mxfp4, ...) override this with
        the appropriate signature for their quantization scheme.
        Unquantized oracles inherit the default, which raises because
        there is no quantization-specific config to build.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement make_quant_config; "
            "this oracle has no quantization-specific config to build."
        )
