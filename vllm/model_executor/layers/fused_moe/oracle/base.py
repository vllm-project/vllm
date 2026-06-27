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

Part of the series of issue #37753.
Follow-up PRs are planned.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

logger = init_logger(__name__)

BackendT = TypeVar("BackendT", bound=Enum)


class MoEKernelOracle(ABC, Generic[BackendT]):
    """Abstract base for MoE kernel-selection oracles.

    Concrete oracles MUST implement methods:
    `backend_enum_cls`, `get_priority_backends`,
    `backend_to_kernel_cls`, `map_backend`.

    The base class provides generic implementations of `select_backend`
    and `make_kernel` that use the above methods to pick a backend and
    construct the kernel. Oracles with quirks
    (env-var gates, platform-specific reordering, LoRA forced routing,
    monolithic-only constraints, etc.) override these.

    MAY override `convert_to_kernel_format` and `make_quant_config`
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
        """Map a backend enum value to its candidate `FusedMoEExperts`
        subclasses, ordered by preference. Returning a list of
        multiple experts variants (e.g. Fp8 TrtLlm has both monolithic
        and modular flavours); callers iterate until one's
        `is_supported_config` returns True. Backends with only a
        single variant return a 1-element list.
        """

    @abstractmethod
    def map_backend(self, runner_backend: MoEBackend) -> BackendT:
        """Map a user-facing `MoEBackend` (from the runner config) to
        this oracle's enum."""

    def select_backend(
        self,
        moe_config: FusedMoEConfig,
        weight_key: "QuantKey | None" = None,
        activation_key: "QuantKey | None" = None,
    ) -> tuple[BackendT, type[mk.FusedMoEExperts] | None]:
        """Generic backend selection — try user-explicit override first,
        then iterate the subclass-provided priority list and ask each
        candidate kernel class via ``is_supported_config`` until one
        accepts. Oracles with quirks (env-var gates, platform-specific
        reordering, LoRA forced routing, etc.) override this entirely.
        """
        available = self.get_priority_backends(moe_config)

        activation_format = (
            mk.FusedMoEActivationFormat.BatchedExperts
            if moe_config.moe_parallel_config.use_batched_activation_format
            else mk.FusedMoEActivationFormat.Standard
        )

        oracle_name = type(self).__name__

        def _make_log_backend(backend: BackendT) -> str:
            available_strs = [b.value for b in available]
            return (
                f"Using {backend.value} {oracle_name} backend out "
                f"of potential backends: {available_strs}."
            )

        def _make_log_unsupported(backend: BackendT, reason: str | None) -> str:
            if reason:
                return (
                    f"{oracle_name} backend {backend.value!r} does not support "
                    f"the deployment configuration since {reason}."
                )
            return (
                f"{oracle_name} backend {backend.value!r} does not support "
                "the deployment configuration."
            )

        def _return_or_raise(
            backend: BackendT,
        ) -> tuple[BackendT, type[mk.FusedMoEExperts]]:
            reason: str | None = None
            for k_cls in self.backend_to_kernel_cls(backend):
                supported, reason = k_cls.is_supported_config(
                    k_cls,
                    moe_config,
                    weight_key,
                    activation_key,
                    activation_format,
                )
                if supported:
                    logger.info_once(_make_log_backend(backend))
                    return backend, k_cls
            raise ValueError(_make_log_unsupported(backend, reason))

        # Handle explicit moe_backend from user.
        runner_backend = moe_config.moe_backend
        if runner_backend != "auto":
            requested_backend = self.map_backend(runner_backend)
            return _return_or_raise(requested_backend)

        # Iterate priorities, return first supported.
        last_reason: str | None = None
        for backend in available:
            for k_cls in self.backend_to_kernel_cls(backend):
                supported, reason = k_cls.is_supported_config(
                    k_cls,
                    moe_config,
                    weight_key,
                    activation_key,
                    activation_format,
                )
                last_reason = reason
                if supported:
                    logger.info_once(_make_log_backend(backend))
                    return backend, k_cls
                logger.debug_once(_make_log_unsupported(backend, reason))

        raise NotImplementedError(
            f"No {oracle_name} backend supports the deployment configuration"
            + (f": {last_reason}." if last_reason else ".")
        )

    def make_kernel(
        self,
        quant_config: FusedMoEQuantConfig,
        moe_config: FusedMoEConfig,
        experts_cls: type[mk.FusedMoEExperts],
        backend: BackendT,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEKernel:
        """Generic kernel construction — build the Prepare/Finalize
        stage, instantiate ``experts_cls``, and combine them into a
        ``FusedMoEKernel``. Subclasses with extra construction
        arguments (e.g. ``w4a8`` needs ``b_strides``) or extra
        constraints (e.g. monolithic experts support-only) override this.
        """
        is_monolithic = issubclass(experts_cls, mk.FusedMoEExpertsMonolithic)
        prepare_finalize = maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            routing_tables=routing_tables,
            allow_new_interface=True,
            use_monolithic=is_monolithic,
        )
        assert prepare_finalize is not None

        logger.info_once("Using %s", prepare_finalize.__class__.__name__)

        if (
            prepare_finalize.activation_format
            == mk.FusedMoEActivationFormat.BatchedExperts
        ):
            max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
            assert max_num_tokens is not None
            experts = experts_cls(
                moe_config=moe_config,
                quant_config=quant_config,
                max_num_tokens=max_num_tokens,
                num_dispatchers=prepare_finalize.num_dispatchers(),
            )
        else:
            experts = experts_cls(
                moe_config=moe_config,
                quant_config=quant_config,
            )

        return mk.FusedMoEKernel(prepare_finalize, experts)

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
