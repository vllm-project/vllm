# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Abstract base class for MoE kernel oracles."""

from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

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

# Sentinel for the per-call `activation_key` override: `None` is a legitimate
# key (BF16 activations), so absence needs its own value. Same idiom as
# `vllm.config.quantization._UNSET`.
_UNSET = cast("QuantKey | None", object())

BackendT = TypeVar("BackendT", bound=Enum)


class MoEKernelOracle(ABC, Generic[BackendT]):
    """Abstract base for MoE kernel-selection oracles.

    Concrete oracles provide `name` and `backend_to_kernel_cls` today; the
    remaining methods below are the #37753 migration target and are still
    implemented per module, so they are declared but not enforced.
    """

    # Display name used by the shared selection helpers below.
    name: ClassVar[str] = "MoE"

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        weight_key: "QuantKey | None" = None,
        activation_key: "QuantKey | None" = None,
        activation_format: mk.FusedMoEActivationFormat = (
            mk.FusedMoEActivationFormat.Standard
        ),
    ) -> None:
        # Deployment inputs shared by every backend probe of one selection.
        self.moe_config = moe_config
        self.weight_key = weight_key
        self.activation_key = activation_key
        self.activation_format = activation_format

    def backend_enum_cls(self) -> type[BackendT]:
        """Return the concrete `Enum` class enumerating this oracle's
        backends (e.g. `UnquantizedMoeBackend`, `Fp8MoeBackend`)."""
        raise NotImplementedError

    def get_priority_backends(self, moe_config: FusedMoEConfig) -> list[BackendT]:
        """Return platform-appropriate backends in priority order for
        this `moe_config`."""
        raise NotImplementedError

    def backend_to_kernel_cls(
        self, backend: BackendT
    ) -> list[type[mk.FusedMoEExperts]]:
        """Map a backend enum value to its concrete `FusedMoEExperts`
        subclasses, in selection priority order."""
        raise NotImplementedError

    def map_backend(self, runner_backend: MoEBackend) -> BackendT:
        """Map a user-facing `MoEBackend` (from the runner config) to
        this oracle's enum."""
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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

    def make_log_backend(
        self,
        backend: BackendT,
        available_backends: list[BackendT] | None = None,
    ) -> str:
        """Message for the backend a select loop settled on."""
        available = (
            f" out of potential backends: {[b.value for b in available_backends]}"
            if available_backends is not None
            else ""
        )
        return f"Using {backend.value} {self.name} MoE backend{available}."

    def make_log_unsupported(self, backend: BackendT, reason: str | None) -> str:
        """Message for a backend that does not support the configuration."""
        since = f" since {reason}" if reason else ""
        return (
            f"{self.name} MoE backend '{backend.value}' does not support the "
            f"deployment configuration{since}."
        )

    def try_backend(
        self,
        backend: BackendT,
        kernel_classes: list[type[mk.FusedMoEExperts]] | None = None,
        activation_key: "QuantKey | None" = _UNSET,
    ) -> tuple[type[mk.FusedMoEExperts] | None, str | None]:
        """Probe `backend`'s candidate kernel classes, first supported wins.

        A miss is an error for an explicitly requested backend and a debug line
        plus the next candidate for the auto-selection loops, so the policy
        stays with the caller. `activation_key` overrides this oracle's own key
        for oracles whose activation key varies per backend.

        Returns:
            `(kernel_cls, None)`, or `(None, reason)` with the last reason.

        """
        reason: str | None = None
        act_quant_key = (
            self.activation_key if activation_key is _UNSET else activation_key
        )
        if kernel_classes is None:
            kernel_classes = self.backend_to_kernel_cls(backend)
        for kernel_cls in kernel_classes:
            supported, reason = kernel_cls.is_supported_config(
                kernel_cls,
                self.moe_config,
                self.weight_key,
                act_quant_key,
                self.activation_format,
            )
            if supported:
                return kernel_cls, None
        return None, reason

    def return_or_raise(
        self,
        backend: BackendT,
        available_backends: list[BackendT] | None = None,
        kernel_classes: list[type[mk.FusedMoEExperts]] | None = None,
        activation_key: "QuantKey | None" = _UNSET,
    ) -> tuple[BackendT, type[mk.FusedMoEExperts]]:
        """Return `backend` with its first supported kernel class, or raise.

        Raises:
            ValueError: no candidate kernel class supports the deployment
                configuration; the message carries the last rejection reason.

        """
        kernel_cls, reason = self.try_backend(backend, kernel_classes, activation_key)
        if kernel_cls is None:
            raise ValueError(self.make_log_unsupported(backend, reason))
        logger.info_once(self.make_log_backend(backend, available_backends))
        return backend, kernel_cls


def make_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    prepare_finalize_kwargs: dict[str, Any] | None = None,
    experts_kwargs: dict[str, Any] | None = None,
    experts_log: str | None = None,
) -> mk.FusedMoEKernel:
    """Assemble a `FusedMoEKernel` from a prepare/finalize and an experts part.

    This is the step every MoE oracle runs once its backend is chosen, so it
    lives here rather than being copy-pasted into each oracle module.

    Args:
        moe_quant_config: Quantization configuration of the layer.
        moe_config: MoE layer configuration.
        experts_cls: Selected experts implementation.
        routing_tables: Optional pre-computed routing tables (EP).
        prepare_finalize_kwargs: Extra kwargs for `maybe_make_prepare_finalize`
            (e.g. `input_dtype`).
        experts_kwargs: Extra kwargs for the experts constructor (e.g. the W4A8
            `b_strides1` / `b_strides2` / `group_size`).
        experts_log: Optional `info_once` template for the selected experts
            class, e.g. `"Using %s MoE backend"`.

    """
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=experts_cls.is_monolithic(),
        **(prepare_finalize_kwargs or {}),
    )
    assert prepare_finalize is not None
    if experts_cls.activation_format() != prepare_finalize.activation_format:
        raise ValueError(
            f"{experts_cls.__name__} expects "
            f"{experts_cls.activation_format().value} activations but the selected "
            f"prepare/finalize uses {prepare_finalize.activation_format.value}"
        )

    logger.info_once("Using %s", prepare_finalize.__class__.__name__)
    if experts_log is not None:
        logger.info_once(experts_log, experts_cls.__name__)

    extra_kwargs = dict(experts_kwargs or {})
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        extra_kwargs["max_num_tokens"] = max_num_tokens
        extra_kwargs["num_dispatchers"] = prepare_finalize.num_dispatchers()

    experts = experts_cls(
        moe_config=moe_config,
        quant_config=moe_quant_config,
        **extra_kwargs,
    )
    return mk.FusedMoEKernel(prepare_finalize, experts)
