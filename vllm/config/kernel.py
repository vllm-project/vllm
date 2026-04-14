# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Callable
from dataclasses import asdict, fields
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, field_validator

from vllm.config.utils import config, get_hash_factors, hash_factors
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@config
class IrOpPriorityConfig:
    """
    Configuration for vLLM IR op priority for dispatching/lowering during the
    forward pass. Each member is a list of strings, which will be passed to
    vllm.ir.ops.<op_name>.set_priority() for the duration of the forward pass.
    A single comma-separated string is accepted as well,

    If specified manually, platform defaults will be appended to the lists.
    See KernelConfig.set_platform_defaults().
    """

    rms_norm: list[str] = Field(default_factory=list)
    """Priority list for vllm.ir.ops.rms_norm"""

    mixer2_rms_norm_gated: list[str] = Field(default_factory=list)
    """Priority list for vllm.ir.ops.rms_norm_gated"""

    def compute_hash(self) -> str:
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Any future fields that don't affect compilation should be excluded.

        Also, manually add IR op impl UUIDs to make sure they affect the compile cache.
        """
        factors = get_hash_factors(self, set())

        # Implementations are hidden from Dynamo,
        # so they don't show up in the traced files list.
        from vllm.ir.op import IrOp

        assert "_impls" not in factors
        factors["_impls"] = {
            name: {
                provider: IrOp.registry[name].impls[provider].uuid() for provider in p
            }
            for name, p in asdict(self).items()
        }

        return hash_factors(factors)

    @field_validator("*", mode="before")
    @classmethod
    def _to_list_str(cls, value: str | list[str]):
        if isinstance(value, str):
            value = value.replace(" ", "").split(",")

        assert all(isinstance(v, str) for v in value)
        return value

    @contextlib.contextmanager
    def set_priority(self):
        """
        Context manager to set the IR op priority for all op members.
        It also imports IR kernel implementations for the current platform
        to ensure all implementations are made available.
        """
        from vllm.ir.op import IrOp
        from vllm.platforms import current_platform

        current_platform.import_ir_kernels()

        with contextlib.ExitStack() as stack:
            for field in fields(self):
                op_priority = getattr(self, field.name)
                assert op_priority is not None, (
                    f"IR op priority for {field.name} must be set"
                )
                logger.debug(
                    "Setting IR op priority for %s to %s", field.name, op_priority
                )
                ir_op = IrOp.registry[field.name]
                stack.enter_context(ir_op.set_priority(op_priority))

            yield

    @classmethod
    def with_default(
        cls, default: list[str], /, **kwargs: list[str]
    ) -> "IrOpPriorityConfig":
        """
        A helper to create an IrOpPriorityConfig where fields not specified in kwargs
        use the given default list.
        """
        for field in fields(cls):
            if field.name not in kwargs:
                kwargs[field.name] = list(default)

        return cls(**kwargs)


MoEBackend = Literal[
    "auto",
    "triton",
    "deep_gemm",
    "cutlass",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "marlin",
    "aiter",
]


@config
class KernelConfig:
    """Configuration for kernel selection and warmup behavior."""

    ir_op_priority: IrOpPriorityConfig = Field(default_factory=IrOpPriorityConfig)
    """
    vLLM IR op priority for dispatching/lowering during the forward pass.
    Platform defaults appended automatically during VllmConfig.__post_init__.
    """

    enable_flashinfer_autotune: bool = None  # type: ignore[assignment]
    """If True, run FlashInfer autotuning during kernel warmup."""

    moe_backend: MoEBackend = "auto"
    """Backend for MoE expert computation kernels. Available options:

    - "auto": Automatically select the best backend based on model and hardware
    - "triton": Use Triton-based fused MoE kernels
    - "deep_gemm": Use DeepGEMM kernels (FP8 block-quantized only)
    - "cutlass": Use vLLM CUTLASS kernels
    - "flashinfer_trtllm": Use FlashInfer with TRTLLM-GEN kernels
    - "flashinfer_cutlass": Use FlashInfer with CUTLASS kernels
    - "flashinfer_cutedsl": Use FlashInfer with CuteDSL kernels (FP4 only)
    - "marlin": Use Marlin kernels (weight-only quantization)
    - "aiter": Use AMD AITer kernels (ROCm only)"""

    @field_validator("moe_backend", mode="before")
    @classmethod
    def _normalize_moe_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower().replace("-", "_")
        return value

    def compute_hash(self) -> str:
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Any future fields that don't affect compilation should be excluded.
        """
        ignored_factors = {
            "enable_flashinfer_autotune",
            "ir_op_priority",  # handled separately below
        }
        factors = get_hash_factors(self, ignored_factors)
        factors["ir_op_priority"] = self.ir_op_priority.compute_hash()
        return hash_factors(factors)

    @field_validator("enable_flashinfer_autotune", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialization is delayed."""
        if value is None:
            return value
        return handler(value)

    def set_platform_defaults(self, vllm_config: "VllmConfig") -> None:
        """Set platform-specific defaults for the kernel config."""
        from vllm.platforms import current_platform

        platform_op_priority = current_platform.get_default_ir_op_priority(vllm_config)
        logger.debug(
            "Setting platform-specific IR op priority defaults: %s, user-defined: %s",
            platform_op_priority,
            self.ir_op_priority,
        )
        for op_name, op_priority in asdict(platform_op_priority).items():
            current_op_priority: list[str] = getattr(self.ir_op_priority, op_name)
            if current_op_priority is None:
                setattr(self.ir_op_priority, op_name, op_priority)
            else:
                # Append platform-specific priorities
                # Must be idempotent because vllm_config.set_platform_defaults() may be
                # called multiple times (due to VllmConfig.__post_init__ manual call).
                unique_op_priority = [
                    op for op in op_priority if op not in current_op_priority
                ]
                current_op_priority.extend(unique_op_priority)

        logger.info(
            "Final IR op priority after setting platform defaults: %s",
            self.ir_op_priority,
        )
