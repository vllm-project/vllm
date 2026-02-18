# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Callable
from dataclasses import fields
from typing import TYPE_CHECKING, Any

from datasets.utils.py_utils import asdict
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

    If specified manually, platform defaults will be appended to the lists.
    See KernelConfig.set_platform_defaults().
    """

    rms_norm: list[str] = Field(default_factory=list)
    """Priority list for vllm.ir.ops.rms_norm"""

    fused_add_rms_norm: list[str] = Field(default_factory=list)
    """Priority list for vllm.ir.ops.fused_add_rms_norm"""

    def compute_hash(self) -> str:
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Any future fields that don't affect compilation should be excluded.
        """

        return hash_factors(get_hash_factors(self, set()))

    @contextlib.contextmanager
    def set_priority(self):
        """
        Context manager to set the IR op priority for all op members.
        It also imports vllm.kernels to ensure all implementations are made available.
        """
        import vllm.kernels  # noqa: F401, registers IR op implementations
        from vllm.ir.op import IrOp

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
            if field not in kwargs:
                kwargs[field.name] = default

        return cls(**kwargs)


@config
class KernelConfig:
    """Configuration for kernel selection and warmup behavior."""

    ir_op_priority: IrOpPriorityConfig = Field(default_factory=IrOpPriorityConfig)
    """vLLM IR op priority for dispatching/lowering during the forward pass."""

    enable_flashinfer_autotune: bool = Field(default=None)
    """If True, run FlashInfer autotuning during kernel warmup."""

    def compute_hash(self) -> str:
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Any future fields that don't affect compilation should be excluded.
        """

        return hash_factors(get_hash_factors(self, set("enable_flashinfer_autotune")))

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
            "Setting platform-specific IR op priority defaults: %s",
            platform_op_priority,
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
