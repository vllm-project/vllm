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


MoEBackend = Literal[
    "auto",
    "triton",
    "deep_gemm",
    "deep_gemm_mega_moe",
    "cutlass",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "marlin",
    "humming",
    "triton_unfused",
    "aiter",
    "emulation",
]


@config
class IrOpPriorityConfig:
    """
    Configuration for vLLM IR op priority for dispatching/lowering during the
    forward pass. Uses a dict to store priorities, automatically synced with
    IrOp.registry.

    Each key is an op_name (matching IrOp.registry keys), and each value is a
    list of strings passed to vllm.ir.ops.<op_name>.set_priority().
    A single comma-separated string is accepted as well.

    If specified manually, platform defaults will be appended to the lists.
    See KernelConfig.set_platform_defaults().
    """

    priorities: dict[str, list[str]] = Field(default_factory=dict)
    """Priority list for each IR op, keyed by op_name."""

    def compute_hash(self) -> str:
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Any future fields that don't affect compilation should be excluded.

        Also, manually add IR op impl UUIDs to make sure they affect the compile cache.
        """
        from vllm.ir.op import IrOp

        factors: dict[str, object] = {"priorities": dict(self.priorities)}
        factors["_impls"] = {
            name: {
                provider: IrOp.registry[name].impls[provider].uuid()
                for provider in priority
                if provider in IrOp.registry[name].impls
            }
            for name, priority in self.priorities.items()
            if name in IrOp.registry
        }

        return hash_factors(factors)

    @field_validator("priorities", mode="before")
    @classmethod
    def _to_list_str(cls, value: dict[str, str | list[str]]):
        result = {}
        for k, v in value.items():
            if isinstance(v, str):
                v = v.replace(" ", "").split(",")
            assert all(isinstance(item, str) for item in v)
            result[k] = v
        return result

    @contextlib.contextmanager
    def set_priority(self):
        """
        Context manager to set the IR op priority for all ops.
        It also imports IR kernel implementations for the current platform
        to ensure all implementations are made available.
        """
        from vllm.ir.op import IrOp
        from vllm.platforms import current_platform

        current_platform.import_ir_kernels()

        with contextlib.ExitStack() as stack:
            for op_name, op_priority in self.priorities.items():
                assert op_priority is not None, (
                    f"IR op priority for {op_name} must be set"
                )
                logger.debug(
                    "Setting IR op priority for %s to %s", op_name, op_priority
                )
                if op_name in IrOp.registry:
                    ir_op = IrOp.registry[op_name]
                    stack.enter_context(ir_op.set_priority(op_priority))

            yield

    @classmethod
    def with_default(
        cls, default: list[str], /, **overrides: list[str]
    ) -> "IrOpPriorityConfig":
        """
        A helper to create an IrOpPriorityConfig where all ops use the given
        default list, with specified overrides.

        Args:
            default: Default priority list for all ops.
            **overrides: Keyword arguments mapping op_name to its priority list.

        Raises:
            KeyError: If an override key is not a registered IR op.
        """
        from vllm.ir.op import IrOp

        # Validate overrides
        for op_name in overrides:
            if op_name not in IrOp.registry:
                raise KeyError(
                    f"Unknown IR op '{op_name}'. "
                    f"Available ops: {list(IrOp.registry.keys())}"
                )

        priorities = {}
        for op_name in IrOp.registry:
            priorities[op_name] = list(overrides.get(op_name, default))

        return cls(priorities=priorities)


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
    - "deep_gemm_mega_moe": Use DeepGEMM mega MoE kernels
    - "cutlass": Use vLLM CUTLASS kernels
    - "flashinfer_trtllm": Use FlashInfer with TRTLLM-GEN kernels
    - "flashinfer_cutlass": Use FlashInfer with CUTLASS kernels
    - "flashinfer_cutedsl": Use FlashInfer with CuteDSL kernels (FP4 only)
    - "marlin": Use Marlin kernels (weight-only quantization)
    - "humming": Use Humming Mixed Precision kernels
    - "triton_unfused": Use Triton unfused MoE kernels
    - "aiter": Use AMD AITer kernels (ROCm only)
    - "emulation": use BF16/FP16 GEMM, dequantizing weights and
                   running QDQ on activations.
    """

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
        from vllm.ir.op import IrOp
        from vllm.platforms import current_platform

        platform_op_priority = current_platform.get_default_ir_op_priority(vllm_config)
        logger.debug(
            "Setting platform-specific IR op priority defaults: %s, user-defined: %s",
            platform_op_priority,
            self.ir_op_priority,
        )

        # Check for missing ops that will get platform defaults
        all_known_ops = set(IrOp.registry.keys())
        configured_ops = set(self.ir_op_priority.priorities.keys())
        platform_ops = set(platform_op_priority.priorities.keys())

        # After merging, these ops will have priorities set
        will_have_priorities = configured_ops | platform_ops
        missing_ops = all_known_ops - will_have_priorities
        if missing_ops:
            logger.warning(
                "IR ops without explicit priority config (will use platform defaults): %s",
                list(missing_ops),
            )

        for op_name, op_priority in platform_op_priority.priorities.items():
            if op_name not in self.ir_op_priority.priorities:
                self.ir_op_priority.priorities[op_name] = list(op_priority)
            else:
                # Append platform-specific priorities
                # Must be idempotent because vllm_config.set_platform_defaults() may be
                # called multiple times (due to VllmConfig.__post_init__ manual call).
                current_op_priority = self.ir_op_priority.priorities[op_name]
                unique_op_priority = [
                    op for op in op_priority if op not in current_op_priority
                ]
                current_op_priority.extend(unique_op_priority)

        logger.info(
            "Final IR op priority after setting platform defaults: %s",
            self.ir_op_priority,
        )