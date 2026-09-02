# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Callable, Iterable
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
    forward pass. Each member is a list of strings, which will be installed
    in worker init via vllm.ir.ops.<op_name>.set_default().
    A single comma-separated string is accepted as well,

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
            for name, p in asdict(self).items()  # type: ignore[call-overload]
        }

        return hash_factors(factors)

    @field_validator("*", mode="before")
    @classmethod
    def _to_list_str(cls, value: str | list[str]):
        if isinstance(value, str):
            value = value.replace(" ", "").split(",")

        assert all(isinstance(v, str) for v in value)
        return value

    def _iter_op_priorities(self):
        """
        Yield (IrOp, priority_list) for each field, after importing platform
        kernels and validating each entry.
        """
        from vllm.ir.op import IrOp
        from vllm.platforms import current_platform

        current_platform.import_ir_kernels()

        for field in fields(self):  # type: ignore[arg-type]
            op_priority = getattr(self, field.name)
            assert op_priority is not None, (
                f"IR op priority for {field.name} must be set"
            )
            logger.debug("Setting IR op priority for %s to %s", field.name, op_priority)
            yield IrOp.registry[field.name], op_priority

    def set_default(self) -> None:
        """
        Permanently set the IR op priority for all op members.
        """
        for ir_op, op_priority in self._iter_op_priorities():
            ir_op.set_default(op_priority)

    @contextlib.contextmanager
    def set_priority(self):
        """
        Context manager to set the IR op priority for all op members.
        It also imports IR kernel implementations for the current platform
        to ensure all implementations are made available.
        """
        with contextlib.ExitStack() as stack:
            for ir_op, op_priority in self._iter_op_priorities():
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
        for field in fields(cls):  # type: ignore[arg-type]
            if field.name not in kwargs:
                kwargs[field.name] = list(default)

        return cls(**kwargs)


MoEBackend = Literal[
    "auto",
    "triton",
    "batched_triton",
    "deep_gemm",
    "deep_gemm_mega_moe",
    "cutlass",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "flashinfer_b12x",
    "b12x",
    "flashinfer_moe_ep_mega_deep_gemm",
    "flashinfer_moe_ep_mega_cutedsl",
    "marlin",
    "humming",
    "triton_unfused",
    "aiter",
    "aiter_triton_mxfp4_bf16",
    "flydsl",
    "hpc",
    "emulation",
]

# Backends that run the mega-MoE model path through the flashinfer moe_ep
# runtime. Only architectures in FLASHINFER_MOE_EP_ARCHITECTURES wire up
# these experts.
FLASHINFER_MOE_EP_BACKENDS = frozenset(
    {
        "flashinfer_moe_ep_mega_deep_gemm",
        "flashinfer_moe_ep_mega_cutedsl",
    }
)

# Backends that run a mega-MoE model path (fused expert module plus
# prepare_megamoe routing): vLLM's native deep_gemm path, which any model
# with a mega-MoE module may use (DeepSeek-V4, Kimi K3), plus the flashinfer
# moe_ep variants.
MEGA_MOE_BACKENDS = frozenset({"deep_gemm_mega_moe"}) | FLASHINFER_MOE_EP_BACKENDS

# Architectures whose model code wires up the flashinfer moe_ep experts. MTP
# and DSpark draft variants inherit the setting from these target models.
FLASHINFER_MOE_EP_ARCHITECTURES = frozenset(
    {
        "DeepseekV4ForCausalLM",
        "DeepSeekV4MTPModel",
    }
)


def validate_flashinfer_moe_ep_model(
    moe_backend: str, architectures: Iterable[str]
) -> None:
    """Reject flashinfer moe_ep backends for models that lack the FI path."""
    if moe_backend not in FLASHINFER_MOE_EP_BACKENDS:
        return
    if not any(arch in FLASHINFER_MOE_EP_ARCHITECTURES for arch in architectures):
        raise ValueError(
            f"moe_backend={moe_backend!r} is only supported for DeepSeek-V4 "
            f"models ({sorted(FLASHINFER_MOE_EP_ARCHITECTURES)}), but the "
            f"model is {list(architectures)}."
        )


LinearBackend = Literal[
    "auto",
    "cutlass",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "flashinfer_trtllm",
    "flashinfer_cudnn",
    "flashinfer_b12x",
    "b12x",
    "marlin",
    "humming",
    "triton",
    "deep_gemm",
    "torch",
    "aiter",
    "machete",
    "fbgemm",
    "conch",
    "exllama",
    "emulation",
    "xpu",
    "xpu_woq",
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

    # TODO(roberto): Remove after registered CuTeDSL warmups are migrated
    # to the shared JIT warmup infrastructure.
    # https://github.com/vllm-project/vllm/pull/47451
    enable_cutedsl_warmup: bool = True
    """Deprecated: run legacy CuTeDSL warmup providers."""

    enable_jit_warmup: bool = True
    """If True, run JIT compile warmup during kernel warmup."""

    enable_bf16x3_router_gemm: bool = False
    """If True, use the experimental SM100 BF16x3 CuteDSL router GEMM."""

    moe_backend: MoEBackend = "auto"
    """Backend for MoE expert computation kernels. Available options:

    - "auto": Automatically select the best backend based on model and hardware
    - "triton": Use Triton-based fused MoE kernels
    - "batched_triton": Use batched Triton experts (moe_mmk) on the batched
      activation format ([E_local, max_num_tokens, K])
    - "deep_gemm": Use DeepGEMM kernels (FP8 block-quantized only)
    - "deep_gemm_mega_moe": Use DeepGEMM mega MoE kernels
    - "cutlass": Use vLLM CUTLASS kernels
    - "flashinfer_trtllm": Use FlashInfer with TRTLLM-GEN kernels
    - "flashinfer_cutlass": Use FlashInfer with CUTLASS kernels
    - "flashinfer_cutedsl": Use FlashInfer with CuteDSL kernels (FP4 only)
    - "flashinfer_b12x": Use FlashInfer CuteDSL fused MoE for SM12x
      (RTX Pro 6000 / DGX Spark)
    - "b12x": Use b12x FP4 MoE kernels on SM12x
    - "flashinfer_moe_ep_mega_deep_gemm": Use the FlashInfer moe_ep
      expert-parallel mega-MoE with the DeepGEMM megakernel, which consumes an
      MXFP4 checkpoint verbatim (Blackwell, requires expert parallel;
      DeepSeek-V4 only)
    - "flashinfer_moe_ep_mega_cutedsl": Same, with the CuteDSL megakernel
      (additionally requires NVSHMEM). The checkpoint selects the weight path:
      an NVFP4 checkpoint is consumed prequantized, MXFP4 weights are
      requantized at load
    - "marlin": Use Marlin kernels (weight-only quantization)
    - "humming": Use Humming Mixed Precision kernels
    - "triton_unfused": Use Triton unfused MoE kernels
    - "aiter": Use AMD AITer kernels (ROCm only)
    - "aiter_triton_mxfp4_bf16": Use the AITER Triton MXFP4 W4A16
      (moe_gemm_a16w4) MoE kernel (ROCm gfx942/gfx950/gfx1250)
    - "flydsl": Use AMD FlyDSL kernels (ROCm only)
    - "hpc": Use HPC kernels (FP8 and Hopper only)
    - "emulation": use BF16/FP16 GEMM, dequantizing weights and
                   running QDQ on activations.
    """

    linear_backend: LinearBackend = "auto"
    """Backend for linear layer GEMM kernels. Available options:

    Layer types without an implementation from the requested backend use
    automatic selection.

    - "auto": Automatically select the best backend based on model and hardware
    - "cutlass": Use CUTLASS-based kernels
    - "flashinfer_cutlass": Use FlashInfer with CUTLASS kernels
    - "flashinfer_cutedsl": Use FlashInfer with CuTe-DSL kernels
      (BF16, NVFP4, MXFP8, W4A16_NVFP4)
    - "flashinfer_trtllm": Use FlashInfer with TensorRT-LLM kernels
    - "flashinfer_cudnn": Use FlashInfer with cuDNN kernels
    - "flashinfer_b12x": Use FlashInfer b12x CuteDSL NVFP4 GEMM (SM120+)
    - "b12x": Use native B12X FP8 and FP4 linear kernels on SM12x
    - "marlin": Use Marlin kernels
    - "triton": Use Triton-based kernels
    - "deep_gemm": Use DeepGEMM kernels
    - "torch": Use PyTorch native scaled_mm kernels
    - "aiter": Use AMD AITer kernels (ROCm only)
    - "machete": Use Machete kernels (mixed-precision)
    - "fbgemm": Use FBGEMM kernels
    - "conch": Use Conch mixed-precision kernels
    - "exllama": Use Exllama mixed-precision kernels
    - "emulation": Use slow dequant-to-BF16 emulation (for testing only)
    - "xpu": Use XPU kernels
    - "xpu_woq": Use XPU kernels for weight-only quantization (e.g. W8A16)
    """

    @field_validator("moe_backend", mode="before")
    @classmethod
    def _normalize_moe_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower().replace("-", "_")
        return value

    @field_validator("linear_backend", mode="before")
    @classmethod
    def _normalize_linear_backend(cls, value: Any) -> Any:
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
            "enable_cutedsl_warmup",
            "enable_jit_warmup",
            "enable_flashinfer_autotune",
            "ir_op_priority",  # handled separately below
        }
        factors = get_hash_factors(self, ignored_factors)
        factors["ir_op_priority"] = self.ir_op_priority.compute_hash()
        return hash_factors(factors)

    @field_validator(
        "enable_flashinfer_autotune",
        "enable_cutedsl_warmup",
        "enable_jit_warmup",
        mode="wrap",
    )
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialization is delayed."""
        if value is None:
            return value
        return handler(value)

    def set_platform_defaults(self, vllm_config: "VllmConfig") -> None:
        """Set platform-specific defaults for the kernel config."""
        from vllm.platforms import current_platform

        if vllm_config.model_config is not None:
            validate_flashinfer_moe_ep_model(
                self.moe_backend, vllm_config.model_config.architectures
            )

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
