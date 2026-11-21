# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel selection shared by linear-attention ops."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheSpec

logger = init_logger(__name__)


@dataclass(frozen=True)
class LinearAttnSelectorConfig:
    """What kernels are selected by, extracted from the engine config like
    `AttentionSelectorConfig`: the platform, the layer's KV cache spec and the
    activation dtype (None when unknown)."""

    platform: PlatformEnum
    device_capability: DeviceCapability | None
    kv_cache_spec: "KVCacheSpec | None" = None
    dtype: torch.dtype | None = None

    @classmethod
    def from_current_platform(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "KVCacheSpec | None" = None,
    ) -> "LinearAttnSelectorConfig":
        from vllm.platforms import current_platform

        try:
            capability = current_platform.get_device_capability()
        except NotImplementedError:
            capability = None
        model_config = vllm_config.model_config
        return cls(
            current_platform._enum,
            capability,
            kv_cache_spec,
            model_config.dtype if model_config is not None else None,
        )


class LinearAttnKernel:
    """A kernel for one linear-attention op.

    Kernels are stateless classes that are never instantiated, like attention
    backends: layers hold the selected class, and per-layer state reaches it
    through the layer argument.
    """

    backend: ClassVar[StrEnum]

    @classmethod
    def supports_platform(cls, platform: PlatformEnum) -> bool:
        raise NotImplementedError

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability | None) -> bool:
        return True

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return True

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def validate_configuration(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        """Return why this kernel can't run under `cfg`; empty if it can."""
        reasons = []
        if not cls.supports_platform(cfg.platform):
            reasons.append(f"platform {cfg.platform.name} not supported")
        if not cls.supports_compute_capability(cfg.device_capability):
            capability = cfg.device_capability
            found = "unknown" if capability is None else capability.as_version_str()
            reasons.append(f"compute capability {found} not supported")
        if cfg.dtype is not None and not cls.supports_dtype(cfg.dtype):
            reasons.append(f"dtype {cfg.dtype} not supported")
        if not cls.is_available():
            reasons.append("required dependencies not available")
        return reasons + cls.validate_kernel(cfg)

    @classmethod
    def validate_kernel(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        """Return why this kernel can't run under `cfg` for reasons the
        `supports_*` checks don't cover, such as layer shapes."""
        return []


def select_linear_attn_kernel(
    op: str,
    kernels: Sequence[type[LinearAttnKernel]],
    cfg: LinearAttnSelectorConfig,
    requested: StrEnum | None = None,
) -> type[LinearAttnKernel]:
    """Select a kernel for `op` from `kernels`, listed in priority order.

    Args:
        op: The op's name, for messages.
        kernels: The op's kernels, in priority order.
        cfg: What the kernels validate against.
        requested: The requested kernel's backend; None selects automatically.

    Raises:
        ValueError: If the requested kernel can't run, or no kernel can.

    """
    by_backend = {kernel.backend: kernel for kernel in kernels}
    if requested is not None:
        kernel = by_backend.get(requested)
        if kernel is None:
            reasons = [f"no {requested} kernel; options: {', '.join(by_backend)}"]
        else:
            reasons = kernel.validate_configuration(cfg)
        if kernel is not None and not reasons:
            logger.info_once("Using %s kernel for %s.", requested, op)
            return kernel
        raise ValueError(f"Backend {requested} can't run {op}: {'; '.join(reasons)}")

    all_reasons = []
    for kernel in kernels:
        reasons = kernel.validate_configuration(cfg)
        if not reasons:
            logger.info_once("Using %s kernel for %s.", kernel.backend, op)
            return kernel
        all_reasons.append(f"{kernel.backend}: {'; '.join(reasons)}")
    raise ValueError(f"No kernel can run {op}. {' | '.join(all_reasons)}")
