# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selector for MLA prefill backends.

This module provides functions for selecting the appropriate MLA prefill
backend based on device capabilities and configuration.
"""

from functools import cache
from typing import TYPE_CHECKING, NamedTuple

import torch

from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

logger = init_logger(__name__)


class MLAPrefillSelectorConfig(NamedTuple):
    """Hashable configuration for MLA prefill backend selection.

    This is analogous to AttentionSelectorConfig and contains model-specific
    configuration needed to select an MLA prefill backend, extracted from
    VllmConfig into a hashable form for caching.
    """

    dtype: torch.dtype
    is_r1_compatible: bool


def is_deepseek_r1_mla_compatible(vllm_config: "VllmConfig") -> bool:
    """Check if model has DeepSeek R1 compatible MLA dimensions.

    DeepSeek R1 MLA dimensions are:
    - qk_nope_head_dim = 128
    - qk_rope_head_dim = 64
    - v_head_dim = 128

    These dimensions are required for optimized backends like TRTLLM_RAGGED,
    FLASHINFER, and CUDNN on Blackwell.
    """
    if vllm_config.model_config is None:
        return False
    hf_text_config = vllm_config.model_config.hf_text_config
    qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 1)
    qk_rope_head_dim = getattr(hf_text_config, "qk_rope_head_dim", 1)
    v_head_dim = getattr(hf_text_config, "v_head_dim", 1)
    return qk_nope_head_dim == 128 and qk_rope_head_dim == 64 and v_head_dim == 128


def _get_mla_prefill_backend_priorities(
    device_capability: DeviceCapability,
) -> list[MLAPrefillBackendEnum]:
    """Get MLA prefill backend priorities based on device capability.

    Args:
        device_capability: The device's compute capability.

    Returns:
        List of backends in priority order (highest priority first).
    """
    if device_capability.major == 10:  # Blackwell
        return [
            MLAPrefillBackendEnum.TRTLLM_RAGGED,
            MLAPrefillBackendEnum.FLASHINFER,
            MLAPrefillBackendEnum.CUDNN,
            MLAPrefillBackendEnum.FLASH_ATTN,
        ]
    else:  # Hopper (SM90) and older
        return [
            MLAPrefillBackendEnum.FLASH_ATTN,
        ]


def get_mla_prefill_backend(
    vllm_config: "VllmConfig",
) -> "type[MLAPrefillBackend]":
    """Select the MLA prefill backend based on configuration and device.

    This function first checks for explicit user preferences via
    mla.prefill_backend in AttentionConfig, then falls back to automatic
    priority-based selection.

    Args:
        vllm_config: The vLLM configuration.

    Returns:
        The selected prefill backend class.
    """
    from vllm.platforms import current_platform

    device_capability = current_platform.get_device_capability()
    if device_capability is None:
        # Fallback for non-CUDA platforms or during profiling
        logger.info_once(
            "Device capability not available, using FlashAttention MLA prefill"
        )
        return MLAPrefillBackendEnum.FLASH_ATTN.get_class()

    attention_config = vllm_config.attention_config

    # Build hashable selector config for caching
    selector_config = MLAPrefillSelectorConfig(
        dtype=vllm_config.model_config.dtype,
        is_r1_compatible=is_deepseek_r1_mla_compatible(vllm_config),
    )

    # Check for explicit backend selection (includes migrated deprecated flags)
    if attention_config.mla_prefill_backend is not None:
        backend_enum = attention_config.mla_prefill_backend
        try:
            backend_cls = backend_enum.get_class()
            invalid_reasons = backend_cls.validate_configuration(
                device_capability, selector_config
            )
            if not invalid_reasons:
                logger.info_once("Using %s for MLA prefill", backend_cls.get_name())
                return backend_cls
            else:
                logger.warning(
                    "Requested MLA prefill backend %s is not valid: %s. "
                    "Falling back to auto-selection.",
                    backend_enum.name,
                    invalid_reasons,
                )
        except ImportError as e:
            logger.warning(
                "Requested MLA prefill backend %s is not available: %s. "
                "Falling back to auto-selection.",
                backend_enum.name,
                e,
            )

    # Auto-select based on priority
    # Pass major/minor as ints for caching (DeviceCapability isn't hashable)
    return _auto_select_mla_prefill_backend(
        device_capability.major,
        device_capability.minor,
        selector_config,
    )


@cache
def _auto_select_mla_prefill_backend(
    major: int,
    minor: int,
    selector_config: MLAPrefillSelectorConfig,
) -> "type[MLAPrefillBackend]":
    """Auto-select the best available MLA prefill backend.

    Args:
        major: Device capability major version (for cache key).
        minor: Device capability minor version (for cache key).
        selector_config: Hashable configuration for backend selection.

    Returns:
        The selected prefill backend class.
    """
    device_capability = DeviceCapability(major=major, minor=minor)
    priorities = _get_mla_prefill_backend_priorities(device_capability)

    for backend_enum in priorities:
        try:
            backend_cls = backend_enum.get_class()
            invalid_reasons = backend_cls.validate_configuration(
                device_capability, selector_config
            )
            if not invalid_reasons:
                logger.info_once("Using %s for MLA prefill", backend_cls.get_name())
                return backend_cls
            else:
                logger.debug(
                    "MLA prefill backend %s not valid: %s",
                    backend_enum.name,
                    invalid_reasons,
                )
        except ImportError as e:
            logger.debug(
                "MLA prefill backend %s not available: %s",
                backend_enum.name,
                e,
            )
            continue

    # Fallback to FlashAttention (should always be available)
    logger.info_once("Using FLASH_ATTN for MLA prefill (fallback)")
    return MLAPrefillBackendEnum.FLASH_ATTN.get_class()
