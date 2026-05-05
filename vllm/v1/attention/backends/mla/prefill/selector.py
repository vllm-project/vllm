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
            MLAPrefillBackendEnum.FLASH_ATTN,
            MLAPrefillBackendEnum.TRTLLM_RAGGED,
            MLAPrefillBackendEnum.FLASHINFER,
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
    mla_prefill_backend in AttentionConfig, then falls back to automatic
    priority-based selection.

    Args:
        vllm_config: The vLLM configuration.

    Returns:
        The selected prefill backend class.
    """
    from vllm.platforms import current_platform

    device_capability = current_platform.get_device_capability()
    if device_capability is None:
        logger.info_once(
            "Device capability not available, using FlashAttention MLA prefill backend."
        )
        return MLAPrefillBackendEnum.FLASH_ATTN.get_class()

    attention_config = vllm_config.attention_config

    selector_config = MLAPrefillSelectorConfig(
        dtype=vllm_config.model_config.dtype,
        is_r1_compatible=is_deepseek_r1_mla_compatible(vllm_config),
    )

    if attention_config.mla_prefill_backend is not None:
        selected_backend = attention_config.mla_prefill_backend
        backend_cls: type[MLAPrefillBackend] | None = None
        try:
            backend_cls = selected_backend.get_class()
            invalid_reasons = backend_cls.validate_configuration(
                device_capability, selector_config
            )
        except ImportError:
            invalid_reasons = ["ImportError"]
        if invalid_reasons:
            raise ValueError(
                f"Selected MLA prefill backend {selected_backend.name} "
                f"is not valid for this configuration. "
                f"Reason: {invalid_reasons}"
            )
        assert backend_cls is not None
        logger.info("Using %s MLA prefill backend.", selected_backend.name)
        return backend_cls

    return _auto_select_mla_prefill_backend(
        device_capability,
        selector_config,
    )


@cache
def _auto_select_mla_prefill_backend(
    device_capability: DeviceCapability,
    selector_config: MLAPrefillSelectorConfig,
) -> "type[MLAPrefillBackend]":
    """Auto-select the best available MLA prefill backend.

    Args:
        device_capability: The device's compute capability.
        selector_config: Hashable configuration for backend selection.

    Returns:
        The selected prefill backend class.
    """
    priorities = _get_mla_prefill_backend_priorities(device_capability)
    all_invalid_reasons: dict[str, list[str]] = {}

    for backend_enum in priorities:
        backend_cls: type[MLAPrefillBackend] | None = None
        try:
            backend_cls = backend_enum.get_class()
            invalid_reasons = backend_cls.validate_configuration(
                device_capability, selector_config
            )
        except ImportError:
            invalid_reasons = ["ImportError"]
        if not invalid_reasons:
            assert backend_cls is not None
            logger.info_once("Using %s MLA prefill backend.", backend_enum.name)
            return backend_cls
        all_invalid_reasons[backend_enum.name] = invalid_reasons

    reasons_str = (
        "{"
        + ", ".join(
            f"{name}: [{', '.join(reasons)}]"
            for name, reasons in all_invalid_reasons.items()
        )
        + "}"
    )
    config_str = repr(selector_config)
    logger.debug_once(
        "Some MLA prefill backends are not valid with %s. Reasons: %s.",
        config_str,
        reasons_str,
    )

    raise ValueError(
        f"No valid MLA prefill backend found with {config_str}. Reasons: {reasons_str}."
    )
