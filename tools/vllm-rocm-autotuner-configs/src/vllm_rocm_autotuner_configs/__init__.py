# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM ROCm Auto-Tuner Configurations.

Provides optimized environment variable settings for vLLM on ROCm-enabled GPUs.
"""

from __future__ import annotations

import logging
from pathlib import Path


class GPUDetectionError(Exception):
    """Raised when GPU detection fails."""

    pass


try:
    from .utils import GPUDetectionError as _GPUDetectionError
    from .utils import (check_amdsmi_available, get_amd_gpu_info,
                        get_amd_gpu_info_safe)

    # Use the imported version if available
    GPUDetectionError = _GPUDetectionError  # type: ignore[misc]
    _HAS_UTILS = True
except ImportError:
    _HAS_UTILS = False

    def get_amd_gpu_info() -> tuple[str, int]:  # type: ignore[misc]
        """Stub function when utils not available."""
        raise GPUDetectionError("amdsmi not available")

    def get_amd_gpu_info_safe() -> tuple[str | None, int]:
        """Stub function when utils not available."""
        return None, 0

    def check_amdsmi_available() -> bool:
        """Stub function when utils not available."""
        return False


__version__ = "0.2.1"

_logger = logging.getLogger(__name__)
_CONFIG_DIR = Path(__file__).parent / "configs"

# Supported architectures
SUPPORTED_ARCHITECTURES = frozenset(["gfx942", "gfx950"])

# Architecture mappings for older GPUs
ARCHITECTURE_MAPPINGS: dict[str, str] = {
    # "gfx90a": "gfx942",  # MI200 series maps to MI300X config
}


class ConfigNotFoundError(Exception):
    """Raised when a configuration file cannot be found."""


class UnsupportedArchitectureError(Exception):
    """Raised when GPU architecture is not supported."""


def get_config_path(
    arch: str | None = None,
    config_file: str | Path | None = None,
) -> Path:
    """Get path to config file for given architecture.

    Simplified to only search:
    1. Explicit config_file if provided
    2. Package bundled configs

    Args:
        arch: GPU architecture (e.g., 'gfx942'). If None, attempts
            auto-detection.
        config_file: Explicit path to config file. Takes precedence over arch.

    Returns:
        Path to config file.

    Raises:
        ConfigNotFoundError: If no suitable config file is found.
        UnsupportedArchitectureError: If architecture is detected but not
            supported.

    Examples:
        >>> # Auto-detect and get config
        >>> path = get_config_path()
        >>>
        >>> # Specific architecture
        >>> path = get_config_path(arch='gfx942')
        >>>
        >>> # Explicit config file
        >>> path = get_config_path(config_file='/path/to/config.json')
    """
    # 1. Explicit config file takes precedence
    if config_file is not None:
        config_path = Path(config_file)
        if not config_path.exists():
            raise ConfigNotFoundError(f"Config file not found: {config_path}")
        _logger.info(f"Using explicit config: {config_path}")  # noqa: G004
        return config_path

    # 2. Try auto-detection if no arch specified
    if arch is None:
        if _HAS_UTILS and check_amdsmi_available():
            try:
                detected_arch, gpu_count = get_amd_gpu_info()
                if gpu_count == 0:
                    raise ConfigNotFoundError(
                        "No AMD GPUs detected. ROCm tuner requires AMD GPUs.")
                if detected_arch and detected_arch != "unknown":
                    arch = detected_arch
                    _logger.info(
                        f"Auto-detected GPU architecture: {arch}"  # noqa: G004
                    )
            except GPUDetectionError as e:
                raise ConfigNotFoundError(f"GPU detection failed: {e}") from e
        else:
            raise ConfigNotFoundError(
                "Cannot auto-detect GPU architecture. "
                "Install with: pip install "
                "vllm-rocm-autotuner-configs[gpu-detect]")

    if arch is None:
        raise ConfigNotFoundError(
            "No architecture specified and auto-detection failed")

    # 3. Apply architecture mappings for older GPUs
    original_arch = arch
    if arch in ARCHITECTURE_MAPPINGS:
        mapped_arch = ARCHITECTURE_MAPPINGS[arch]
        _logger.info(f"Mapping {arch} -> {mapped_arch}")  # noqa: G004
        arch = mapped_arch

    # 4. Check if architecture is supported
    if arch not in SUPPORTED_ARCHITECTURES:
        raise UnsupportedArchitectureError(
            f"GPU architecture '{original_arch}' is not supported. "
            f"Supported: {', '.join(sorted(SUPPORTED_ARCHITECTURES))}")

    # 5. Look for config in package
    config_file_path = _CONFIG_DIR / f"rocm_config_{arch}.json"
    if not config_file_path.exists():
        raise ConfigNotFoundError(
            f"Config file for architecture '{arch}' not found in package")

    _logger.info(
        f"Using package config: {config_file_path.name}")  # noqa: G004
    return config_file_path


def list_available_configs() -> list[str]:
    """List all available GPU architectures with configs.

    Returns:
        Sorted list of architecture names.

    Examples:
        >>> configs = list_available_configs()
        >>> print(f"Available configs: {', '.join(configs)}")
        Available configs: gfx942, gfx950
    """
    if not _CONFIG_DIR.exists():
        return []

    configs = []
    for config_file in _CONFIG_DIR.glob("rocm_config_*.json"):
        arch = config_file.stem.replace("rocm_config_", "")
        configs.append(arch)

    return sorted(configs)


__all__ = [
    "__version__",
    "get_config_path",
    "list_available_configs",
    "ConfigNotFoundError",
    "UnsupportedArchitectureError",
    "SUPPORTED_ARCHITECTURES",
    "ARCHITECTURE_MAPPINGS",
    "get_amd_gpu_info",
    "get_amd_gpu_info_safe",
    "check_amdsmi_available",
    "GPUDetectionError",
]
