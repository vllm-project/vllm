# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for GPU detection and configuration management."""

from __future__ import annotations

import contextlib
import logging
import sys
from functools import lru_cache

_logger = logging.getLogger("vllm_rocm_autotuner_configs")

try:
    from amdsmi import (AmdSmiException, amdsmi_get_gpu_asic_info,
                        amdsmi_get_processor_handles, amdsmi_init,
                        amdsmi_shut_down)

    _HAS_AMDSMI = True
except (ImportError, KeyError, OSError) as e:
    _HAS_AMDSMI = False
    _logger.debug(
        f"amdsmi not available ({type(e).__name__}). "  # noqa: G004
        "GPU auto-detection unavailable. "
        "Install with: pip install vllm-rocm-autotuner-configs[gpu-detect]")


class GPUDetectionError(Exception):
    """Raised when GPU detection fails."""

    pass


@lru_cache(maxsize=1)
def get_amd_gpu_info() -> tuple[str, int]:
    """
    Gets GPU architecture and count for AMD GPUs.

    Returns:
        Tuple of (architecture_name, num_gpus)
        - architecture_name: GPU architecture string (e.g., 'gfx942', 'gfx90a')
        - num_gpus: Number of detected GPUs

    Examples:
        >>> arch, count = get_amd_gpu_info()
        >>> print(f"Found {count} GPU(s) with architecture {arch}")
        Found 8 GPU(s) with architecture gfx942

    Raises:
        GPUDetectionError: If detection fails or amdsmi is not available
    """
    if not _HAS_AMDSMI:
        raise GPUDetectionError(
            "amdsmi library not available. "
            "Install with: pip install vllm-rocm-autotuner-configs[gpu-detect]"
        )

    try:
        amdsmi_init()
        devices = amdsmi_get_processor_handles()

        if not devices:
            _logger.warning("No AMD GPUs detected")
            return "unknown", 0

        all_info = []
        for dev in devices:
            try:
                info = amdsmi_get_gpu_asic_info(dev)
                all_info.append(info)
            except AmdSmiException as e:
                _logger.warning(
                    f"Failed to get info for device {dev}: {e}")  # noqa: G004
                continue

        if not all_info:
            raise GPUDetectionError("Failed to get info for any GPU")

        num_gpus = len(all_info)
        unique_device_ids = {info.get("device_id") for info in all_info}
        if len(unique_device_ids) == 1:
            gfx_version = all_info[0].get("target_graphics_version", "unknown")
            arch = _normalize_arch_name(gfx_version)
            _logger.debug("Detected %s GPU(s) with architecture %s", num_gpus,
                          arch)
            return arch, num_gpus
        else:
            _logger.warning(
                f"Detected mixed GPU architectures: {unique_device_ids}. "  # noqa: G004
                "Using 'mixed' as architecture name.")
            return "mixed", num_gpus

    except AmdSmiException as e:
        raise GPUDetectionError(f"AMDSMI error: {e}") from e
    finally:
        with contextlib.suppress(AmdSmiException, NameError):
            amdsmi_shut_down()


def _normalize_arch_name(gfx_version: str) -> str:
    """
    Normalize GPU architecture name to standard format.

    Args:
        gfx_version: Raw GPU architecture string from amdsmi

    Returns:
        Normalized architecture name (e.g., 'gfx942')

    Examples:
        >>> _normalize_arch_name('gfx942:sramecc+:xnack-')
        'gfx942'
        >>> _normalize_arch_name('gfx90a')
        'gfx90a'
    """
    if not gfx_version or gfx_version == "unknown":
        return "unknown"

    base_arch = gfx_version.split(
        ":")[0] if ":" in gfx_version else gfx_version
    base_arch = base_arch.strip().lower()

    return base_arch


def get_amd_gpu_info_safe() -> tuple[str | None, int]:
    """
    Safe version of get_amd_gpu_info that never raises exceptions.

    Returns:
        Tuple of (architecture_name, num_gpus)
        - architecture_name: GPU architecture string or None if detection failed
        - num_gpus: Number of detected GPUs, 0 if detection failed

    Examples:
        >>> arch, count = get_amd_gpu_info_safe()
        >>> if arch:
        ...     print(f"Detected {arch}")
        ... else:
        ...     print("GPU detection failed")
    """
    try:
        return get_amd_gpu_info()
    except GPUDetectionError as e:
        _logger.debug(f"GPU detection failed: {e}")  # noqa: G004
        return None, 0
    except Exception as e:
        _logger.warning(
            f"Unexpected error during GPU detection: {e}")  # noqa: G004
        return None, 0


def check_amdsmi_available() -> bool:
    """
    Check if amdsmi library is available.

    Returns:
        True if amdsmi is available, False otherwise

    Examples:
        >>> if check_amdsmi_available():
        ...     arch, count = get_amd_gpu_info()
        ... else:
        ...     print("Install amdsmi for GPU auto-detection")
    """
    return _HAS_AMDSMI


def clear_gpu_info_cache() -> None:
    """
    Clear the cached GPU information.

    Useful if GPUs are added/removed during runtime or for testing.

    Examples:
        >>> get_amd_gpu_info()  # Cached result
        >>> # ... GPU configuration changes ...
        >>> clear_gpu_info_cache()
        >>> get_amd_gpu_info()  # Fresh detection
    """
    get_amd_gpu_info.cache_clear()


def main() -> int:
    """
    Command-line interface for GPU detection.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect AMD GPU architecture and count")
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--json",
                        action="store_true",
                        help="Output in JSON format")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not check_amdsmi_available():
        print(
            "Error: amdsmi library not found.\n"
            "Install with: pip install vllm-rocm-autotuner-configs[gpu-detect]",
            file=sys.stderr,
        )
        return 1

    try:
        arch, num_gpus = get_amd_gpu_info()

        if args.json:
            import json

            result = {
                "architecture": arch,
                "num_gpus": num_gpus,
                "amdsmi_available": True,
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"Number of AMD GPUs: {num_gpus}")
            if num_gpus > 0:
                print(f"GPU Architecture: {arch}")
            else:
                print("No AMD GPUs detected")

        return 0

    except GPUDetectionError as e:
        if args.json:
            import json

            result = {
                "error": str(e),
                "architecture": None,
                "num_gpus": 0,
                "amdsmi_available": True,
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
