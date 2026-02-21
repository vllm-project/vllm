# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for Helion kernel management."""

import logging

from vllm.platforms import current_platform

logger = logging.getLogger(__name__)

# Maps known variant GPU names (after lowercase/underscore normalization)
# to their canonical form.
#
# Names that are already canonical after normalization are NOT listed here.
# For example, "NVIDIA H200" normalizes to "nvidia_h200" which needs no
# further mapping, and AMD ROCm names like "AMD_Instinct_MI300X" come from
# a controlled lookup table in rocm.py and normalize cleanly to
# "amd_instinct_mi300x". Only names with variant suffixes (form factor,
# memory size, memory type, etc.) that should be stripped need entries.
#
# To add a new GPU variant: run `canonicalize_gpu_name()` without the alias
# to see the normalized name, then add a mapping here if it contains variant
# suffixes that should be stripped (e.g. Blackwell/Rubin variants).
_GPU_NAME_ALIASES: dict[str, str] = {
    # H100 variants
    "nvidia_h100_pcie": "nvidia_h100",
    "nvidia_h100_sxm5": "nvidia_h100",
    "nvidia_h100_80gb_hbm3": "nvidia_h100",
    "nvidia_h100_nvl": "nvidia_h100",
    # H200 variants
    "nvidia_h200_nvl": "nvidia_h200",
    "nvidia_h200_141gb_hbm3e": "nvidia_h200",
    # A100 variants
    "nvidia_a100_sxm4_80gb": "nvidia_a100",
    "nvidia_a100_sxm4_40gb": "nvidia_a100",
    "nvidia_a100_pcie_80gb": "nvidia_a100",
    "nvidia_a100_pcie_40gb": "nvidia_a100",
    "nvidia_a100_80gb_pcie": "nvidia_a100",
    # V100 variants (Tesla-branded)
    "tesla_v100_sxm2_32gb": "tesla_v100",
    "tesla_v100_sxm2_16gb": "tesla_v100",
    "tesla_v100_pcie_32gb": "tesla_v100",
    "tesla_v100_pcie_16gb": "tesla_v100",
    # AMD ROCm variants (from _ROCM_DEVICE_ID_NAME_MAP in rocm.py)
    "amd_instinct_mi300x_hf": "amd_instinct_mi300x",
    # ADD MORE HERE
}


def get_gpu_name(device_id: int | None = None) -> str:
    if device_id is None:
        logger.warning(
            "get_gpu_name() called without device_id, defaulting to 0. "
            "This may return the wrong device name in multi-node setups."
        )
        device_id = 0
    return current_platform.get_device_name(device_id)


def canonicalize_gpu_name(name: str) -> str:
    """
    Canonicalize GPU name for use as a platform identifier.

    Converts to lowercase, replaces spaces and hyphens with underscores,
    and maps known variant names to their canonical form via _GPU_NAME_ALIASES.
    e.g., "NVIDIA H100 80GB HBM3" -> "nvidia_h100"
          "NVIDIA A100-SXM4-80GB" -> "nvidia_a100"
          "AMD Instinct MI300X"   -> "amd_instinct_mi300x"
    """
    if not name or not name.strip():
        raise ValueError("GPU name cannot be empty")
    name = name.lower()
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    if name in _GPU_NAME_ALIASES:
        return _GPU_NAME_ALIASES[name]
    return name


def get_canonical_gpu_name(device_id: int | None = None) -> str:
    return canonicalize_gpu_name(get_gpu_name(device_id))
