# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for Helion kernel management."""

import torch


def get_gpu_name(device_id: int | None = None) -> str:
    if device_id is None:
        device_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_id)
    return props.name


def canonicalize_gpu_name(name: str) -> str:
    """
    Canonicalize GPU name for use as a platform identifier.

    Converts to lowercase and replaces spaces and hyphens with underscores.
    e.g., "NVIDIA A100-SXM4-80GB" -> "nvidia_a100_sxm4_80gb"

    Raises ValueError if name is empty.
    """
    if not name or not name.strip():
        raise ValueError("GPU name cannot be empty")
    name = name.lower()
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    return name


def get_canonical_gpu_name(device_id: int | None = None) -> str:
    return canonicalize_gpu_name(get_gpu_name(device_id))
