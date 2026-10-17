# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform queries for the shared Inkling implementation.

Keeps per-platform device introspection out of the kernels, which otherwise
have to reach for CUDA-specific APIs to size their grids.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform


def num_compute_units(device: torch.device) -> int:
    """SM / CU / Xe-core count used to size persistent Triton grids."""
    index = device.index
    if index is None:
        index = torch.accelerator.current_device_index()
    return current_platform.num_compute_units(index)
