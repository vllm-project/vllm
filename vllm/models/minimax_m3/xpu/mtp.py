# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 MTP — Intel XPU entry point.

Importing ``xpu.model`` installs the XPU ``MiniMAXGemmaRMSNorm`` into the NVIDIA
namespaces, so the reused NVIDIA MTP builds with the XPU norm.
"""

import vllm.models.minimax_m3.xpu.model  # noqa: F401  (installs XPU RMSNorm)
from vllm.models.minimax_m3.nvidia.mtp import MiniMaxM3MTP

__all__ = ["MiniMaxM3MTP"]
