# SPDX-License-Identifier: Apache-2.0
"""Genesis kernel-level drop-in replacements for vLLM weak spots.

Each kernel module provides a professional, platform-aware implementation
of a vLLM code path that was identified as broken, suboptimal, or crude.

Design goals per kernel:
  - Works on NVIDIA CUDA / AMD ROCm / Intel XPU / CPU (graceful skip).
  - Matches or exceeds upstream behavior in all metrics.
  - TDD-first: test suite before implementation.
  - Upstream-ready: code structure suitable for submission as vLLM PR.

Modules:
  router_softmax   — fp32-upcast MoE router softmax (Patch 31, universal)
  dequant_buffer   — TurboQuant shared pre-allocation manager (Patch 22)
  gdn_dual_stream  — Platform-aware dual-stream dispatcher (Patch 7)
  marlin_tuning    — Per-SM Marlin kernel auto-tuner (Patch 17/18)
  fp8_dispatcher   — Ampere/Ada/Hopper FP8 path selector (Patch 1/2)

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from vllm._genesis.kernels.router_softmax import router_softmax

__all__ = [
    "router_softmax",
]
