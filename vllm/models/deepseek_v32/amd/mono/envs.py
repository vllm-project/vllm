# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment variables of the GLM-5.2 MonoKernel. The enable switch
``VLLM_ROCM_USE_GLM5_MONOKERNEL`` is registered in ``vllm.envs``."""

from __future__ import annotations

import json
import os

ENABLE = "VLLM_ROCM_USE_GLM5_MONOKERNEL"
# JSON object of LiveConfig overrides
CONFIG = "VLLM_ROCM_GLM5_MONOKERNEL_CONFIG"
# poll-error fail-stop: "1" (default: raise), "warn" or "0"
FAILSTOP = "MONO_LIVE_FAILSTOP"
# test only: "rank=R,step=N,us=D" stalls rank R at kernel step N (read at build time)
FAULT_KERNEL = "MONO_FAULT_KERNEL"


def config_overrides() -> dict:
    over = json.loads(os.environ.get(CONFIG, "") or "{}")
    if not isinstance(over, dict):
        raise ValueError(f"{CONFIG} must be a JSON object, got {over!r}")
    return over


def failstop_mode() -> str:
    """Fail-stop mode: "raise" (default), "warn" (log, continue) or "off" (expired
    steps emit wrong tokens silently)."""
    v = os.environ.get(FAILSTOP, "1").strip().lower()
    return {"0": "off", "off": "off", "warn": "warn"}.get(v, "raise")


def fault_kernel() -> tuple[int, int, int] | None:
    spec = os.environ.get(FAULT_KERNEL)
    if not spec:
        return None
    kv = dict(x.split("=") for x in spec.split(","))
    return int(kv["rank"]), int(kv["step"]), int(kv["us"])
