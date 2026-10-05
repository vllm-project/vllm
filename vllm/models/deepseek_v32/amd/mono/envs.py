# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every environment variable the GLM-5.2 MonoKernel reads, in one place.

* ``VLLM_ROCM_USE_GLM5_MONOKERNEL``: enable switch ("1" / "true"). Read through
  ``vllm.envs`` when registered there.
* ``VLLM_ROCM_GLM5_MONOKERNEL_CONFIG``: JSON object of ``LiveConfig`` overrides.
* ``MONO_LIVE_FAILSTOP``: poll-error fail-stop, "1" (default: raise), "warn" or "0".
* ``MONO_FAULT_KERNEL``: test only, read at kernel build time (``fault_kernel``).
"""

from __future__ import annotations

import json
import os

ENABLE = "VLLM_ROCM_USE_GLM5_MONOKERNEL"
CONFIG = "VLLM_ROCM_GLM5_MONOKERNEL_CONFIG"
FAILSTOP = "MONO_LIVE_FAILSTOP"
FAULT_KERNEL = "MONO_FAULT_KERNEL"


def env_bool(name: str) -> bool:
    """Parse a bool the vLLM ROCm way: "1" / "true" (any case) -> True."""
    return os.getenv(name, "False").strip().lower() in ("true", "1")


def enabled() -> bool:
    """The enable switch, from vllm.envs if registered, else os.environ (envs'
    ``__getattr__`` raises for unknown names)."""
    from vllm import envs

    if ENABLE in envs.environment_variables:
        return bool(getattr(envs, ENABLE))
    return env_bool(ENABLE)


def config_overrides() -> dict:
    """``LiveConfig`` overrides from the JSON config variable ({} when unset)."""
    over = json.loads(os.environ.get(CONFIG, "") or "{}")
    if not isinstance(over, dict):
        raise ValueError(f"{CONFIG} must be a JSON object, got {over!r}")
    return over


def failstop_mode() -> str:
    """Fail-stop mode: "raise" (default), "warn" (log and continue) or "off" (expired
    steps then emit wrong tokens silently)."""
    v = os.environ.get(FAILSTOP, "1").strip().lower()
    return {"0": "off", "off": "off", "warn": "warn"}.get(v, "raise")


def fault_kernel() -> tuple[int, int, int] | None:
    """Test only: ``MONO_FAULT_KERNEL="rank=R,step=N,us=D"`` -> (R, N, D), else None."""
    spec = os.environ.get(FAULT_KERNEL)
    if not spec:
        return None
    kv = dict(x.split("=") for x in spec.split(","))
    return int(kv["rank"]), int(kv["step"]), int(kv["us"])
