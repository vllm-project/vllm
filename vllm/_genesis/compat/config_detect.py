# SPDX-License-Identifier: Apache-2.0
"""Compat re-export shim — `vllm._genesis.compat.config_detect`.

See `compat/gpu_profile.py` for the migration rationale.

Author: Sandermage(Sander) Barzov Aleksandr.
"""
from __future__ import annotations

from vllm._genesis.config_detect import *  # noqa: F401, F403
from vllm._genesis import config_detect as _legacy

__all__ = list(getattr(_legacy, "__all__", [
    name for name in dir(_legacy) if not name.startswith("_")
]))

for _name in __all__:
    globals()[_name] = getattr(_legacy, _name)
