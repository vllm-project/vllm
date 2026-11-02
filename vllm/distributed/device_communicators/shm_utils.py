# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import platform
import sys
from multiprocessing import shared_memory


def attach_shared_memory(name: str | None) -> shared_memory.SharedMemory:
    """Attach without taking the creator's resource-tracker/unlink ownership."""
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    # Python 3.11/3.12 have no public non-tracking attachment parameter.
    from unittest.mock import patch

    with patch("multiprocessing.resource_tracker.register", lambda *a, **k: None):
        return shared_memory.SharedMemory(name=name)


def resolve_shm_wait_backend(mode: str) -> tuple[str, str | None]:
    """Resolve local-writer policy; never hide a broken native installation."""
    if mode not in ("auto", "stock", "futex"):
        raise ValueError("VLLM_ASA_SHM_WAIT must be auto, stock, or futex")
    if mode == "stock":
        return "stock", None

    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    supported = (
        sys.platform.startswith("linux")
        and platform.machine().lower() in ("x86_64", "amd64", "aarch64", "arm64")
        and sys.maxsize > 2**32
        and sys.implementation.name == "cpython"
        and gil_enabled
    )
    if not supported:
        reason = (
            "futex requires 64-bit Linux x86_64/aarch64 CPython with the GIL enabled"
        )
        if mode == "futex":
            raise ValueError(reason)
        return "stock", reason

    try:
        importlib.import_module("vllm._asa_shmwait")
    except ModuleNotFoundError as error:
        if mode != "auto" or error.name != "vllm._asa_shmwait":
            raise
        return "stock", "optional vllm._asa_shmwait extension is not installed"
    return "futex", None
