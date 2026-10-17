# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import ctypes
import importlib.util
import os
from collections.abc import Iterator, Mapping

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def find_nccl_library() -> str:
    """Return NCCL/RCCL shared library name to load.

    Uses `VLLM_NCCL_SO_PATH` if set; otherwise chooses by torch backend.
    """
    so_file = envs.VLLM_NCCL_SO_PATH
    if so_file:
        logger.info(
            "Found nccl from environment variable VLLM_NCCL_SO_PATH=%s", so_file
        )
    else:
        if torch.version.cuda is not None:
            so_file = "libnccl.so.2"
        elif torch.version.hip is not None:
            so_file = "librccl.so.1"
        else:
            raise ValueError("NCCL only supports CUDA and ROCm backends.")
        logger.debug_once("Found nccl from library %s", so_file)
    return so_file


def find_nccl_include_paths() -> list[str] | None:
    """Return possible include paths containing `nccl.h`.

    Considers `VLLM_NCCL_INCLUDE_PATH` and the `nvidia-nccl-cuXX` package.
    """
    paths: list[str] = []
    inc = envs.VLLM_NCCL_INCLUDE_PATH
    if inc and os.path.isdir(inc):
        paths.append(inc)

    try:
        spec = importlib.util.find_spec("nvidia.nccl")
        if spec and (locs := getattr(spec, "submodule_search_locations", None)):
            for loc in locs:
                inc_dir = os.path.join(loc, "include")
                if os.path.exists(os.path.join(inc_dir, "nccl.h")):
                    paths.append(inc_dir)
    except Exception as e:
        logger.debug("Failed to find nccl include path from nvidia.nccl package: %s", e)

    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out or None


def find_nccl_library_paths() -> list[str] | None:
    """Return possible library paths containing `libnccl.so`.

    Looks inside the `nvidia-nccl-cuXX` pip package.
    """
    paths: list[str] = []
    try:
        spec = importlib.util.find_spec("nvidia.nccl")
        if spec and (locs := getattr(spec, "submodule_search_locations", None)):
            for loc in locs:
                lib_dir = os.path.join(loc, "lib")
                if os.path.isdir(lib_dir):
                    paths.append(lib_dir)
    except Exception as e:
        logger.debug("Failed to find nccl library path from nvidia.nccl package: %s", e)
    return paths or None


def query_nccl_gin_type(
    group: torch.distributed.ProcessGroup, *, railed: bool = False
) -> int | None:
    """Return the full or railed GIN type, or ``None`` on query failure."""
    from vllm.distributed.device_communicators.pynccl_wrapper import (
        NCCL_COMM_PROPERTIES_LAYOUT_VERSION,
        NCCLLibrary,
        ncclCommProperties,
    )

    try:
        backend = group._get_backend(torch.device("cuda"))
        # GIN is a property of this initialized communicator, not just the
        # NCCL version. ncclCommQueryProperties requires its ncclComm_t.
        comm_ptr = backend._comm_ptr()
        if comm_ptr == 0:
            return None
    except Exception:
        logger.warning(
            "Failed to extract NCCL comm pointer from process group",
            exc_info=True,
        )
        return None

    try:
        nccl = NCCLLibrary()
        query_fn = nccl._funcs.get("ncclCommQueryProperties")
        if query_fn is None:
            return None

        props = ncclCommProperties()
        ctypes.memset(ctypes.addressof(props), 0, ctypes.sizeof(props))
        props.size = ctypes.sizeof(props)
        props.magic = 0xCAFEBEEF
        props.version = min(
            nccl.ncclGetRawVersion(), NCCL_COMM_PROPERTIES_LAYOUT_VERSION
        )
        result = query_fn(ctypes.c_void_p(comm_ptr), ctypes.byref(props))
    except Exception:
        logger.warning("Failed to query NCCL communicator properties", exc_info=True)
        return None

    if result != 0:
        logger.warning("ncclCommQueryProperties returned error %d", result)
        return None
    return props.railedGinType if railed else props.ginType


# Values the variables set by `pin_nccl_env` had before it: None until something
# is pinned, empty if NCCL cannot re-read them.
_unpinned_env: dict[str, str | None] | None = None


def pin_nccl_env(pins: dict[str, str]) -> None:
    """Set NCCL variables for this process's own communicators.

    NCCL never reconciles these across ranks, so communicators shared with a
    process that does not set them are created without them, see
    `unpinned_nccl_env`. NCCL caches most variables on first read; listing the
    pins in NCCL_NO_CACHE makes it re-read them per communicator. NCCL parses
    that list once, so this must run before the process's first NCCL call.
    """
    global _unpinned_env
    launch_env = _swap_env(pins)
    if _unpinned_env is not None:
        return
    _unpinned_env = launch_env if _nccl_has_no_cache() else {}
    if _unpinned_env:
        no_cache = os.environ.get("NCCL_NO_CACHE")
        os.environ["NCCL_NO_CACHE"] = ",".join(filter(None, [no_cache, *pins]))


@contextlib.contextmanager
def unpinned_nccl_env() -> Iterator[None]:
    """Create communicators shared with other processes without the pins."""
    if _unpinned_env == {}:
        logger.warning_once(
            "NCCL_NO_CACHE needs CUDA NCCL >= 2.29.7, so the NCCL settings vLLM "
            "pinned also apply here; a process sharing this communicator must "
            "set the same NCCL_* variables or it will hang."
        )
    pinned = _swap_env(_unpinned_env or {})
    try:
        yield
    finally:
        _swap_env(pinned)


def _nccl_has_no_cache() -> bool:
    # NCCL_NO_CACHE arrived in NCCL 2.29.7; ask the library vLLM loads.
    if torch.version.cuda is None:
        return False
    version = ctypes.c_int()
    try:
        ctypes.CDLL(find_nccl_library()).ncclGetVersion(ctypes.byref(version))
    except Exception:
        return False
    return version.value >= 22907


def _swap_env(values: Mapping[str, str | None]) -> dict[str, str | None]:
    """Set (None: unset) the given variables and return their previous values."""
    old = {name: os.environ.get(name) for name in values}
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    return old
