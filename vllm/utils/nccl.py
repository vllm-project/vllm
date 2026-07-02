# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ctypes
import importlib.util
import os

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
    group: torch.distributed.ProcessGroup,
) -> int | None:
    """Query NCCL GIN (GPU-Initiated Networking) support for a process group.

    Returns the ``ncclGinType_t`` value (0 = NONE, 2 = PROXY, 3 = GDAKI,
    4 = GPI), or ``None`` if the query could not be performed (e.g. NCCL
    too old or the PyTorch backend does not expose the comm pointer).
    """
    from vllm.distributed.device_communicators.pynccl_wrapper import (
        NCCLLibrary,
        ncclCommProperties,
    )

    try:
        backend = group._get_backend(torch.device("cuda"))
        if not hasattr(backend, "_comm_ptr"):
            logger.warning("PyTorch NCCL backend does not expose _comm_ptr")
            return None
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
    except Exception:
        logger.warning("Failed to load NCCL library", exc_info=True)
        return None

    query_fn = nccl._funcs.get("ncclCommQueryProperties")
    if query_fn is None:
        logger.warning("ncclCommQueryProperties not available (NCCL < 2.29)")
        return None

    props = ncclCommProperties()
    ctypes.memset(ctypes.addressof(props), 0, ctypes.sizeof(props))
    # NCCL validates these fields (mirrors NCCL_COMM_PROPERTIES_INITIALIZER)
    props.size = ctypes.sizeof(props)
    props.magic = 0xCAFEBEEF
    props.version = nccl.ncclGetRawVersion()

    try:
        result = query_fn(ctypes.c_void_p(comm_ptr), ctypes.byref(props))
    except Exception:
        logger.warning("ncclCommQueryProperties call failed", exc_info=True)
        return None

    if result != 0:
        logger.warning("ncclCommQueryProperties returned error %d", result)
        return None

    return props.ginType
