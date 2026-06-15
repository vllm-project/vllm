# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL communicator suspend/resume for sleep mode (NCCL >= 2.29.7).

Releases the GPU memory held by idle NCCL communicators while vLLM is asleep,
using NCCL's native ``ncclCommSuspend`` / ``ncclCommResume`` (memory manager).
Topology and connection state are preserved, so resume is cheap (no re-init /
no bootstrap rendezvous). On NCCL < 2.29.7 the API is absent and every entry
point is a graceful no-op (``is_supported()`` returns False).

This is a ctypes shim over the ``libnccl.so`` that torch already loaded, so the
``ncclComm_t`` handles (raw pointers owned by vLLM's ``PyNcclCommunicator``)
refer to the same library state.
"""

from __future__ import annotations

import ctypes

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Bitmask for ncclCommSuspend: release dynamic GPU memory allocations while
# keeping topology / connection state.
NCCL_SUSPEND_MEM = 0x01
NCCL_SUCCESS = 0

_nccl_lib: ctypes.CDLL | None = None
_load_attempted = False


def _get_nccl_lib() -> ctypes.CDLL | None:
    """Resolve the ``libnccl.so`` torch already loaded and bind the API.

    Returns None if NCCL is not loaded or is older than 2.29.7 (no suspend
    symbols).
    """
    global _nccl_lib, _load_attempted
    if _load_attempted:
        return _nccl_lib
    _load_attempted = True

    try:
        import psutil

        nccl_path = next(
            (
                m.path
                for m in psutil.Process().memory_maps()
                if "libnccl.so" in m.path.lower()
            ),
            None,
        )
    except Exception:
        nccl_path = None

    if nccl_path is None:
        logger.warning(
            "libnccl not found in process memory maps; NCCL suspend/resume disabled."
        )
        return None

    try:
        lib = ctypes.CDLL(nccl_path)
    except OSError as e:
        logger.warning("Failed to load %s: %s; NCCL suspend disabled.", nccl_path, e)
        return None

    if not hasattr(lib, "ncclCommSuspend") or not hasattr(lib, "ncclCommResume"):
        logger.warning(
            "NCCL at %s lacks ncclCommSuspend/Resume (needs >= 2.29.7); NCCL "
            "memory offload disabled.",
            nccl_path,
        )
        return None

    lib.ncclCommSuspend.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ncclCommSuspend.restype = ctypes.c_int
    lib.ncclCommResume.argtypes = [ctypes.c_void_p]
    lib.ncclCommResume.restype = ctypes.c_int
    logger.info("NCCL suspend/resume enabled via %s", nccl_path)
    _nccl_lib = lib
    return lib


def is_supported() -> bool:
    """Whether the loaded NCCL supports ncclCommSuspend/Resume (>= 2.29.7)."""
    return _get_nccl_lib() is not None


def _collect_pynccl_comm_handles() -> list[tuple[str, int]]:
    """Enumerate ``(name, ncclComm_t)`` for every active PyNcclCommunicator.

    Walks the well-known process-group coordinators (TP / PP / DP / EP / ... /
    world) and reads the raw ``ncclComm_t`` from each one's device
    communicator. Handles are de-duplicated by pointer value because a single
    communicator can be shared across coordinators.
    """
    from vllm.distributed import parallel_state as ps

    handles: list[tuple[str, int]] = []
    seen: set[int] = set()

    group_getters = {
        "world": "_WORLD",
        "tp": "_TP",
        "dcp": "_DCP",
        "pp": "_PP",
        "dp": "_DP",
        "ep": "_EP",
        "eplb": "_EPLB",
        "pcp": "_PCP",
    }
    for name, attr in group_getters.items():
        group = getattr(ps, attr, None)
        if group is None:
            continue
        device_comm = getattr(group, "device_communicator", None)
        if device_comm is None:
            continue
        pynccl = getattr(device_comm, "pynccl_comm", None)
        if pynccl is None or getattr(pynccl, "disabled", True):
            continue
        comm = getattr(pynccl, "comm", None)
        if comm is None:
            continue
        # ncclComm_t is a ctypes.c_void_p; .value is the raw pointer int.
        ptr = comm.value if isinstance(comm, ctypes.c_void_p) else int(comm)
        if not ptr or ptr in seen:
            continue
        seen.add(ptr)
        handles.append((name, ptr))
    return handles


def _gpu_used_bytes() -> int:
    free, total = torch.cuda.mem_get_info()
    return total - free


def suspend_nccl_comms() -> dict:
    """Suspend (release the memory of) all active PyNccl communicators.

    Must be called collectively on every rank with the communicators idle;
    ``ncclCommSuspend`` performs an internal cross-rank barrier. Returns a
    small telemetry dict. No-op (and reports ``skipped``) if NCCL is too old
    or there are no warm communicators.
    """
    lib = _get_nccl_lib()
    if lib is None:
        return {"skipped": "nccl_suspend_unsupported", "freed_bytes": 0, "n": 0}

    handles = _collect_pynccl_comm_handles()
    if not handles:
        return {"skipped": "no_pynccl_comms", "freed_bytes": 0, "n": 0}

    before = _gpu_used_bytes()
    n_ok = 0
    for name, ptr in handles:
        rc = lib.ncclCommSuspend(ctypes.c_void_p(ptr), NCCL_SUSPEND_MEM)
        if rc != NCCL_SUCCESS:
            logger.warning("ncclCommSuspend(%s) failed rc=%d", name, rc)
        else:
            n_ok += 1
    freed = before - _gpu_used_bytes()
    logger.info(
        "NCCL suspend: %d/%d comms, freed %.1f MiB",
        n_ok,
        len(handles),
        freed / 1024**2,
    )
    return {"freed_bytes": freed, "n": n_ok, "total": len(handles)}


def resume_nccl_comms() -> dict:
    """Resume all previously suspended PyNccl communicators.

    Must be called collectively on every rank before any collective op uses the
    communicators again. No-op if NCCL is too old or there are no comms.
    """
    lib = _get_nccl_lib()
    if lib is None:
        return {"skipped": "nccl_suspend_unsupported", "reclaimed_bytes": 0, "n": 0}

    handles = _collect_pynccl_comm_handles()
    if not handles:
        return {"skipped": "no_pynccl_comms", "reclaimed_bytes": 0, "n": 0}

    before = _gpu_used_bytes()
    n_ok = 0
    for name, ptr in handles:
        rc = lib.ncclCommResume(ctypes.c_void_p(ptr))
        if rc != NCCL_SUCCESS:
            logger.warning("ncclCommResume(%s) failed rc=%d", name, rc)
        else:
            n_ok += 1
    torch.accelerator.synchronize()
    reclaimed = _gpu_used_bytes() - before
    logger.info(
        "NCCL resume: %d/%d comms, reclaimed %.1f MiB",
        n_ok,
        len(handles),
        reclaimed / 1024**2,
    )
    return {"reclaimed_bytes": reclaimed, "n": n_ok, "total": len(handles)}
