# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
UCX-based one-shot AllReduce for DP metadata synchronization.

Replaces Gloo's TCP AllReduce with UCX tag-matching over IB RDMA.
Falls back to Gloo if UCX is unavailable or init fails.

Usage:
    from vllm.distributed.device_communicators.ucx_dp_communicator import (
        try_init_ucx_dp,
        get_ucx_dp_communicator,
    )

    # Once, after DP group is created:
    try_init_ucx_dp(dp_rank, dp_size, gloo_group)

    # Per-iteration:
    comm = get_ucx_dp_communicator()
    if comm is not None:
        comm.allreduce_inplace(tensor)  # UCX/RDMA path
    else:
        dist.all_reduce(tensor, group=gloo_group)  # fallback
"""

import ctypes
import ctypes.util
import logging
import os
import subprocess
import threading

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_communicator: "UCXDPCommunicator | None" = None
_init_lock = threading.Lock()


def get_ucx_dp_communicator() -> "UCXDPCommunicator | None":
    return _communicator


def try_init_ucx_dp(
    rank: int,
    world_size: int,
    gloo_group,
    max_msg_bytes: int = 1024,
) -> bool:
    global _communicator
    with _init_lock:
        if _communicator is not None:
            return True
        try:
            comm = UCXDPCommunicator(rank, world_size, max_msg_bytes)
            comm.bootstrap(gloo_group)
            _communicator = comm
            logger.info(
                "UCX DP communicator ready (rank %d/%d, RDMA allreduce)",
                rank,
                world_size,
            )
            return True
        except Exception:
            logger.warning(
                "UCX DP communicator unavailable, using Gloo fallback",
                exc_info=True,
            )
            return False


class UCXDPCommunicator:
    def __init__(
        self,
        rank: int,
        world_size: int,
        max_msg_bytes: int = 1024,
    ):
        self.rank = rank
        self.world_size = world_size
        self._lib = _load_library()
        self._state = ctypes.c_void_p()

        addr_ptr = ctypes.c_void_p()
        addr_len = ctypes.c_size_t()
        rc = self._lib.ucx_dp_init(
            rank,
            world_size,
            max_msg_bytes,
            ctypes.byref(self._state),
            ctypes.byref(addr_ptr),
            ctypes.byref(addr_len),
        )
        if rc != 0:
            raise RuntimeError("ucx_dp_init failed")

        assert addr_ptr.value is not None
        self._address = ctypes.string_at(addr_ptr.value, addr_len.value)
        self._lib.ucx_dp_release_address(self._state, addr_ptr)

    def bootstrap(self, gloo_group) -> None:
        """Exchange UCX worker addresses via existing Gloo group."""
        all_addrs: list = [None] * self.world_size
        dist.all_gather_object(all_addrs, self._address, group=gloo_group)

        for i, addr in enumerate(all_addrs):
            if i == self.rank:
                continue
            rc = self._lib.ucx_dp_connect(
                self._state,
                i,
                addr,
                len(addr),
            )
            if rc != 0:
                raise RuntimeError(f"ucx_dp_connect to rank {i} failed")

    def allreduce_inplace(self, tensor: torch.Tensor) -> None:
        """In-place sum allreduce of a contiguous CPU int32
        tensor."""
        assert tensor.is_contiguous() and tensor.device.type == "cpu"
        nbytes = tensor.nelement() * tensor.element_size()
        rc = self._lib.ucx_dp_allreduce_inplace(
            self._state,
            ctypes.c_void_p(tensor.data_ptr()),
            nbytes,
        )
        if rc != 0:
            raise RuntimeError("ucx_dp_allreduce_inplace failed")

    def finalize(self) -> None:
        if self._state:
            self._lib.ucx_dp_finalize(self._state)
            self._state = ctypes.c_void_p()

    def __del__(self):
        self.finalize()


# ---- library loading ----

_lib_cache: ctypes.CDLL | None = None


def _load_library() -> ctypes.CDLL:
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache

    so_path = _find_or_compile()
    if so_path is None:
        raise RuntimeError(
            "Cannot find or compile _ucx_dp_sync.so. "
            "Place ucx_dp_sync.c next to this file, or set "
            "VLLM_UCX_DP_LIB to the .so path."
        )

    lib = ctypes.CDLL(so_path)
    _setup_signatures(lib)
    _lib_cache = lib
    return lib


def _setup_signatures(lib: ctypes.CDLL) -> None:
    c_int = ctypes.c_int
    c_size_t = ctypes.c_size_t
    c_void_p = ctypes.c_void_p
    POINTER = ctypes.POINTER

    lib.ucx_dp_init.restype = c_int
    lib.ucx_dp_init.argtypes = [
        c_int,
        c_int,
        c_size_t,
        POINTER(c_void_p),
        POINTER(c_void_p),
        POINTER(c_size_t),
    ]

    lib.ucx_dp_release_address.restype = None
    lib.ucx_dp_release_address.argtypes = [c_void_p, c_void_p]

    lib.ucx_dp_connect.restype = c_int
    lib.ucx_dp_connect.argtypes = [
        c_void_p,
        c_int,
        c_void_p,
        c_size_t,
    ]

    lib.ucx_dp_allreduce_inplace.restype = c_int
    lib.ucx_dp_allreduce_inplace.argtypes = [
        c_void_p,
        c_void_p,
        c_size_t,
    ]

    lib.ucx_dp_finalize.restype = None
    lib.ucx_dp_finalize.argtypes = [c_void_p]


def _find_or_compile() -> str | None:
    env_path = os.environ.get("VLLM_UCX_DP_LIB")
    if env_path and os.path.isfile(env_path):
        return env_path

    here = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, "_ucx_dp_sync.so")
    if os.path.isfile(so_path):
        return so_path

    c_path = os.path.join(here, "ucx_dp_sync.c")
    if not os.path.isfile(c_path):
        return None

    if not _has_ucp():
        logger.warning("libucp.so not found — cannot compile UCX DP sync")
        return None

    logger.info("Compiling _ucx_dp_sync.so …")
    cflags: list[str] = []
    ldflags: list[str] = []
    try:
        pc = subprocess.run(
            ["pkg-config", "--cflags", "--libs", "ucx"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if pc.returncode == 0:
            for tok in pc.stdout.strip().split():
                if tok.startswith(("-I", "-D")):
                    cflags.append(tok)
                else:
                    ldflags.append(tok)
    except Exception:
        pass

    cmd = [
        "gcc",
        "-shared",
        "-fPIC",
        "-O2",
        *cflags,
        "-o",
        so_path,
        c_path,
        "-lucp",
        "-lucs",
        *ldflags,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        logger.warning("Compile failed: %s", result.stderr)
        return None

    return so_path


def _has_ucp() -> bool:
    return ctypes.util.find_library("ucp") is not None
