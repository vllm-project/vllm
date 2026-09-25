# SPDX-License-Identifier: Apache-2.0
"""POSIX shared memory double buffer with ROCm host registration for multi-rank direct I/O.

Allows a designated reader process (Rank 0) to stream safetensors shards directly
into shared host memory (/dev/shm) via O_DIRECT, while all local TP ranks (0..N-1)
concurrently slice parameters and execute zero-copy GPU DMA transfers.
"""

import atexit
import ctypes
import logging
from multiprocessing import shared_memory
import os
import signal
import struct
import time
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Default slot size: 16 GiB (sufficient for largest safetensors shard in modern LLMs)
DEFAULT_SLOT_SIZE = 16 * 1024 * 1024 * 1024

# ROCm HIP Host Register bindings
_HIP_LIB: ctypes.CDLL | None = None
_HIP_INIT_ATTEMPTED = False


def _get_hip_lib() -> ctypes.CDLL | None:
    global _HIP_LIB, _HIP_INIT_ATTEMPTED
    if _HIP_INIT_ATTEMPTED:
        return _HIP_LIB
    _HIP_INIT_ATTEMPTED = True
    try:
        lib = ctypes.CDLL("libamdhip64.so")
        lib.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        lib.hipHostRegister.restype = ctypes.c_int
        lib.hipHostUnregister.argtypes = [ctypes.c_void_p]
        lib.hipHostUnregister.restype = ctypes.c_int
        _HIP_LIB = lib
        logger.debug("Successfully loaded libamdhip64.so for host memory registration")
    except Exception as err:
        logger.debug("libamdhip64.so not available: %s", err)
        _HIP_LIB = None
    return _HIP_LIB


def register_rocm_host_memory(buf: memoryview) -> bool:
    """Registers host memory buffer with ROCm page tables (hipHostRegister)."""
    lib = _get_hip_lib()
    if lib is None:
        return False
    try:
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        res = lib.hipHostRegister(addr, len(buf), 0)
        return res == 0
    except Exception as e:
        logger.debug("hipHostRegister failed: %s", e)
        return False


def unregister_rocm_host_memory(buf: memoryview) -> bool:
    """Unregisters host memory buffer from ROCm page tables (hipHostUnregister)."""
    lib = _get_hip_lib()
    if lib is None:
        return False
    try:
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        res = lib.hipHostUnregister(addr)
        return res == 0
    except Exception as e:
        logger.debug("hipHostUnregister failed: %s", e)
        return False


class SharedPinnedBufferPool:
    """Double-buffered shared memory pool mapped across local TP processes."""

    def __init__(
        self,
        prefix: str,
        slot_size: int = DEFAULT_SLOT_SIZE,
        is_creator: bool = False,
        tp_rank: int = 0,
        tp_size: int = 1,
    ):
        self.prefix = prefix
        self.slot_size = slot_size
        self.is_creator = is_creator
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        self.slot_names = [
            f"{prefix}_slot_0",
            f"{prefix}_slot_1",
        ]
        self.ctrl_name = f"{prefix}_ctrl"

        self.slots: list[shared_memory.SharedMemory] = []
        self.ctrl_shm: shared_memory.SharedMemory | None = None
        self._registered_slots: list[bool] = [False, False]

        self._init_shared_memory()

        if self.is_creator:
            atexit.register(self.unlink)
            self._register_signal_traps()

    def _register_signal_traps(self) -> None:
        """Installs signal traps to ensure shared memory segments are unlinked on exit."""
        def _signal_handler(signum: int, frame: Any) -> None:
            self.unlink()
            signal.default_int_handler(signum, frame)

        try:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
        except (ValueError, RuntimeError):
            # Not in main thread
            pass

    def _init_shared_memory(self) -> None:
        """Creates or attaches to the POSIX shared memory slots."""
        for slot_idx, name in enumerate(self.slot_names):
            if self.is_creator:
                # Remove stale segment if left by a prior crashed process
                try:
                    old = shared_memory.SharedMemory(name=name)
                    old.close()
                    old.unlink()
                except FileNotFoundError:
                    pass
                shm = shared_memory.SharedMemory(create=True, size=self.slot_size, name=name)
            else:
                # Wait for creator to instantiate the segment
                shm = None
                deadline = time.time() + 60.0
                while time.time() < deadline:
                    try:
                        shm = shared_memory.SharedMemory(create=False, name=name)
                        break
                    except FileNotFoundError:
                        time.sleep(0.01)
                if shm is None:
                    raise TimeoutError(f"Timed out waiting for shared memory segment {name}")

            self.slots.append(shm)
            # Register with ROCm page tables if GPU is active
            if torch.cuda.is_available():
                self._registered_slots[slot_idx] = register_rocm_host_memory(shm.buf)

        # Initialize or attach to control segment (128 bytes: active_slot, shard_idx, seq, flags)
        ctrl_size = 128
        if self.is_creator:
            try:
                old_ctrl = shared_memory.SharedMemory(name=self.ctrl_name)
                old_ctrl.close()
                old_ctrl.unlink()
            except FileNotFoundError:
                pass
            self.ctrl_shm = shared_memory.SharedMemory(create=True, size=ctrl_size, name=self.ctrl_name)
            # Clear entire control buffer to zeros (barrier generations, status flags)
            self.ctrl_shm.buf[:ctrl_size] = b"\x00" * ctrl_size
        else:
            deadline = time.time() + 60.0
            while time.time() < deadline:
                try:
                    self.ctrl_shm = shared_memory.SharedMemory(create=False, name=self.ctrl_name)
                    break
                except FileNotFoundError:
                    time.sleep(0.01)
            if self.ctrl_shm is None:
                raise TimeoutError(f"Timed out waiting for control segment {self.ctrl_name}")

    def barrier(self) -> None:
        """Synchronizes all TP ranks."""
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            return

        if self.tp_size <= 1:
            return

        assert self.ctrl_shm is not None
        # Disjoint-slot sense barrier across standalone multiprocess workers
        cur_gen = struct.unpack_from("<I", self.ctrl_shm.buf, 0)[0]
        next_gen = cur_gen + 1
        # Each rank marks its own arrival at offset 16 + 4 * tp_rank
        struct.pack_into("<I", self.ctrl_shm.buf, 16 + 4 * self.tp_rank, next_gen)

        start_time = time.time()
        timeout = 60.0
        if self.tp_rank == 0:
            while True:
                arrivals = [
                    struct.unpack_from("<I", self.ctrl_shm.buf, 16 + 4 * r)[0]
                    for r in range(self.tp_size)
                ]
                if all(a == next_gen for a in arrivals):
                    break
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Rank 0 timed out waiting for barrier {next_gen}. Arrivals: {arrivals}"
                    )
                time.sleep(0.0005)
            # Release all waiting peer ranks
            struct.pack_into("<I", self.ctrl_shm.buf, 0, next_gen)
        else:
            while True:
                gen = struct.unpack_from("<I", self.ctrl_shm.buf, 0)[0]
                if gen == next_gen:
                    break
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Rank {self.tp_rank} timed out waiting for barrier {next_gen}. Current gen: {gen}"
                    )
                time.sleep(0.0005)

    def get_slot_buffer(self, slot_idx: int) -> memoryview:
        """Returns the memoryview buffer of the specified slot."""
        return self.slots[slot_idx].buf

    def close(self) -> None:
        """Closes local views without unlinking shared memory."""
        for slot_idx, shm in enumerate(self.slots):
            if self._registered_slots[slot_idx]:
                unregister_rocm_host_memory(shm.buf)
                self._registered_slots[slot_idx] = False
            try:
                shm.close()
            except Exception:
                pass
        self.slots.clear()

        if self.ctrl_shm is not None:
            try:
                self.ctrl_shm.close()
            except Exception:
                pass
            self.ctrl_shm = None

    def unlink(self) -> None:
        """Unlinks shared memory segments from /dev/shm (creator only)."""
        self.close()
        if self.is_creator:
            for name in self.slot_names:
                try:
                    shm = shared_memory.SharedMemory(name=name)
                    shm.unlink()
                except Exception:
                    pass
            try:
                ctrl = shared_memory.SharedMemory(name=self.ctrl_name)
                ctrl.unlink()
            except Exception:
                pass

    def __enter__(self) -> "SharedPinnedBufferPool":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.is_creator:
            self.unlink()
        else:
            self.close()
