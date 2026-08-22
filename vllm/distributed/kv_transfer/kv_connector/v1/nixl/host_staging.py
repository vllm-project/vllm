# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage writes to host-memory KV through device memory.

Direct same-node device-to-host transfers may fall back to TCP loopback. This
module instead reads into device staging buffers, then copies into the host
destination. Requests complete only after every chunk reaches host memory.
"""

from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_CUDA_MEMCPY_DEVICE_TO_HOST = 2
_SHUTDOWN_TIMEOUT_S = 2.0
_SHUTDOWN_POLL_INTERVAL_S = 0.001


def _load_cudart() -> ctypes.CDLL | None:
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        lib.cudaMemcpyAsync.restype = ctypes.c_int
        lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        return lib
    return None


class _Pool:
    """Staging descriptors for a single descriptor length."""

    def __init__(
        self,
        *,
        desc_len: int,
        device: torch.device,
        nixl_wrapper: Any,
        memory_type: str,
        backends: Any,
        stage_bytes: int,
        num_slots: int,
    ) -> None:
        self.desc_len = desc_len
        self.nixl_wrapper = nixl_wrapper
        # Each slot holds a whole number of descriptors so a descriptor is
        # never torn across slots.
        self.descs_per_slot = max(stage_bytes // (desc_len * num_slots), 1)
        total_descs = self.descs_per_slot * num_slots
        self.buffer = torch.empty(
            total_descs * desc_len, dtype=torch.uint8, device=device
        )
        base = self.buffer.data_ptr()
        device_id = device.index or 0
        # The staging buffer must be registered with NIXL before descriptors
        # over it can be prepared, exactly as the KV caches are.
        self.reg_descs = nixl_wrapper.get_reg_descs(
            [(base, self.buffer.numel(), device_id, "")], memory_type
        )
        nixl_wrapper.register_memory(self.reg_descs, backends=backends)
        blocks = [
            (base + i * desc_len, desc_len, device_id) for i in range(total_descs)
        ]
        descs = nixl_wrapper.get_xfer_descs(blocks, memory_type)
        self.handle = nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
        self.stage_ptrs = np.array(
            [base + i * desc_len for i in range(total_descs)], dtype=np.uint64
        )
        self.slots = [
            _Slot(
                pool=self,
                event=torch.cuda.Event(),
                desc_ids=np.arange(
                    s * self.descs_per_slot,
                    (s + 1) * self.descs_per_slot,
                    dtype=np.int64,
                ),
            )
            for s in range(num_slots)
        ]
        self.free_slots = list(self.slots)
        self.bytes = total_descs * desc_len


@dataclass
class _Slot:
    """One staging region plus the event guarding its in-flight copy."""

    pool: _Pool
    event: torch.cuda.Event
    # Descriptor ids into this pool's staging dlist.
    desc_ids: np.ndarray


@dataclass
class _ReqState:
    """Per-request pipeline state: chunks waiting, reading, and copying."""

    # (remote_desc_ids, local_desc_ids, remote_handle, desc_len)
    queued: list[tuple[np.ndarray, np.ndarray, int, int]] = field(default_factory=list)
    # (xfer_handle, slot, local_desc_ids)
    reading: list[tuple[int, _Slot, np.ndarray]] = field(default_factory=list)
    copying: list[_Slot] = field(default_factory=list)
    failed: bool = False
    aborted: bool = False


class HostWriteStager:
    """Pipelines remote reads into host destinations through device memory."""

    def __init__(
        self,
        *,
        desc_lens: np.ndarray,
        host_addrs: np.ndarray,
        device: torch.device,
        nixl_wrapper: Any,
        memory_type: str,
        backends: Any,
        stage_bytes: int,
        num_slots: int,
    ) -> None:
        self.desc_lens = desc_lens
        self.host_addrs = host_addrs
        self.nixl_wrapper = nixl_wrapper

        cudart = _load_cudart()
        if cudart is None:
            raise RuntimeError(
                "host KV write staging requires libcudart for device-to-host "
                "copies; set VLLM_NIXL_HOST_STAGE_BYTES=0 to disable"
            )
        self._cudart: ctypes.CDLL = cudart

        lengths = sorted({int(x) for x in np.unique(desc_lens)})
        per_pool_bytes = max(stage_bytes // len(lengths), 1)
        self._pools: dict[int, _Pool] = {
            length: _Pool(
                desc_len=length,
                device=device,
                nixl_wrapper=nixl_wrapper,
                memory_type=memory_type,
                backends=backends,
                stage_bytes=per_pool_bytes,
                num_slots=num_slots,
            )
            for length in lengths
        }
        self._copy_stream = torch.cuda.Stream(device=device)
        self._reqs: dict[str, _ReqState] = {}
        self._closed = False
        self._shutdown_thread: threading.Thread | None = None

        logger.info(
            "NIXL host write staging enabled: %.2f GiB device staging across "
            "%d descriptor length(s) %s, %d slots each",
            sum(pool.bytes for pool in self._pools.values()) / 2**30,
            len(lengths),
            lengths,
            num_slots,
        )

    @property
    def active_req_ids(self) -> set[str]:
        return set(self._reqs)

    def submit(
        self,
        req_id: str,
        remote_desc_ids: np.ndarray,
        local_desc_ids: np.ndarray,
        remote_handle: int,
    ) -> None:
        """Queue a read, split by descriptor length then chunked, and start it."""
        remote_desc_ids = np.asarray(remote_desc_ids)
        local_desc_ids = np.asarray(local_desc_ids)
        state = self._reqs.setdefault(req_id, _ReqState())
        if state.aborted:
            raise RuntimeError(f"host staging request {req_id!r} is draining")
        lengths = self.desc_lens[local_desc_ids]
        for length, pool in self._pools.items():
            mask = lengths == length
            if not mask.any():
                continue
            remote_group = remote_desc_ids[mask]
            local_group = local_desc_ids[mask]
            chunk = pool.descs_per_slot
            for i in range(0, len(remote_group), chunk):
                state.queued.append(
                    (
                        remote_group[i : i + chunk],
                        local_group[i : i + chunk],
                        remote_handle,
                        length,
                    )
                )
        self._pump(state)

    def _pump(self, state: _ReqState) -> None:
        """Post queued chunks whose pool has a free slot."""
        remaining: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        for entry in state.queued:
            remote_ids, local_ids, remote_handle, length = entry
            pool = self._pools[length]
            if not pool.free_slots:
                remaining.append(entry)
                continue
            slot = pool.free_slots.pop()
            stage_ids = slot.desc_ids[: len(remote_ids)]
            handle = None
            try:
                handle = self.nixl_wrapper.make_prepped_xfer(
                    "READ", pool.handle, stage_ids, remote_handle, remote_ids
                )
                self.nixl_wrapper.transfer(handle)
            except Exception:
                state.failed = True
                state.queued = []
                if handle is None:
                    pool.free_slots.append(slot)
                else:
                    state.reading.append((handle, slot, local_ids))
                raise
            state.reading.append((handle, slot, local_ids))
        state.queued = remaining

    def _start_copy(self, slot: _Slot, local_ids: np.ndarray) -> None:
        """Copy a completed chunk from staging into its host destinations."""
        stream = self._copy_stream
        pool = slot.pool
        with torch.cuda.stream(stream):
            try:
                for j, local_id in enumerate(local_ids):
                    rc = self._cudart.cudaMemcpyAsync(
                        ctypes.c_void_p(int(self.host_addrs[local_id])),
                        ctypes.c_void_p(int(pool.stage_ptrs[slot.desc_ids[j]])),
                        ctypes.c_size_t(pool.desc_len),
                        ctypes.c_int(_CUDA_MEMCPY_DEVICE_TO_HOST),
                        ctypes.c_void_p(stream.cuda_stream),
                    )
                    if rc != 0:
                        raise RuntimeError(
                            f"cudaMemcpyAsync failed staging host KV read: rc={rc}"
                        )
            finally:
                slot.event.record(stream)

    def advance(self) -> tuple[set[str], set[str]]:
        """Drive the pipeline. Returns (fully staged req_ids, failed req_ids)."""
        done: set[str] = set()
        failed: set[str] = set()
        for req_id, state in list(self._reqs.items()):
            if state.failed or state.aborted:
                state.queued.clear()
            still_reading = []
            for handle, slot, local_ids in state.reading:
                try:
                    xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                except Exception:
                    state.failed = True
                    state.queued.clear()
                    try:
                        self.nixl_wrapper.release_xfer_handle(handle)
                    except Exception:
                        still_reading.append((handle, slot, local_ids))
                    else:
                        slot.pool.free_slots.append(slot)
                    continue
                if xfer_state == "PROC":
                    still_reading.append((handle, slot, local_ids))
                    continue
                try:
                    self.nixl_wrapper.release_xfer_handle(handle)
                except Exception:
                    state.failed = True
                    state.queued.clear()
                    still_reading.append((handle, slot, local_ids))
                    continue
                if xfer_state != "DONE":
                    state.failed = True
                    state.queued.clear()
                    slot.pool.free_slots.append(slot)
                    continue
                if state.failed or state.aborted:
                    slot.pool.free_slots.append(slot)
                    continue
                try:
                    self._start_copy(slot, local_ids)
                except Exception:
                    logger.exception("host KV staging copy failed for %s", req_id)
                    state.failed = True
                    state.queued.clear()
                    state.copying.append(slot)
                    continue
                state.copying.append(slot)
            state.reading = still_reading

            still_copying = []
            for slot in state.copying:
                if slot.event.query():
                    slot.pool.free_slots.append(slot)
                else:
                    still_copying.append(slot)
            state.copying = still_copying

            if not state.failed and not state.aborted:
                try:
                    self._pump(state)
                except Exception:
                    logger.exception("host KV staging read failed for %s", req_id)
                    state.failed = True
                    state.queued.clear()

            if not state.reading and not state.copying and not state.queued:
                self._reqs.pop(req_id, None)
                if not state.aborted:
                    (failed if state.failed else done).add(req_id)
        return done, failed

    def abort(self, req_id: str) -> None:
        state = self._reqs.get(req_id)
        if state is None:
            return
        state.aborted = True
        state.queued.clear()

    def _begin_shutdown(self) -> None:
        """Stop issuing reads and mark all staged requests aborted."""
        if self._closed:
            return
        for req_id in tuple(self._reqs):
            self.abort(req_id)

    def _poll_shutdown(self, cancel: bool = False) -> bool:
        """Advance shutdown without blocking and return whether it completed."""
        if self._closed:
            return True
        if cancel:
            for state in self._reqs.values():
                still_reading = []
                for read in state.reading:
                    handle, slot, _ = read
                    try:
                        self.nixl_wrapper.release_xfer_handle(handle)
                    except Exception:
                        still_reading.append(read)
                    else:
                        slot.pool.free_slots.append(slot)
                state.reading = still_reading
        self.advance()
        if self._reqs:
            return False
        self._copy_stream.synchronize()
        for pool in self._pools.values():
            self.nixl_wrapper.release_dlist_handle(pool.handle)
            self.nixl_wrapper.deregister_memory(pool.reg_descs)
        self._closed = True
        return True

    def _reap_shutdown(self, on_complete: Callable[[], None] | None) -> None:
        try:
            while not self._poll_shutdown(cancel=True):
                time.sleep(_SHUTDOWN_POLL_INTERVAL_S)
            if on_complete is not None:
                on_complete()
            logger.info("Deferred NIXL host-staging cleanup completed.")
        finally:
            self._shutdown_thread = None

    def shutdown(
        self,
        drain_timeout: float | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> bool:
        """Bound shutdown and defer cleanup while transfers remain active."""
        if self._shutdown_thread is not None:
            return False
        if self._closed:
            return True
        if drain_timeout is None:
            drain_timeout = _SHUTDOWN_TIMEOUT_S

        self._begin_shutdown()
        deadline = time.monotonic() + drain_timeout
        drained = self._poll_shutdown()
        while not drained and time.monotonic() < deadline:
            time.sleep(_SHUTDOWN_POLL_INTERVAL_S)
            drained = self._poll_shutdown()
        if not drained:
            drained = self._poll_shutdown(cancel=True)
        if drained:
            return True

        logger.warning(
            "NIXL host-staging shutdown timed out after %.1fs; retaining "
            "registered memory until active operations become terminal.",
            drain_timeout,
        )
        self._shutdown_thread = threading.Thread(
            target=self._reap_shutdown,
            args=(on_complete,),
            name="vllm-nixl-host-staging-shutdown",
            daemon=True,
        )
        self._shutdown_thread.start()
        return False
