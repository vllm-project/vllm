# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-mode KV lease slots for the NIXL connector.

A producer slot holds the lease nonce until the request's blocks are released,
so a reader that still sees the nonce after its READ knows the blocks were not
reused during it.
"""

import secrets
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.distributed.device_communicators.shm_broadcast import memory_fence
from vllm.distributed.kv_transfer.kv_connector.utils import EngineId
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqId
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

LEASE_SLOT_BYTES = 8
# Below 2**53 so routers that parse JSON numbers as doubles keep nonces exact.
_NONCE_LIMIT = 1 << 53
_NUM_LANDING_CELLS = 4096


def get_num_lease_slots(
    vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"
) -> int:
    assert vllm_config.kv_transfer_config is not None
    num_slots = vllm_config.kv_transfer_config.get_from_extra_config(
        "kv_lease_slots", None
    )
    if num_slots is None:
        # Each scheduled or held prefill owns a block; preempted ones keep
        # their lease without one.
        num_slots = (
            kv_cache_config.num_blocks + vllm_config.scheduler_config.max_num_seqs
        )
    if int(num_slots) < 1:
        raise ValueError(f"kv_lease_slots must be positive, got {num_slots}")
    return int(num_slots)


def lease_verification_enabled(vllm_config: "VllmConfig") -> bool:
    assert vllm_config.kv_transfer_config is not None
    enabled = vllm_config.kv_transfer_config.get_from_extra_config(
        "kv_lease_verification", True
    )
    if not isinstance(enabled, bool):
        raise ValueError(f"kv_lease_verification must be a bool, got {enabled!r}")
    return enabled


class LeaseSlotAllocator:
    """Scheduler-side lease slots of a producer engine.

    A slot is freed once workers zeroed it (``finished_sending``) or with a
    clear that workers apply before the arms of the same step.
    """

    def __init__(self, num_slots: int):
        self._num_slots = num_slots
        self._free = list(range(num_slots - 1, -1, -1))
        self._leases: dict[ReqId, tuple[int, int]] = {}
        # Not `random`: engines seed it, which would repeat nonces on restart.
        self._next_nonce = secrets.randbelow(_NONCE_LIMIT // 2) + 1
        self._to_arm: dict[ReqId, tuple[int, int]] = {}
        self._to_clear: set[ReqId] = set()

    def get(self, req_id: ReqId) -> tuple[int, int] | None:
        return self._leases.get(req_id)

    def acquire(self, req_id: ReqId) -> tuple[int, int] | None:
        """Return the (slot, nonce) lease of ``req_id``, allocating it once."""
        if (lease := self._leases.get(req_id)) is not None:
            return lease
        if not self._free:
            logger.error(
                "Out of NIXL lease slots (%d): the KV of request %s gets no "
                "lease, so readers decline it. Raise kv_lease_slots.",
                self._num_slots,
                req_id,
            )
            return None
        nonce = self._next_nonce
        self._next_nonce = nonce + 1 if nonce + 1 < _NONCE_LIMIT else 1
        lease = (self._free.pop(), nonce)
        self._leases[req_id] = lease
        self._to_arm[req_id] = lease
        return lease

    def release(self, req_id: ReqId, *, clear: bool) -> None:
        """Free the lease of ``req_id``.

        Args:
            req_id: The request whose lease ends.
            clear: Whether workers must zero the slot themselves, having no
                send record of the request that would do it.

        """
        lease = self._leases.pop(req_id, None)
        if lease is None:
            return
        # A lease not sent to workers yet was never written.
        if clear and self._to_arm.pop(req_id, None) is None:
            self._to_clear.add(req_id)
        self._free.append(lease[0])

    def take_updates(self) -> tuple[dict[ReqId, tuple[int, int]], set[ReqId]]:
        updates = self._to_arm, self._to_clear
        self._to_arm, self._to_clear = {}, set()
        return updates


def lease_slot_descs(base_addr: int, num_slots: int) -> np.ndarray:
    """NIXL ``(addr, len, device_id)`` descriptors, one per DRAM slot."""
    descs = np.zeros((num_slots, 3), dtype=np.uint64)
    descs[:, 0] = np.uint64(base_addr) + np.uint64(LEASE_SLOT_BYTES) * np.arange(
        num_slots, dtype=np.uint64
    )
    descs[:, 1] = LEASE_SLOT_BYTES
    return descs


def _alloc_slots(num_slots: int) -> torch.Tensor:
    # Only the CPU and the NIC touch the slots; NIXL pins them on registration.
    return torch.zeros(num_slots, dtype=torch.int64)


class ProducerLeaseSlots:
    """Lease slots that a producer worker exports to readers."""

    def __init__(self, num_slots: int):
        self.tensor = _alloc_slots(num_slots)
        self._slots = self.tensor.numpy()
        self._leases: dict[ReqId, tuple[int, int]] = {}
        self._dirty = False

    def nonce(self, req_id: ReqId) -> int | None:
        lease = self._leases.get(req_id)
        return None if lease is None else lease[1]

    def update(
        self, to_clear: Iterable[ReqId], to_arm: dict[ReqId, tuple[int, int]]
    ) -> None:
        # Clears first: a cleared slot may be armed again in the same step.
        for req_id in to_clear:
            self.clear(req_id)
        for req_id, (slot, nonce) in to_arm.items():
            self._slots[slot] = nonce
            self._leases[req_id] = (slot, nonce)
            self._dirty = True

    def clear(self, req_id: ReqId) -> None:
        """Zero the slot of ``req_id``; must precede releasing its blocks."""
        if (lease := self._leases.pop(req_id, None)) is not None:
            self._slots[lease[0]] = 0
            self._dirty = True

    def publish(self) -> None:
        """Order slot stores before the step results that release blocks."""
        if self._dirty:
            memory_fence()
            self._dirty = False


@dataclass
class _LeaseCheck:
    engine_id: EngineId
    remote_rank: int
    slot: int
    nonce: int


@dataclass
class _SlotRead:
    req_id: ReqId
    slot: int
    cell: int
    nonce: int


@dataclass
class _LeaseRead:
    handle: int
    remote: tuple[EngineId, int]
    entries: list[_SlotRead]


class LeaseVerifier:
    """Reader-side check that remote lease slots still hold their nonces.

    Runs after all block READs of a request completed, with one READ per
    remote rank for all requests checked in a step.
    """

    def __init__(self, nixl_wrapper: Any, num_cells: int = _NUM_LANDING_CELLS):
        self.tensor = _alloc_slots(num_cells)
        self._cells = self.tensor.numpy()
        self._free_cells = list(range(num_cells - 1, -1, -1))
        self._nixl = nixl_wrapper
        self.local_handle: int | None = None
        self._remote_handles: dict[EngineId, dict[int, int]] = defaultdict(dict)
        # Requests whose block READs are still running.
        self._checks: dict[ReqId, list[_LeaseCheck]] = defaultdict(list)
        # Requests waiting for free landing cells.
        self._queued: dict[ReqId, list[_LeaseCheck]] = {}
        self._reads: list[_LeaseRead] = []
        self._outstanding: dict[ReqId, int] = {}
        self._reasons: dict[ReqId, str] = {}
        self._verified: set[ReqId] = set()
        self._failures: dict[ReqId, str] = {}

    def add_remote(self, engine_id: EngineId, remote_rank: int, handle: int) -> None:
        self._remote_handles[engine_id][remote_rank] = handle

    def can_verify(self, engine_id: EngineId, remote_ranks: Iterable[int]) -> bool:
        handles = self._remote_handles.get(engine_id, {})
        return self.local_handle is not None and all(
            rank in handles for rank in remote_ranks
        )

    def add_check(
        self,
        req_id: ReqId,
        engine_id: EngineId,
        remote_rank: int,
        slot: int,
        nonce: int,
    ) -> None:
        self._checks[req_id].append(_LeaseCheck(engine_id, remote_rank, slot, nonce))

    def has_checks(self, req_id: ReqId) -> bool:
        return req_id in self._checks

    def discard(self, req_id: ReqId) -> None:
        self._checks.pop(req_id, None)

    def busy_engines(self) -> set[EngineId]:
        return {read.remote[0] for read in self._reads}

    def step(self, reads_done: Iterable[ReqId]) -> tuple[set[ReqId], dict[ReqId, str]]:
        """Start checks of requests whose block READs completed.

        Returns:
            Requests whose leases all held, and failed requests with reasons.

        """
        for req_id in reads_done:
            self._queued[req_id] = self._checks.pop(req_id)
        self._post_queued()
        self._poll()
        verified, failures = self._verified, self._failures
        self._verified, self._failures = set(), {}
        return verified, failures

    def _post_queued(self) -> None:
        batches: dict[tuple[EngineId, int], list[_SlotRead]] = defaultdict(list)
        for req_id, checks in list(self._queued.items()):
            if len(checks) > len(self._free_cells):
                break
            del self._queued[req_id]
            for check in checks:
                cell = self._free_cells.pop()
                self._cells[cell] = 0
                batches[(check.engine_id, check.remote_rank)].append(
                    _SlotRead(req_id, check.slot, cell, check.nonce)
                )
            self._outstanding[req_id] = len(checks)
        for remote, entries in batches.items():
            self._post_read(remote, entries)

    def _post_read(
        self, remote: tuple[EngineId, int], entries: list[_SlotRead]
    ) -> None:
        engine_id, remote_rank = remote
        remote_handle = self._remote_handles.get(engine_id, {}).get(remote_rank)
        handle = None
        try:
            if remote_handle is None:
                raise RuntimeError("remote engine state was released")
            assert self.local_handle is not None
            handle = self._nixl.make_prepped_xfer(
                "READ",
                self.local_handle,
                [entry.cell for entry in entries],
                remote_handle,
                [entry.slot for entry in entries],
            )
            if self._nixl.transfer(handle) == "ERR":
                raise RuntimeError("transfer was not posted")
        except Exception as e:
            logger.warning(
                "Failed to post NIXL lease slot READ to engine %s rank %s: %s",
                engine_id,
                remote_rank,
                e,
            )
            # The backend may still write the cells of an unreleased handle.
            reusable = handle is None or self._release(handle)
            self._resolve(remote, entries, done=False, reusable=reusable)
            return
        self._reads.append(_LeaseRead(handle, remote, entries))

    def _poll(self) -> None:
        for read in list(self._reads):
            try:
                state = self._nixl.check_xfer_state(read.handle)
            except Exception:
                state = "ERR"
            if state == "PROC":
                continue
            self._reads.remove(read)
            reusable = self._release(read.handle)
            self._resolve(read.remote, read.entries, state == "DONE", reusable)

    def _resolve(
        self,
        remote: tuple[EngineId, int],
        entries: list[_SlotRead],
        done: bool,
        reusable: bool,
    ) -> None:
        for entry in entries:
            if reusable:
                self._free_cells.append(entry.cell)
            req_id = entry.req_id
            if not done:
                self._reasons.setdefault(
                    req_id,
                    f"the lease slot of engine {remote[0]} rank {remote[1]} "
                    "could not be read",
                )
            elif self._cells[entry.cell] != entry.nonce:
                self._reasons.setdefault(
                    req_id,
                    f"engine {remote[0]} rank {remote[1]} released the KV "
                    "blocks before the read completed",
                )
            self._outstanding[req_id] -= 1
            if self._outstanding[req_id] == 0:
                del self._outstanding[req_id]
                if (reason := self._reasons.pop(req_id, None)) is None:
                    self._verified.add(req_id)
                else:
                    self._failures[req_id] = reason

    def _release(self, handle: int) -> bool:
        try:
            self._nixl.release_xfer_handle(handle)
        except Exception:
            logger.warning("Failed to release NIXL lease slot READ.", exc_info=True)
            return False
        return True

    def release_engine(self, engine_id: EngineId) -> None:
        """Release the descriptors of a remote engine without in-flight READs."""
        for handle in self._remote_handles.pop(engine_id, {}).values():
            self._nixl.release_dlist_handle(handle)

    def shutdown(self) -> None:
        for read in self._reads:
            self._release(read.handle)
        self._reads.clear()
        if self.local_handle is not None:
            self._nixl.release_dlist_handle(self.local_handle)
            self.local_handle = None
