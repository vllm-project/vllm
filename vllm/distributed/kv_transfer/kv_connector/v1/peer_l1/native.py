# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only native owner, placement replica, and callback cleanup queue."""

import ctypes
import fcntl
import hashlib
import os
import shutil
import subprocess
from collections.abc import Callable
from enum import IntEnum
from functools import lru_cache
from pathlib import Path

import numpy as np

from vllm import envs


class Operation(IntEnum):
    RELEASE = 3
    CANCEL = 4
    CLEANUPS = 6
    ACQUIRE_EXACT = 7
    HELLO = 8
    ACQUIRE_LEASED = 9
    SUBSCRIBE = 10
    ACQUIRE_DEFERRED = 11
    ACQUIRE_DEFERRED_LEASED = 12


@lru_cache(maxsize=1)
def library():
    source = Path(__file__).with_suffix("")
    digest = hashlib.sha256(
        b"".join(
            path.read_bytes()
            for path in sorted(source.iterdir())
            if path.suffix in (".h", ".cpp")
        )
    ).hexdigest()[:20]
    build = Path(envs.VLLM_CACHE_ROOT) / "peer_l1" / digest
    build.mkdir(parents=True, exist_ok=True)
    target = build / "owner.so"
    with (build / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            compiler = os.environ.get("CXX") or shutil.which("c++")
            if not compiler:
                raise RuntimeError("PeerL1Connector requires a C++17 compiler (CXX)")
            temporary = target.with_suffix(f".{os.getpid()}.so")
            try:
                result = subprocess.run(
                    [
                        compiler,
                        "-std=c++17",
                        "-O3",
                        "-shared",
                        "-fPIC",
                        "-pthread",
                        str(source / "owner.cpp"),
                        "-o",
                        str(temporary),
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode:
                    raise RuntimeError(f"Peer L1 native build failed: {result.stderr}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    lib = ctypes.CDLL(str(target))
    ptr, u64, size = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t
    string, integer = ctypes.c_char_p, ctypes.c_int
    signatures = {
        "owner_start": [string, u64, u64, u64, u64, integer, u64, u64, ptr, ptr],
        "owner_stop": [ptr],
        "owner_publish": [ptr, ptr, ptr, size],
        "owner_retire": [ptr, ctypes.c_int64, integer],
        "owner_absorb": [ptr, ptr, size, ptr, u64, ptr, size, u64],
        "owner_deferred_rows": [ptr, ptr, size],
        "owner_complete_absorb": [ptr, u64, ptr],
        **{
            f"owner_{name}": [ptr]
            for name in (
                "reserve",
                "pending",
                "refs",
                "wake_fd",
                "freeze",
            )
        },
        "owner_thaw": [ptr, integer],
        "owner_rpc": [string, u64, u64, u64, u64, ptr, ptr, size, integer],
        "replica_start": [ptr, ptr, size],
        "replica_stop": [ptr],
        "replica_generation": [ptr, size],
        "replica_window": [ptr, size],
        "replica_query": [ptr, ptr, size, ptr],
        "cleanup_start": [integer],
        "cleanup_stop": [ptr],
        "cleanup_pending": [ptr],
        "cleanup_create": [ptr, string, u64, u64, u64, u64, u64, ptr],
    }
    lib.l1_owner_error.restype = ctypes.c_char_p

    def check(result, function, args):
        if result < 0:
            raise RuntimeError(lib.l1_owner_error().decode())
        return result

    for name, args in signatures.items():
        function = getattr(lib, f"l1_{name}")
        function.argtypes = args
        function.restype = ctypes.c_int64
        if name != "owner_rpc":
            function.errcheck = check
    lib.l1_cleanup_enqueue.argtypes = [ptr]
    lib.l1_cleanup_enqueue.restype = None
    return lib


def key_array(keys: list[bytes]) -> np.ndarray:
    return np.frombuffer(b"".join(keys), dtype=np.uint64).reshape(-1, 4)


class NativeOwner:
    STATS = (
        "owner_grants",
        "owner_rows_pinned",
        "owner_busy_window",
        "owner_busy_cap",
        "owner_absorbs",
        "owner_withheld_high_water",
        "owner_withheld_blocks",
        "owner_reserve_before_absorb",
        "owner_reserve",
        "owner_effective_window",
        "owner_deferred_grants",
        "owner_deferred_rows",
        "owner_deferred_reject_frozen",
        "owner_deferred_reject_lease_expiry",
        "owner_deferred_reject_missing_key",
        "owner_deferred_reject_cap",
        "owner_deferred_reject_budget",
    )

    def __init__(
        self,
        endpoint: str,
        epoch: int,
        rows: int,
        *,
        pin_cap: int = 4096,
        window: int = 512,
        lease_ms: int = 1000,
        log_capacity: int = 4096,
        initial_window: int | None = None,
        clock: Callable[[], float] | None = None,
    ):
        if min(rows, pin_cap, window, lease_ms, log_capacity) <= 0:
            raise ValueError("Peer L1 owner limits must be positive")
        if initial_window is not None and not 0 <= initial_window <= window:
            raise ValueError("Initial window must be between zero and pin_window")
        self.endpoint, self.epoch = endpoint, epoch
        self._lib = library()
        self._published = bytearray(rows + 1)
        self._published_buffer = (ctypes.c_uint8 * (rows + 1)).from_buffer(
            self._published
        )
        self.published_rows = memoryview(self._published)[:rows].toreadonly()
        self._clock = (
            ctypes.CFUNCTYPE(ctypes.c_int64)(lambda: int(clock() * 1_000_000_000))
            if clock is not None
            else None
        )
        self._handle = self._lib.l1_owner_start(
            endpoint.encode(),
            epoch,
            rows,
            pin_cap,
            window,
            lease_ms,
            log_capacity,
            window if initial_window is None else initial_window,
            ctypes.addressof(self._published_buffer),
            self._clock,
        )
        self._absorbed = np.empty(rows * 2, dtype=np.int64)
        self._stats = np.zeros(len(self.STATS), dtype=np.uint64)
        self._deferred_snapshot = 0

    @property
    def wake_fd(self) -> int:
        return self._lib.l1_owner_wake_fd(self._handle)

    @property
    def refs(self) -> int:
        return self._lib.l1_owner_refs(self._handle)

    def publish(self, keys: list[bytes], rows: list[int]) -> None:
        if len(keys) != len(rows):
            raise ValueError("Each peer key needs one row")
        identities = key_array(keys)
        row_ids = np.asarray(rows, dtype=np.int64)
        self._lib.l1_owner_publish(
            self._handle, identities.ctypes.data, row_ids.ctypes.data, len(rows)
        )

    def retire(self, row: int, *, withhold: bool) -> bool:
        return bool(self._lib.l1_owner_retire(self._handle, row, withhold))

    def retire_for_pool(self, row: int, *, withhold: bool) -> int:
        """Retire; 2 transfers an unabsorbed pin into an immediate pool ref."""
        return self._lib.l1_owner_retire(self._handle, row, 1 if withhold else 2)

    def absorb(
        self,
        free_blocks: int | None = None,
        *,
        in_use_rows: list[int] | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        if free_blocks is not None and free_blocks < 0:
            raise ValueError("Free block count must be nonnegative")
        if in_use_rows is None:
            self.deferred_rows()
        in_use = np.asarray(in_use_rows or (), dtype=np.int64)
        n = self._lib.l1_owner_absorb(
            self._handle,
            self._absorbed.ctypes.data,
            self._absorbed.size,
            self._stats.ctypes.data,
            (1 << 64) - 1 if free_blocks is None else free_blocks,
            in_use.ctypes.data,
            len(in_use),
            self._deferred_snapshot,
        )
        self._deferred_snapshot = 0
        result: tuple[list[int], list[int], list[int]] = ([], [], [])
        for action, row in self._absorbed[:n].reshape(-1, 2):
            result[action - 1].append(int(row))
        return result

    def deferred_rows(self) -> list[int]:
        """Snapshot queued candidates; the scheduler must not allocate before absorb."""
        if not self._published[-1]:
            self._deferred_snapshot = 0
            return []
        n = self._lib.l1_owner_deferred_rows(
            self._handle, self._absorbed.ctypes.data, self._absorbed.size
        )
        self._deferred_snapshot = int(self._absorbed[0])
        return self._absorbed[1:n].tolist()

    def absorb_stats(self) -> dict[str, int]:
        """Return the counter deltas and gauges from the last atomic absorb."""
        return dict(zip(self.STATS, map(int, self._stats), strict=True))

    def complete_absorb(self, free_blocks: int) -> None:
        """Raise the conservative budget after applying refcounts, before allocation."""
        if free_blocks < 0:
            raise ValueError("Free block count must be nonnegative")
        self._lib.l1_owner_complete_absorb(
            self._handle, free_blocks, self._stats.ctypes.data
        )

    def admission_reserve(self) -> int:
        return self._lib.l1_owner_reserve(self._handle)

    def pending(self) -> bool:
        return bool(self._lib.l1_owner_pending(self._handle))

    def freeze(self) -> bool:
        return bool(self._lib.l1_owner_freeze(self._handle))

    def thaw(self, *, clear: bool = False) -> None:
        self._lib.l1_owner_thaw(self._handle, clear)

    def close(self) -> None:
        if self._handle:
            self._lib.l1_owner_stop(self._handle)
            self._handle = 0


class PlacementReplica:
    def __init__(self, peers: list[tuple[str, int]]):
        self._lib = library()
        self._peers = peers
        endpoints = (ctypes.c_char_p * len(peers))(
            *(endpoint.encode() for endpoint, _ in peers)
        )
        epochs = np.asarray([epoch for _, epoch in peers], dtype=np.uint64)
        self._handle = self._lib.l1_replica_start(
            endpoints, epochs.ctypes.data, len(peers)
        )

    def generation(self, peer: int) -> int:
        return self._lib.l1_replica_generation(self._handle, peer)

    def window(self, peer: int) -> int:
        return self._lib.l1_replica_window(self._handle, peer)

    def query(self, keys: list[bytes]) -> np.ndarray:
        identities = key_array(keys)
        short = np.ascontiguousarray(identities[:, 0])
        result = np.empty((len(keys), len(self._peers)), dtype=np.uint8)
        self._lib.l1_replica_query(
            self._handle, short.ctypes.data, len(keys), result.ctypes.data
        )
        return result

    def close(self) -> None:
        if self._handle:
            self._lib.l1_replica_stop(self._handle)
            self._handle = 0


class CleanupQueue:
    def __init__(self, timeout_ms: int = 50):
        self._lib = library()
        self._handle = self._lib.l1_cleanup_start(timeout_ms)

    @property
    def callback_address(self) -> int:
        address = ctypes.cast(self._lib.l1_cleanup_enqueue, ctypes.c_void_p).value
        assert address is not None
        return address

    @property
    def pending(self) -> int:
        return self._lib.l1_cleanup_pending(self._handle)

    def create(
        self, endpoint: str, epoch: int, reader: int, seq: int, *, next_node: int = 0
    ) -> int:
        return self._lib.l1_cleanup_create(
            self._handle, endpoint.encode(), 3, epoch, reader, seq, 0, next_node
        )

    def enqueue(self, node: int) -> None:
        self._lib.l1_cleanup_enqueue(node)

    def close(self) -> None:
        if self._handle:
            self._lib.l1_cleanup_stop(self._handle)
            self._handle = 0
