# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native owner and placement bindings for schedulers; cleanup for both processes."""

import ctypes
import hashlib
import shlex
import shutil
import struct
from collections.abc import Callable
from enum import IntEnum
from functools import lru_cache
from pathlib import Path

import numpy as np

from vllm import envs


class Operation(IntEnum):
    RELEASE = 3
    CANCEL = 4
    ACQUIRE_EXACT = 7
    HELLO = 8
    ACQUIRE_LEASED = 9
    ACQUIRE_DEFERRED = 11
    ACQUIRE_DEFERRED_LEASED = 12
    MEMBERS = 13


@lru_cache(maxsize=1)
def library():
    """Build/load native control code; may wait on the build lock or compiler."""
    from torch.utils.cpp_extension import get_cxx_compiler, is_ninja_available, load

    compiler = shlex.split(get_cxx_compiler())
    if not compiler or shutil.which(compiler[0]) is None:
        raise RuntimeError("Install a C++17 compiler or set CXX for PeerL1Connector")
    if not is_ninja_available():
        raise RuntimeError("Install ninja to build the PeerL1Connector native library")
    source_dir = Path(__file__).with_suffix("")
    digest = hashlib.sha256()
    for path in sorted(source_dir.rglob("*")):
        if path.suffix in (".h", ".cpp"):
            digest.update(str(path.relative_to(source_dir)).encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    name = f"vllm_peer_l1_{digest.hexdigest()[:12]}"
    build_directory = Path(envs.VLLM_CACHE_ROOT) / "peer_l1" / name
    build_directory.mkdir(parents=True, exist_ok=True)
    load(
        name=name,
        sources=[str(source_dir / "owner.cpp")],
        build_directory=str(build_directory),
        extra_cflags=["-std=c++17", "-O3", "-pthread"],
        extra_ldflags=["-pthread"],
        with_cuda=False,
        is_python_module=False,
    )
    lib = ctypes.CDLL(str(build_directory / f"{name}.so"))
    ptr, u64, size = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t
    string, integer = ctypes.c_char_p, ctypes.c_int
    signatures = {
        "owner_start": [
            string,
            u64,
            u64,
            u64,
            u64,
            integer,
            u64,
            u64,
            ptr,
            ptr,
            ptr,
            u64,
        ],
        "owner_gate": [ptr, integer],
        "owner_stop": [ptr],
        "owner_describe": [ptr, ptr, size, ptr, size],
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
        "owner_rpc": [string, u64, u64, u64, u64, ptr, ptr, size, integer, ptr],
        "replica_start": [ptr, ptr, ptr, size, ptr],
        "replica_stop": [ptr],
        "replica_generation": [ptr, size],
        "replica_window": [ptr, size],
        "replica_capacity": [ptr, size],
        "replica_query": [ptr, ptr, size, ptr],
        "replica_query_any": [ptr, ptr, size, ptr],
        "cleanup_start": [integer],
        "cleanup_stop": [ptr],
        "cleanup_pending": [ptr],
        "cleanup_create": [ptr, string, u64, u64, u64, u64, u64, ptr, ptr],
    }
    lib.l1_owner_stat_schema.argtypes = []
    lib.l1_owner_stat_schema.restype = ctypes.c_char_p
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


def credential_array(credential: bytes) -> np.ndarray:
    if credential and len(credential) != 32:
        raise ValueError("Peer credential must contain 32 bytes")
    return np.frombuffer(credential or bytes(32), dtype=np.uint64)


def description(
    endpoint: str,
    epoch: int,
    *,
    members: bool = False,
    credential: bytes = b"",
    timeout_ms: int = 50,
) -> bytes:
    """Fetch HELLO or MEMBERS synchronously, bounded by `timeout_ms`."""
    out = np.empty(65536, dtype=np.int64)
    auth = credential_array(credential)
    n = library().l1_owner_rpc(
        endpoint.encode(),
        Operation.MEMBERS if members else Operation.HELLO,
        epoch,
        0,
        0,
        0,
        out.ctypes.data,
        len(out),
        timeout_ms,
        auth.ctypes.data,
    )
    if n <= 0 or not 0 <= out[0] <= (n - 1) * 8:
        raise ConnectionError("Peer description unavailable or unauthorized")
    return out[1:n].tobytes()[: int(out[0])]


def _blob_words(data: bytes) -> np.ndarray:
    if len(data) > 524280:
        raise ValueError("Peer description exceeds 512 KiB")
    return np.frombuffer(
        struct.pack("<Q", len(data)) + data + bytes(-len(data) % 8), dtype=np.int64
    )


class NativeOwner:
    """Bind scheduler-owned cache rows to a native RPC and subscription service.

    Calls run on the scheduler thread under the native owner's mutex. RPCs
    can grant pins between calls; pool references are reconciled at absorb.
    """

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
        protected_cap: int | None = None,
        credential: bytes = b"",
        clock: Callable[[], float] | None = None,
    ):
        if min(rows, pin_cap, window, lease_ms, log_capacity) <= 0:
            raise ValueError("Peer L1 owner limits must be positive")
        if initial_window is not None and not 0 <= initial_window <= window:
            raise ValueError("Initial window must be between zero and pin_window")
        if protected_cap is not None and protected_cap < 0:
            raise ValueError("Protected capacity must be nonnegative")
        self.endpoint, self.epoch = endpoint, epoch
        self._lib = library()
        self._published = bytearray(rows + 1)
        self._published_buffer = (ctypes.c_uint8 * (rows + 1)).from_buffer(
            self._published
        )
        self.published_rows = memoryview(self._published)[:rows].toreadonly()
        auth = credential_array(credential)
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
            auth.ctypes.data,
            self._clock,
            (1 << 64) - 1 if protected_cap is None else protected_cap,
        )
        self._absorbed = np.empty(rows * 2, dtype=np.int64)
        self.stat_schema = [
            line.split(":")
            for line in self._lib.l1_owner_stat_schema().decode().splitlines()
        ]
        self._stats = np.zeros(len(self.stat_schema), dtype=np.uint64)
        self._deferred_snapshot = 0

    @property
    def wake_fd(self) -> int:
        return self._lib.l1_owner_wake_fd(self._handle)

    def describe(self, hello: bytes, members: bytes) -> None:
        h, m = _blob_words(hello), _blob_words(members)
        self._lib.l1_owner_describe(
            self._handle, h.ctypes.data, len(h), m.ctypes.data, len(m)
        )

    @property
    def refs(self) -> int:
        return self._lib.l1_owner_refs(self._handle)

    def publish(self, keys: list[bytes], rows: list[int]) -> None:
        """Expose full-block aliases after the caller proves GPU writes finished."""
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
        """Remove aliases before pool reuse and report remaining external protection.

        Mode 1 withholds a free row; mode 2 adopts an in-use row's pin. Return
        2 requires the scheduler to take a pool reference before local free.
        """
        return self._lib.l1_owner_retire(self._handle, row, 1 if withhold else 2)

    def absorb(
        self,
        free_blocks: int | None = None,
        *,
        in_use_rows: list[int] | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """Return rows to touch, free and return from withholding, in that order.

        The scheduler must apply all actions without allocation interleaving.
        `in_use_rows` must describe the preceding deferred-row snapshot.
        """
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
        """Snapshot queued candidates before reading their pool refcounts.

        The scheduler must not allocate until `absorb` and its returned pool
        actions are applied; newer queued entries wait for the next snapshot.
        """
        if not self._handle:
            raise RuntimeError("peer L1 handle is closed")
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
        return dict(
            zip(
                (entry[0] for entry in self.stat_schema),
                map(int, self._stats),
                strict=True,
            )
        )

    def complete_absorb(self, free_blocks: int) -> None:
        """Refresh capacity after pool actions, before the scheduler allocates again.

        `free_blocks` is the physical free count after applying those actions.
        """
        if free_blocks < 0:
            raise ValueError("Free block count must be nonnegative")
        self._lib.l1_owner_complete_absorb(
            self._handle, free_blocks, self._stats.ctypes.data
        )

    def set_gate(self, open: bool) -> None:
        self._lib.l1_owner_gate(self._handle, open)

    def admission_reserve(self) -> int:
        return self._lib.l1_owner_reserve(self._handle)

    def pending(self) -> bool:
        return bool(self._lib.l1_owner_pending(self._handle))

    def freeze(self) -> bool:
        """Refuse new grants; return whether native ownership permits a reset."""
        return bool(self._lib.l1_owner_freeze(self._handle))

    def thaw(self, *, clear: bool = False) -> None:
        """Resume grants; clear aliases only after a successful caller-side reset."""
        self._lib.l1_owner_thaw(self._handle, clear)

    def close(self) -> None:
        """Join native services before releasing the Python-owned publication buffer."""
        if self._handle:
            self._lib.l1_owner_stop(self._handle)
            self._handle = 0


class PlacementReplica:
    """Read advisory peer placement in the scheduler process.

    A native subscription thread updates hints under snapshot and shard locks.
    Hints may be stale or collide; only an owner grant authorizes a read.
    """

    def __init__(self, peers: list[tuple[str, int, int]], *, credential: bytes = b""):
        self._lib = library()
        self._peers = peers
        endpoints = (ctypes.c_char_p * len(peers))(
            *(endpoint.encode() for endpoint, _, _ in peers)
        )
        epochs = np.asarray([epoch for _, epoch, _ in peers], dtype=np.uint64)
        key_caps = np.asarray([cap for _, _, cap in peers], dtype=np.uint64)
        auth = credential_array(credential)
        self._handle = self._lib.l1_replica_start(
            endpoints,
            epochs.ctypes.data,
            key_caps.ctypes.data,
            len(peers),
            auth.ctypes.data,
        )

    def generation(self, peer: int) -> int:
        return self._lib.l1_replica_generation(self._handle, peer)

    def window(self, peer: int) -> int:
        return self._lib.l1_replica_window(self._handle, peer)

    def capacity(self, peer: int) -> int:
        """Return advertised remaining protected rows, or zero when gated."""
        return self._lib.l1_replica_capacity(self._handle, peer)

    def query(self, keys: list[bytes]) -> np.ndarray:
        """Return a `[key][peer]` presence matrix, taking native shared locks.

        This scheduler-thread call may wait for a placement update to finish.
        """
        identities = key_array(keys)
        short = np.ascontiguousarray(identities[:, 0])
        result = np.empty((len(keys), len(self._peers)), dtype=np.uint8)
        self._lib.l1_replica_query(
            self._handle, short.ctypes.data, len(keys), result.ctypes.data
        )
        return result

    def try_query_any(self, keys: list[bytes]) -> np.ndarray | None:
        """Return per-key presence on any peer without waiting for update locks.

        `None` means a snapshot is missing or a lock is contended, not absence.
        """
        identities = key_array(keys)
        short = np.ascontiguousarray(identities[:, 0])
        result = np.empty(len(keys), dtype=np.uint8)
        acquired = self._lib.l1_replica_query_any(
            self._handle, short.ctypes.data, len(keys), result.ctypes.data
        )
        return result if acquired else None

    def close(self) -> None:
        if self._handle:
            self._lib.l1_replica_stop(self._handle)
            self._handle = 0


class CleanupQueue:
    """Send grant releases on a native thread in the worker or scheduler process.

    GPU callbacks enqueue preallocated nodes without Python or socket I/O.
    Keep the queue alive until all callbacks using it have completed.
    """

    def __init__(self, timeout_ms: int = 50, *, credential: bytes = b""):
        self.credential = credential
        self._credential = credential_array(credential)
        self._lib = library()
        self._handle = self._lib.l1_cleanup_start(timeout_ms)

    @property
    def callback_address(self) -> int:
        """Return the native `void(void*)` callback accepting a created node list."""
        address = ctypes.cast(self._lib.l1_cleanup_enqueue, ctypes.c_void_p).value
        assert address is not None
        return address

    @property
    def pending(self) -> int:
        return self._lib.l1_cleanup_pending(self._handle)

    def create(
        self,
        endpoint: str,
        epoch: int,
        reader: int,
        seq: int,
        *,
        next_node: int = 0,
        lease_ms: int = 0,
    ) -> int:
        """Allocate a release node on the caller thread without sending it.

        `next_node` links a batch for one GPU callback. The caller must enqueue
        the node exactly once, directly or through that callback.
        """
        return self._lib.l1_cleanup_create(
            self._handle,
            endpoint.encode(),
            Operation.RELEASE,
            epoch,
            reader,
            seq,
            lease_ms,
            next_node,
            self._credential.ctypes.data,
        )

    def enqueue(self, node: int) -> None:
        """Transfer a node to the native sender without waiting for a remote reply."""
        if not self._handle:
            raise RuntimeError("peer L1 handle is closed")
        self._lib.l1_cleanup_enqueue(node)

    def close(self) -> None:
        """Join the sender after the caller has stopped all callback producers."""
        if self._handle:
            self._lib.l1_cleanup_stop(self._handle)
            self._handle = 0
