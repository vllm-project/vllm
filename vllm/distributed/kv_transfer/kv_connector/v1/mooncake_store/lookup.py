# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix lookups against the Mooncake store, and the process serving them.

A lookup builds and probes one key per Store chunk and rank namespace, so for
long prompts it holds the GIL for milliseconds, including inside the native
`batch_is_exist`. Rank 0 therefore serves the scheduler's lookups from a
separate process with its own lookup-only store client, so they never delay
the worker's GPU launches.
"""

import functools
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import msgspec
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.coordinator import (  # noqa: E501
    ExternalCachedBlockPool,
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import (  # noqa: E501
    BlobBlockHashes,
    ChunkedTokenDatabase,
    MooncakeLookupResult,
    TailKeyBoundary,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.layout import (  # noqa: E501
    PoolKey,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.metrics import (  # noqa: E501
    MooncakeStoreConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.protocol import (  # noqa: E501
    LOOKUP_MSG,
    RESP_ERR,
    encode_lookup_response,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.selective_lease import (  # noqa: E501
    SelectiveLookupLease,
)
from vllm.distributed.mooncake_store import MooncakeStoreConfig, setup_mooncake_store
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import KVCacheGroupSpec

logger = init_logger(__name__)

# The lookup client only queries the master; it moves no KV data.
_LOOKUP_CLIENT_BUFFER_SIZE = 16 * 1024 * 1024
_LOOKUP_PROCESS_READY = b"ready\n"
_LOOKUP_PROCESS_MAX_RESTART_DELAY_S = 30.0
_LOOKUP_PROCESS_START_TIMEOUT_S = 600.0


class MooncakeStoreLookup:
    """Prefix lookup against the Mooncake store.

    Shared by `MooncakeStoreWorker` and the lookup process, which set the
    attributes below.
    """

    store: Any
    coord: MooncakeStoreCoordinator
    token_dbs: list[ChunkedTokenDatabase]
    hash_block_size: int
    _capacity_only: bool
    _kv_cache_groups: list[KVCacheGroupSpec]
    _lookup_key_prefixes: tuple[tuple[str, ...], ...]
    _selective_lease: SelectiveLookupLease | None
    kv_connector_stats: MooncakeStoreConnectorStats
    _kv_connector_stats_lock: threading.Lock

    def _record_kv_connector_operation(
        self,
        operation: str,
        duration_seconds: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        with self._kv_connector_stats_lock:
            self.kv_connector_stats.record_operation(
                operation=operation,
                duration_seconds=duration_seconds,
                num_keys=num_keys,
                num_bytes=num_bytes,
                status=status,
                num_failed_keys=num_failed_keys,
            )

    def lookup(
        self, num_tokens: int, block_hashes: Sequence[BlockHash]
    ) -> MooncakeLookupResult:
        """Check how many prefix tokens exist in the store.

        Checks across all rank-specific key namespaces that may be loaded. A
        hit covering all `num_tokens` is re-derived below the request end so
        the last token is recomputed for sampling.
        """
        if self._capacity_only:
            return MooncakeLookupResult(0)

        token_len = self.coord.align_lookup_length(num_tokens)
        if not block_hashes or token_len <= 0:
            return MooncakeLookupResult(0)

        # Build per-(group, hash) candidate keys expanded across rank namespaces.
        # candidate_meta stores (group, hash_bytes, end_token) for each key slice.
        candidate_keys: list[str] = []
        candidate_meta: list[tuple[int, bytes, int]] = []
        fine_grained = self.coord.enable_partial_hash_hits
        lookup_masks = None if fine_grained else self.coord.lookup_mask(token_len)
        for g_idx, db in enumerate(self.token_dbs):
            if not self._kv_cache_groups[g_idx].kv_cache_spec.prefix_cacheable:
                continue
            spec_block_size = self._kv_cache_groups[g_idx].kv_cache_spec.block_size
            key_prefixes = self._lookup_key_prefixes[g_idx]
            lookup_mask = None if lookup_masks is None else lookup_masks[g_idx]
            unit_size = self.hash_block_size if fine_grained else db.chunk_size
            max_units = (
                min(token_len, len(block_hashes) * self.hash_block_size) // unit_size
            )
            for unit_id in range(max_units):
                block_id = unit_id * unit_size // spec_block_size
                if lookup_mask is not None and (
                    block_id >= len(lookup_mask) or not lookup_mask[block_id]
                ):
                    continue
                end_token = (unit_id + 1) * unit_size
                h = block_hashes[end_token // self.hash_block_size - 1]
                hash_hex = h.hex()
                for key_prefix in key_prefixes:
                    candidate_keys.append(
                        PoolKey.build_key_string(key_prefix, hash_hex)
                    )
                candidate_meta.append((g_idx, bytes(h), end_token))

        if not candidate_keys:
            return MooncakeLookupResult(0)

        exists_fn = (
            self.store.batch_is_exist
            if self._selective_lease is None
            else self._selective_lease.probe
        )
        lookup_start = time.perf_counter()
        try:
            res = exists_fn(candidate_keys)
            self._record_kv_connector_operation(
                "lookup_exists",
                time.perf_counter() - lookup_start,
                len(candidate_keys),
            )
        except Exception as e:
            self._record_kv_connector_operation(
                "lookup_exists",
                time.perf_counter() - lookup_start,
                len(candidate_keys),
                status="error",
                num_failed_keys=len(candidate_keys),
            )
            logger.error("Remote connection failed in lookup: %s", e)
            return MooncakeLookupResult(0)

        # A (group, hash) is "present" only when every namespace that will be
        # loaded has it (per-group count: sharded groups need every rank's
        # shard, replicated groups one namespace per unique KV head).
        exists_set = set()
        pos = 0
        previous_page = None
        preceding_chunks_present = True
        for g_idx, hash_bytes, end_token in candidate_meta:
            spec_block_size = self._kv_cache_groups[g_idx].kv_cache_spec.block_size
            page = (g_idx, (end_token - 1) // spec_block_size)
            if page != previous_page:
                preceding_chunks_present = True
                previous_page = page
            count = len(self._lookup_key_prefixes[g_idx])
            present = all(res[pos + j] == 1 for j in range(count))
            # A page-prefix hit covers every Store chunk before its final chunk.
            if preceding_chunks_present and present:
                exists_set.add((g_idx, hash_bytes))
            if end_token % self.token_dbs[g_idx].chunk_size == 0:
                preceding_chunks_present &= present
            pos += count

        hit, load_masks = self._longest_hit(
            num_tokens, token_len, block_hashes, exists_set
        )
        if self._selective_lease is None:
            return hit
        return self._selective_lease.lease_hit(
            block_hashes,
            hit,
            load_masks,
            exists_set,
            functools.partial(self._longest_hit, num_tokens, token_len, block_hashes),
        )

    def _longest_hit(
        self,
        num_tokens: int,
        token_len: int,
        block_hashes: Sequence[BlockHash],
        exists_set: set[tuple[int, bytes]],
    ) -> tuple[MooncakeLookupResult, tuple[list[bool], ...]]:
        """Longest prefix hit and its load masks over the stored group/hash pairs."""
        cached_block_pool = ExternalCachedBlockPool(
            self.hash_block_size,
            exists_set,
        )
        load_masks, hit_length = self.coord.find_longest_cache_hit(
            block_hashes,
            token_len,
            cached_block_pool,
        )
        if hit_length >= num_tokens:
            usable_length = self.coord.align_lookup_length(num_tokens - 1)
            if usable_length <= 0:
                return MooncakeLookupResult(0), ()
            load_masks, hit_length = self.coord.find_longest_cache_hit(
                block_hashes,
                usable_length,
                cached_block_pool,
            )
        return (
            MooncakeLookupResult(
                hit_length,
                self._tail_key_boundaries(
                    block_hashes,
                    hit_length,
                    cached_block_pool,
                    load_masks,
                ),
            ),
            load_masks,
        )

    def _tail_key_boundaries(
        self,
        block_hashes: Sequence[BlockHash],
        hit_length: int,
        cached_block_pool: ExternalCachedBlockPool,
        load_masks: tuple[list[bool], ...],
    ) -> tuple[TailKeyBoundary, ...]:
        """Return the hash boundary used to store each group's tail block.

        With fine-grained prefix matching, `hit_length` may fall within a
        physical cache block and may not align with the hash boundary used to
        store that block. For each KV-cache group, return the token boundary
        whose hash was used as the store key.
        """
        if hit_length <= 0:
            return ()

        boundaries = []
        hit_boundary_hash_idx = hit_length // self.hash_block_size - 1
        for group_id, db in enumerate(self.token_dbs):
            if not load_masks[group_id] or not load_masks[group_id][-1]:
                continue
            chunk_id = cdiv(hit_length, db.chunk_size) - 1
            boundary_tokens = hit_length
            contains_hit_boundary = cached_block_pool.contains(
                group_id, block_hashes[hit_boundary_hash_idx]
            )
            if not self.coord.enable_partial_hash_hits:
                assert contains_hit_boundary
            if not contains_hit_boundary:
                next_chunk_hash_idx = min(
                    (chunk_id + 1) * db.chunk_size // self.hash_block_size,
                    len(block_hashes),
                )
                for hash_idx in range(hit_boundary_hash_idx + 1, next_chunk_hash_idx):
                    if cached_block_pool.contains(group_id, block_hashes[hash_idx]):
                        boundary_tokens = (hash_idx + 1) * self.hash_block_size
                        break
                else:
                    raise AssertionError(
                        f"No tail key found for cache group {group_id} at "
                        f"hit length {hit_length}"
                    )
            boundaries.append(TailKeyBoundary(group_id, boundary_tokens))
        return tuple(boundaries)


@dataclass
class LookupProcessSpec:
    """What the lookup process needs to answer lookups like rank 0."""

    socket_path: str
    store_config: MooncakeStoreConfig
    local_hostname: str
    coord: MooncakeStoreCoordinator
    token_dbs: list[ChunkedTokenDatabase]
    lookup_key_prefixes: tuple[tuple[str, ...], ...]
    hash_block_size: int
    selective_lease: bool


class _ProcessLookup(MooncakeStoreLookup):
    def __init__(self, spec: LookupProcessSpec):
        # Probing existence needs no segment, buffer pool or RDMA device.
        store_config = replace(
            spec.store_config,
            mode="standalone-store",
            metadata_server=spec.store_config.metadata_server or "P2PHANDSHAKE",
            protocol="tcp",
            device_name="",
            device_name_per_gpu=None,
            global_segment_size=0,
            local_buffer_size=_LOOKUP_CLIENT_BUFFER_SIZE,
        )
        from mooncake.store import MooncakeDistributedStore  # type: ignore

        self.store = MooncakeDistributedStore()
        setup_mooncake_store(self.store, store_config, spec.local_hostname)
        self.coord = spec.coord
        self.token_dbs = spec.token_dbs
        self.hash_block_size = spec.hash_block_size
        self._capacity_only = False
        self._kv_cache_groups = spec.coord.kv_cache_groups
        self._lookup_key_prefixes = spec.lookup_key_prefixes
        self.kv_connector_stats = MooncakeStoreConnectorStats()
        self._kv_connector_stats_lock = threading.Lock()
        self._selective_lease = (
            SelectiveLookupLease(
                self.store,
                self.token_dbs,
                self._kv_cache_groups,
                self._lookup_key_prefixes,
                self._record_kv_connector_operation,
            )
            if spec.selective_lease
            else None
        )

    def take_stats(self) -> bytes:
        stats = self.kv_connector_stats
        if stats.is_empty():
            return b""
        self.kv_connector_stats = MooncakeStoreConnectorStats()
        return msgspec.msgpack.encode(stats.data)


class LookupProcess:
    """Owns the lookup process, restarting it if it exits.

    A thread blocked on the process (so off the GIL) watches startup and exit.
    If the process exits after startup, lookups time out and count as misses
    until it is restarted.
    """

    def __init__(self, spec: LookupProcessSpec):
        self._spec = pickle.dumps(spec)
        self._lock = threading.Lock()
        self._closed = False
        self._ready = threading.Event()
        self._start_failed = False
        self.proc = self._spawn()
        threading.Thread(
            target=self._monitor, name="MooncakeLookupProcessMonitor", daemon=True
        ).start()

    def _spawn(self) -> "subprocess.Popen[bytes]":
        proc = subprocess.Popen(
            [sys.executable, "-c", f"from {__name__} import main; main()"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        assert proc.stdin is not None
        proc.stdin.write(self._spec)
        proc.stdin.close()
        return proc

    def _monitor(self) -> None:
        restart_delay_s = 1.0
        while True:
            proc = self.proc
            assert proc.stdout is not None
            start = time.monotonic()
            if proc.stdout.readline() == _LOOKUP_PROCESS_READY:
                logger.info(
                    "Mooncake lookup process (pid %d) ready in %.1fs",
                    proc.pid,
                    time.monotonic() - start,
                )
                restart_delay_s = 1.0
                self._ready.set()
            returncode = proc.wait()
            proc.stdout.close()
            if self._closed:
                return
            if not self._ready.is_set():
                logger.error(
                    "Mooncake lookup process failed to start (exit code %d)",
                    returncode,
                )
                self._start_failed = True
                self._ready.set()
                return
            logger.error(
                "Mooncake lookup process exited with code %d; store lookups "
                "miss until it restarts in %.0fs",
                returncode,
                restart_delay_s,
            )
            time.sleep(restart_delay_s)
            restart_delay_s = min(
                2 * restart_delay_s, _LOOKUP_PROCESS_MAX_RESTART_DELAY_S
            )
            with self._lock:
                if self._closed:
                    return
                self.proc = self._spawn()

    def wait_ready(self) -> None:
        """Block until the first process serves lookups."""
        if not self._ready.wait(_LOOKUP_PROCESS_START_TIMEOUT_S):
            raise RuntimeError(
                "Mooncake lookup process not ready after "
                f"{_LOOKUP_PROCESS_START_TIMEOUT_S:.0f}s"
            )
        if self._start_failed:
            raise RuntimeError("Mooncake lookup process failed to start")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.proc.kill()
        self.proc.wait()


def _exit_with_parent(parent_pid: int) -> None:
    while os.getppid() == parent_pid:
        time.sleep(1.0)
    os._exit(0)


def _serve(spec: LookupProcessSpec, ready_fd: int) -> None:
    lookup = _ProcessLookup(spec)
    ctx = zmq.Context()  # type: ignore[attr-defined]
    ipc_path = spec.socket_path.removeprefix("ipc://")
    if os.path.exists(ipc_path):
        os.unlink(ipc_path)
    socket = make_zmq_socket(
        ctx,
        spec.socket_path,
        zmq.REP,  # type: ignore[attr-defined]
        bind=True,
    )
    os.write(ready_fd, _LOOKUP_PROCESS_READY)
    os.close(ready_fd)
    while True:
        frames = socket.recv_multipart(copy=False)
        if bytes(frames[0]) != LOOKUP_MSG:
            logger.warning("Mooncake lookup process got unknown msg_type")
            socket.send(RESP_ERR)
            continue
        num_tokens = int.from_bytes(frames[1], byteorder="big")
        hash_len = int.from_bytes(frames[2], byteorder="big")
        try:
            result = lookup.lookup(
                num_tokens, BlobBlockHashes(frames[3].buffer, hash_len)
            )
        except Exception:
            logger.exception("Mooncake lookup failed")
            result = MooncakeLookupResult(0)
        response = encode_lookup_response(result)
        stats = lookup.take_stats()
        socket.send_multipart((response, stats) if stats else (response,))


def main() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    threading.Thread(
        target=_exit_with_parent, args=(os.getppid(),), daemon=True
    ).start()
    # Keep stdout for the ready signal only.
    ready_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    spec = pickle.loads(sys.stdin.buffer.read())
    _serve(spec, ready_fd)
