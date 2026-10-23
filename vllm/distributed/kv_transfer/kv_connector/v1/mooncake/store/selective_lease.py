# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selective read leases for Mooncake store prefix lookups.

``batch_is_exist`` leases every key it finds, so a lookup renews the whole
prompt history, including the Mamba state at every earlier block boundary of a
hybrid model. With ``VLLM_MOONCAKE_STORE_SELECTIVE_LEASE`` the lookup probes
lease-free (``batch_probe_key``) and leases only the keys the hit will load.
"""

import time
from collections.abc import Callable, Sequence
from importlib import metadata
from typing import Any, Protocol

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
    ChunkedTokenDatabase,
    MooncakeLookupResult,
    PoolKey,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import KVCacheGroupSpec

logger = init_logger(__name__)

# Lease rounds before giving up on a hit whose keys keep being evicted between
# probe and lease. Only a hit whose lease found every key is returned.
_LEASE_ROUNDS = 2
# Key probed at startup to check that the master serves batch_probe_key.
_SENTINEL_KEY = "vllm-selective-lease-probe"
_MOONCAKE_DISTRIBUTIONS = (
    "mooncake-transfer-engine",
    "mooncake-transfer-engine-cuda13",
    "mooncake-transfer-engine-non-cuda",
)


class _RecordOperation(Protocol):
    def __call__(
        self,
        operation: str,
        duration_seconds: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None: ...


def _installed_mooncake() -> str:
    for dist in _MOONCAKE_DISTRIBUTIONS:
        try:
            return f"{dist} {metadata.version(dist)}"
        except metadata.PackageNotFoundError:
            continue
    return "unknown version"


class SelectiveLookupLease:
    """Leases only the keys a prefix-lookup hit will load.

    The lookup probes its candidates with :meth:`probe`, computes the hit as
    usual, and passes it through :meth:`lease_hit`.
    """

    def __init__(
        self,
        store: Any,
        coord: MooncakeStoreCoordinator,
        token_dbs: Sequence[ChunkedTokenDatabase],
        kv_cache_groups: Sequence[KVCacheGroupSpec],
        lookup_key_prefixes: Sequence[Sequence[str]],
        record_operation: _RecordOperation,
    ):
        self._check_supported(store)
        self.store = store
        self.coord = coord
        self.token_dbs = token_dbs
        self.kv_cache_groups = kv_cache_groups
        self.lookup_key_prefixes = lookup_key_prefixes
        self.record_operation = record_operation

    @staticmethod
    def _check_supported(store: Any) -> None:
        """Raise unless the client and the master both serve lease-free probes.

        A master without the probe RPC answers each key with an error code
        instead of raising, which would turn every lookup into a silent miss.
        """
        if not callable(getattr(store, "batch_probe_key", None)):
            raise RuntimeError(
                "VLLM_MOONCAKE_STORE_SELECTIVE_LEASE needs Mooncake 0.3.14 or "
                f"newer with batch_probe_key; installed: {_installed_mooncake()}"
            )
        res = store.batch_probe_key([_SENTINEL_KEY])
        if len(res) != 1 or res[0] < 0:
            raise RuntimeError(
                "VLLM_MOONCAKE_STORE_SELECTIVE_LEASE: the Mooncake master "
                f"rejected batch_probe_key (result {list(res)}); the master and "
                "every client must run the same Mooncake version "
                f"(installed client: {_installed_mooncake()})"
            )

    def probe(self, keys: list[str]) -> list[int]:
        """Existence of ``keys``, without granting a lease."""
        return self.store.batch_probe_key(keys)

    def lease_hit(
        self,
        block_hashes: Sequence[BlockHash],
        hit: MooncakeLookupResult,
        exists_set: set[tuple[int, bytes]],
        longest_hit: Callable[[set[tuple[int, bytes]]], MooncakeLookupResult],
    ) -> MooncakeLookupResult:
        """Lease the keys a load of ``hit`` reads and return the hit to load.

        ``exists_set`` holds the probed ``(group, hash)`` pairs and
        ``longest_hit`` recomputes the hit from them. A key evicted since the
        probe shortens the hit, whose tail keys then differ, so the shorter hit
        is leased in turn. Returns an empty hit if the lease fails or the hit
        keeps shrinking.
        """
        try:
            for _ in range(_LEASE_ROUNDS):
                if hit.hit_length <= 0:
                    return hit
                evicted = self._lease(block_hashes, hit)
                if evicted is None:
                    break
                if not evicted:
                    return hit
                exists_set = exists_set - evicted
                hit = longest_hit(exists_set)
        except Exception:
            # The lookup server has no handler of its own; report a miss.
            logger.exception("Failed to lease a Mooncake lookup hit")
        return MooncakeLookupResult(0)

    def loaded_keys(
        self,
        block_hashes: Sequence[BlockHash],
        hit: MooncakeLookupResult,
    ) -> tuple[list[str], list[tuple[int, bytes]]]:
        """Keys a load of ``hit`` reads, in every rank namespace, and their
        ``(group, hash)`` pairs.

        Mirrors ``KVCacheStoreRecvingThread._handle_request``: each group's
        load mask, with the tail chunk keyed by the hit's tail boundary.
        """
        hit_length = hit.hit_length
        load_masks = self.coord.load_mask(block_hashes, hit_length)
        tail_tokens = {b.group_id: b.num_tokens for b in hit.tail_key_boundaries}
        keys: list[str] = []
        key_meta: list[tuple[int, bytes]] = []
        for g_idx, db in enumerate(self.token_dbs):
            if not self.kv_cache_groups[g_idx].kv_cache_spec.prefix_cacheable:
                continue
            mask = load_masks[g_idx]
            key_prefixes = self.lookup_key_prefixes[g_idx]
            for start, end, block_hash in db.process_tokens(hit_length, block_hashes):
                chunk_idx = start // db.block_size
                if chunk_idx >= len(mask) or not mask[chunk_idx]:
                    continue
                if end == hit_length and g_idx in tail_tokens:
                    block_hash = block_hashes[
                        tail_tokens[g_idx] // db.hash_block_size - 1
                    ]
                keys.extend(
                    PoolKey.build_key_string(key_prefix, block_hash.hex())
                    for key_prefix in key_prefixes
                )
                key_meta.extend([(g_idx, bytes(block_hash))] * len(key_prefixes))
        return keys, key_meta

    def _lease(
        self,
        block_hashes: Sequence[BlockHash],
        hit: MooncakeLookupResult,
    ) -> set[tuple[int, bytes]] | None:
        """Lease the keys a load of ``hit`` reads.

        Returns the ``(group, hash)`` pairs no longer in the store, or None
        when the lease RPC fails, including per-key error results.
        """
        keys, key_meta = self.loaded_keys(block_hashes, hit)
        if not keys:
            return set()
        start = time.perf_counter()
        try:
            res = self.store.batch_is_exist(keys)
        except Exception as e:
            logger.error("Remote connection failed leasing a lookup hit: %s", e)
            res = [-1] * len(keys)
        num_errors = sum(1 for r in res if r < 0)
        if num_errors:
            self.record_operation(
                "lookup_lease",
                time.perf_counter() - start,
                len(keys),
                status="error",
                num_failed_keys=num_errors,
            )
            return None
        evicted = [meta for meta, r in zip(key_meta, res, strict=True) if r != 1]
        self.record_operation(
            "lookup_lease",
            time.perf_counter() - start,
            len(keys),
            status="partial_failure" if evicted else "ok",
            num_failed_keys=len(evicted),
        )
        return set(evicted)
