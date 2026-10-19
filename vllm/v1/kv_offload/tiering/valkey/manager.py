# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ValkeyTierManager: KV cache offloading to a stock Valkey server.

Each chunk is one string key holding the chunk's row of the primary view.
Stores pass the row to SET and loads GET straight into it, so no Python-side
copy is made in either direction.

Key naming:  <key_prefix>/<tier_type>/<model>_<digest>_r0/<hhh>/<hh>_g<group>/
             <hash_hex>.bin  (FileMapper's layout, so incompatible models and
             layouts never share keys)
"""

import functools
import json
from collections.abc import Iterable
from typing import TYPE_CHECKING

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.backpressure import BackpressureDetector
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.pooled import PartialLoadError, PooledTierManager
from vllm.v1.kv_offload.tiering.valkey.client import ValkeyClient

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class ValkeyAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for ValkeyTierManager."""

    def __init__(self, tier: "ValkeyTierManager", tier_type: str) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        names = [self._tier.key_mapper.get_file_name(k) for k in keys]
        return self._tier.client.exists_each(names)


class ValkeyTierManager(PooledTierManager):
    """Secondary tier backed by a Valkey server or cluster.

    Eviction is left to Valkey: set ``ttl_seconds`` or a ``maxmemory-policy``.
    A chunk evicted between lookup and load fails that load and is recomputed.

    Cross-process sharing works as for the fs tier: identical token content
    maps to identical keys across instances pointed at the same Valkey.
    """

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        startup_nodes: str,
        cluster_mode: bool = False,
        key_prefix: str = "vllm",
        ttl_seconds: int | None = None,
        timeout_ms: int = 5000,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
        backpressure_detector: BackpressureDetector | None = None,
    ):
        """Args:
        offloading_spec: Contains normalized offloading configuration and
            blocks_per_chunk.
        primary_kv_view: Memoryview of the primary tier's CPU KV cache.
        tier_type: Tier type identifier, set by SecondaryTierFactory.
        startup_nodes: Comma-separated ``host:port`` seed nodes.
        cluster_mode: Connect as a cluster client.
        key_prefix: Prefix for every key this tier writes.
        ttl_seconds: Expire stored chunks after this many seconds.
        timeout_ms: Per-request timeout. Size it for one chunk transfer.
        n_read_threads: Number of read-priority I/O threads.
        n_write_threads: Number of write-priority I/O threads.
        enable_kv_events: Emit BlockStored KV events for blocks
            successfully stored to this tier. Effective only when KV
            cache events are enabled globally (kv_events_config).
        locality: Whether this tier's storage is LOCAL or REMOTE relative
            to the publishing vLLM instance.
        backpressure_detector: Optional backpressure detector.

        """
        super().__init__(
            offloading_spec,
            primary_kv_view,
            tier_type,
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_valkey",
            enable_kv_events=enable_kv_events,
            locality=locality,
            backpressure_detector=backpressure_detector,
        )
        self._startup_nodes = startup_nodes
        self._cluster_mode = cluster_mode
        self._ttl_seconds = ttl_seconds
        self._timeout_ms = timeout_ms
        self._primary_bytes = primary_kv_view.cast("B")

        # Tier type in the namespace: plain and large-object keys are different
        # Valkey types and must never be read as each other.
        self.key_mapper = FileMapper.from_offloading_spec(
            root_dir=f"{key_prefix}/{tier_type}",
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
        )

        try:
            self.client: ValkeyClient = self._create_client()
            # Doubles as the connectivity probe; records what the keys hold.
            run_config = json.dumps(
                self.key_mapper.get_run_config(), indent=2, sort_keys=True
            ).encode()
            self.client.set(
                self.key_mapper.get_config_file_path(), memoryview(run_config), None
            )
        except Exception as error:
            self._pool.shutdown(wait=True)
            raise RuntimeError(
                f"Valkey tier '{tier_type}' cannot reach {startup_nodes}: {error}"
            ) from error

        self._lookup_manager = ValkeyAsyncLookupManager(
            tier=self, tier_type=self.tier_type
        )

    def _create_client(self) -> ValkeyClient:
        from vllm.v1.kv_offload.tiering.valkey.glide_client import GlideValkeyClient

        return GlideValkeyClient(
            self._startup_nodes, self._cluster_mode, self._timeout_ms
        )

    def _close_client(self) -> None:
        self.client.close()

    def _chunk_view(self, offset: int) -> memoryview:
        return self._primary_bytes[offset : offset + self._block_size_bytes]

    def _store_chunk(self, key: str, offset: int) -> None:
        self.client.set(key, self._chunk_view(offset), self._ttl_seconds)

    def _load_chunk(self, key: str, offset: int) -> bool:
        """Returns whether the chunk at ``key`` was loaded whole."""
        written = self.client.get_into(key, self._chunk_view(offset))
        return written == self._block_size_bytes

    def _store_chunks(self, keys: list[str], offsets: list[int]) -> None:
        for key, offset in zip(keys, offsets):
            self._store_chunk(key, offset)

    def _load_chunks(self, keys: list[str], offsets: list[int]) -> None:
        for index, (key, offset) in enumerate(zip(keys, offsets)):
            try:
                loaded = self._load_chunk(key, offset)
            except Exception as error:
                raise PartialLoadError(index) from error
            if not loaded:
                raise PartialLoadError(index)

    def _key_names(self, job_metadata: TransferJob) -> list[str]:
        return [self.key_mapper.get_file_name(key) for key in job_metadata.keys]

    @override
    def submit_store(self, job_metadata: TransferJob) -> None:
        task = functools.partial(
            self._store_chunks,
            self._key_names(job_metadata),
            self._chunk_offsets(job_metadata),
        )
        self._enqueue_store(job_metadata, task)

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        task = functools.partial(
            self._load_chunks,
            self._key_names(job_metadata),
            self._chunk_offsets(job_metadata),
        )
        self._enqueue_load(job_metadata, task)

    @override
    def shutdown(self) -> None:
        super().shutdown()
        self._close_client()
