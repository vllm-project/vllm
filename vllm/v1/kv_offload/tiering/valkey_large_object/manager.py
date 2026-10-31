# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ValkeyLargeObjectTierManager: KV cache offloading to Valkey with the
valkey-large-object module, moving chunks by RDMA.

The primary view is registered as disjoint RDMA regions of about
``rdma_region_bytes`` each, covering whole chunks; each transfer is a window of
its chunk's region, so the server reads and writes the CPU KV cache directly.
A glide region serves one transfer at a time, so each region has a lock and
transfers to chunks in different regions run in parallel. Registered memory
totals one copy of the primary view.

RDMA transfers have no timeout and can only be cancelled by closing the
client. A watchdog thread closes the client when a transfer overruns
``timeout_ms`` or the transport fails, then registers a fresh client.
"""

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.backpressure import BackpressureDetector
from vllm.v1.kv_offload.tiering.valkey.manager import ValkeyTierManager
from vllm.v1.kv_offload.tiering.valkey_large_object.client import (
    LargeObjectClient,
    ValkeyTransportError,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)

T = TypeVar("T")


class _Transport:
    """One client and its registered regions, replaced as a unit."""

    __slots__ = ("client", "failed", "torn_down", "in_flight", "condition")

    def __init__(self, client: LargeObjectClient) -> None:
        self.client = client
        # Set once a transfer fails or overruns; new transfers fail fast.
        self.failed = False
        self.torn_down = False
        # Thread ident -> monotonic start time of its transfer.
        self.in_flight: dict[int, float] = {}
        self.condition = threading.Condition()


class ValkeyLargeObjectTierManager(ValkeyTierManager):
    """Secondary tier backed by Valkey with the valkey-large-object module."""

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        startup_nodes: str,
        rdma_provider: str = "efa-direct",
        rdma_interface: str | None = None,
        rdma_region_bytes: int = 16 << 20,
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
        rdma_provider: ``"efa-direct"``, or ``"tcp"`` for RDMA emulated over
            TCP (development and CI).
        rdma_interface: Fabric domain to pin to when a host has several.
        rdma_region_bytes: Target size of each registered region, rounded
            down to whole chunks (at least one). Smaller regions allow more
            parallel transfers at the cost of more registrations.
        cluster_mode: Connect as a cluster client.
        key_prefix: Prefix for every key this tier writes.
        ttl_seconds: Expire stored chunks after this many seconds.
        timeout_ms: Deadline for one chunk transfer; an overrun reconnects.
        n_read_threads: Number of read-priority I/O threads.
        n_write_threads: Number of write-priority I/O threads.
        enable_kv_events: Emit BlockStored KV events for blocks
            successfully stored to this tier. Effective only when KV
            cache events are enabled globally (kv_events_config).
        locality: Whether this tier's storage is LOCAL or REMOTE relative
            to the publishing vLLM instance.
        backpressure_detector: Optional backpressure detector.

        """
        # Read by _create_client during the base constructor.
        self._rdma_provider = rdma_provider
        self._rdma_interface = rdma_interface
        assert primary_kv_view.strides is not None
        chunk_bytes = primary_kv_view.strides[0]
        self._region_bytes = max(1, rdma_region_bytes // chunk_bytes) * chunk_bytes
        num_regions = -(-primary_kv_view.nbytes // self._region_bytes)
        self._region_locks = [threading.Lock() for _ in range(num_regions)]
        super().__init__(
            offloading_spec,
            primary_kv_view,
            tier_type,
            startup_nodes,
            cluster_mode=cluster_mode,
            key_prefix=key_prefix,
            ttl_seconds=ttl_seconds,
            timeout_ms=timeout_ms,
            n_read_threads=n_read_threads,
            n_write_threads=n_write_threads,
            enable_kv_events=enable_kv_events,
            locality=locality,
            backpressure_detector=backpressure_detector,
        )
        self.client: LargeObjectClient
        self._transport = _Transport(self.client)
        self._stopping = threading.Event()
        self._wake_watchdog = threading.Event()
        self._watchdog = threading.Thread(
            target=self._watch, name="vllm_kv_lo_watch", daemon=True
        )
        self._watchdog.start()

    @override
    def _create_client(self) -> LargeObjectClient:
        from vllm.v1.kv_offload.tiering.valkey_large_object.glide_client import (
            GlideLargeObjectClient,
        )

        client = GlideLargeObjectClient(
            self._startup_nodes,
            self._cluster_mode,
            self._timeout_ms,
            self._rdma_provider,
            self._rdma_interface,
        )
        self._register_regions(client)
        return client

    def _register_regions(self, client: LargeObjectClient) -> None:
        """Register the primary view as consecutive regions of
        ``self._region_bytes``; region ``i`` starts at ``i * _region_bytes``."""
        try:
            for start in range(0, len(self._primary_bytes), self._region_bytes):
                client.register_region(
                    self._primary_bytes[start : start + self._region_bytes]
                )
        except Exception:
            client.close()
            client.release_regions()
            raise

    def _transfer(
        self,
        offset: int,
        operation: Callable[[LargeObjectClient, int, int], T],
    ) -> T:
        """Run ``operation(client, region, region_offset)`` for the chunk at
        ``offset`` in the primary view, holding that region's lock."""
        region, region_offset = divmod(offset, self._region_bytes)
        with self._region_locks[region]:
            transport = self._transport
            thread = threading.get_ident()
            with transport.condition:
                if transport.failed:
                    raise ValkeyTransportError("transport is reconnecting")
                transport.in_flight[thread] = time.monotonic()
            try:
                return operation(transport.client, region, region_offset)
            except ValkeyTransportError:
                with transport.condition:
                    transport.failed = True
                self._wake_watchdog.set()
                raise
            finally:
                with transport.condition:
                    del transport.in_flight[thread]
                    transport.condition.notify_all()

    @override
    def _store_chunk(self, key: str, offset: int) -> None:
        self._transfer(
            offset,
            lambda client, region, region_offset: client.rdma_set(
                key, region, region_offset, self._block_size_bytes
            ),
        )
        if self._ttl_seconds is not None:
            # The chunk is stored; without a TTL it lives until Valkey evicts it.
            try:
                self._transport.client.expire(key, self._ttl_seconds)
            except Exception as error:
                logger.debug("EXPIRE %s failed: %s", key, error)

    @override
    def _load_chunk(self, key: str, offset: int) -> bool:
        written = self._transfer(
            offset,
            lambda client, region, region_offset: client.rdma_get(
                key, region, region_offset, self._block_size_bytes
            ),
        )
        return written == self._block_size_bytes

    def _watch(self) -> None:
        timeout = self._timeout_ms / 1000
        while not self._stopping.is_set():
            self._wake_watchdog.wait(timeout / 4)
            self._wake_watchdog.clear()
            if self._stopping.is_set():
                return
            transport = self._transport
            with transport.condition:
                now = time.monotonic()
                overdue = any(
                    timeout < now - start for start in transport.in_flight.values()
                )
                if not (transport.failed or overdue):
                    continue
                transport.failed = True
            self._replace(transport)

    def _replace(self, transport: _Transport) -> None:
        if not transport.torn_down:
            logger.warning(
                "Valkey large-object tier '%s': a transfer failed or exceeded "
                "%d ms; reconnecting.",
                self.tier_type,
                self._timeout_ms,
            )
            self._tear_down(transport)
        try:
            client = self._create_client()
        except Exception as error:
            # Retried on the next tick; transfers fail fast until then.
            logger.debug("Reconnecting to %s failed: %s", self._startup_nodes, error)
            return
        self.client = client
        self._transport = _Transport(client)

    def _tear_down(self, transport: _Transport) -> None:
        # Closing revokes the region before blocked transfers return, so the
        # server cannot write into chunks the primary tier reclaims after their
        # jobs fail.
        transport.client.close()
        with transport.condition:
            transport.condition.wait_for(lambda: not transport.in_flight)
        transport.client.release_regions()
        transport.torn_down = True

    @override
    def _close_client(self) -> None:
        # Runs after the pool has drained, with the watchdog still able to
        # cancel a hung transfer during that drain.
        self._stopping.set()
        self._wake_watchdog.set()
        self._watchdog.join()
        if not self._transport.torn_down:
            self._tear_down(self._transport)
