# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disaggregated prefill (NIXL, pull mode) under simulated execution.

Simulated execution loads this module's `NixlConnector` in place of the real
one. Its scheduler side is the real `NixlPullConnectorScheduler`: the prefill
instance holds a finished request's blocks under a lease, the decode instance
waits in WAITING_FOR_REMOTE_KVS, and the kv_transfer_params between them are
the real ones. Its worker side, `SimulatedNixlWorker`, moves no data and
needs no NIXL: the prefill and decode engines run in one process (see
`vllm.v1.worker.simulated.disaggregated`) and their workers find each other
by engine id.
"""

import heapq
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
    NixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    _group_kv_bytes_per_block,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.simulated.timing import Clock

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.worker.simulated.kv_transfer import TransferRate

# Simulated NIXL workers of the engines in this process, by engine id.
_workers: "weakref.WeakValueDictionary[str, SimulatedNixlWorker]" = (
    weakref.WeakValueDictionary()
)


def simulated_nixl_worker(engine_id: str) -> "SimulatedNixlWorker":
    """The simulated NIXL worker of an engine in this process."""
    return _workers[engine_id]


@dataclass(order=True)
class _Event:
    time: float
    req_id: str = field(compare=False)


class SimulatedNixlWorker:
    """The NIXL worker of one engine, as a timing model.

    A decode instance reads a request's KV from its prefill instance when the
    scheduler hands it over: the read starts once both instances' network
    links are free (reads are serial per link) and takes
    `TransferRate.seconds()` of the bytes of the blocks it fills. When it
    completes, the request is reported in finished_recving and the prefill
    instance is notified, which reports the request in finished_sending and
    so frees its blocks. Leases never expire: a simulated decode instance
    always reads or releases the blocks, which is what the real one's
    heartbeats keep the lease alive for (the scheduler sends them on the
    wall clock, which would make a simulation depend on the host's speed)."""

    # Attributes the NixlConnector reads off its worker.
    pcp_rank = 0
    pcp_dcp_sharded = False
    use_host_buffer = False
    copy_blocks = None
    xfer_handshake_metadata = None

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig
    ):
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = engine_id
        # Per transfer group, in the order of ReqMeta.local_block_ids.
        groups = kv_cache_config.kv_cache_groups
        self.bytes_per_block = [
            _group_kv_bytes_per_block(groups[group_id])
            for group_id in kv_cache_config.transfer_group_ids
        ]
        self.clock: Clock | None = None
        self.rate: TransferRate | None = None
        self.link_free_at = 0.0
        # Decode side: reads in flight, and whether the scheduler awaits them.
        self._recving: list[_Event] = []
        # Prefill side: read notifications, and the requests whose blocks may
        # still be read.
        self._notifs: list[_Event] = []
        self._reqs_to_process: set[str] = set()
        if engine_id in _workers:
            raise ValueError(f"Two simulated engines share engine_id {engine_id}.")
        _workers[engine_id] = self

    def configure(self, clock: Clock, rate: "TransferRate") -> None:
        self.clock = clock
        self.rate = rate

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        pass

    def set_host_xfer_buffer_ops(self, copy_operation: Any) -> None:
        pass

    def start_load_kv(self, metadata: NixlConnectorMetadata) -> None:
        assert self.clock is not None and self.rate is not None
        now = self.clock.now()
        for req_id, meta in metadata.reqs_to_recv.items():
            assert meta.remote is not None
            producer = _workers.get(meta.remote.engine_id)
            if producer is None:
                raise ValueError(
                    f"No simulated engine {meta.remote.engine_id} to read "
                    f"{req_id}'s KV from."
                )
            num_bytes = sum(
                len(ids) * size
                for ids, size in zip(meta.local_block_ids, self.bytes_per_block)
            )
            end = now
            if num_bytes:
                start = max(now, self.link_free_at, producer.link_free_at)
                end = start + self.rate.seconds(num_bytes)
                self.link_free_at = producer.link_free_at = end
            # A full local prefix hit, or an aborted request, only notifies.
            if meta.awaiting_kvs or any(meta.local_block_ids):
                heapq.heappush(self._recving, _Event(end, req_id))
            heapq.heappush(producer._notifs, _Event(end, meta.remote.request_id))

        self._reqs_to_process.update(metadata.reqs_in_batch)
        self._reqs_to_process.difference_update(metadata.reqs_not_processed)

    def get_transfer_results(self) -> KVConnectorTransferResults:
        assert self.clock is not None
        now = self.clock.now()
        done_recving = set()
        while self._recving and self._recving[0].time <= now:
            done_recving.add(heapq.heappop(self._recving).req_id)
        done_sending = set()
        while self._notifs and self._notifs[0].time <= now:
            req_id = heapq.heappop(self._notifs).req_id
            # As the real worker, drop reads of requests no longer tracked.
            if req_id in self._reqs_to_process:
                self._reqs_to_process.remove(req_id)
                done_sending.add(req_id)
        return KVConnectorTransferResults(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=set(),
        )

    def get_finished(self) -> tuple[set[str], set[str]]:
        results = self.get_transfer_results()
        return results.finished_sending, results.finished_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def get_kv_connector_stats(self) -> None:
        return None

    def next_completion(self) -> float | None:
        times = [q[0].time for q in (self._recving, self._notifs) if q]
        return min(times, default=None)

    def shutdown(self) -> None:
        if _workers.get(self.engine_id) is self:
            del _workers[self.engine_id]


class NixlConnector(NixlPullConnector):
    """The pull-mode NixlConnector with a simulated worker side. It keeps the
    real connector's name, which configuration checks look for."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        NixlBaseConnector.__init__(self, vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlPullConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
        else:
            self.worker = SimulatedNixlWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
            self.connector_worker = self.worker  # type: ignore[assignment]

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.worker.start_load_kv(self._connector_metadata)

    def set_xfer_handshake_metadata_pp_aware(self, metadata: dict) -> None:
        """No side channel: simulated workers find each other in process."""
