# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV connector transfers under simulated execution.

A connector's scheduler side runs unmodified; on the worker, a simulation that
moves no data completes each transfer after its modeled duration:
`SimulatedOffloadingWorker` for CPU offloading, and `kv_nixl` for NIXL.
"""

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.simulation import KVTransferTiming
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    get_offloading_group_ids,
    group_kv_bytes_per_block,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
from vllm.v1.worker.simulated.timing import Clock, load_step_log

logger = init_logger(__name__)

# Fitted records this many times slower than predicted (a connection's setup,
# or a NIXL agent's first postings to new memory) are outliers.
_OUTLIER_FACTOR = 20
_MAX_REFITS = 3
# Rows whose units per count vary within this ratio have a single shape.
_SAME_SHAPE_RATIO = 1.05


@dataclass(frozen=True)
class PostCost:
    """Host time a NIXL decode rank spends posting a step's reads, which holds
    its engine thread: `per_read_s` per read plus `per_run_s` per run of
    blocks that are consecutive on both instances. NIXL merges such a run
    into one descriptor per KV cache region, so the second term grows as the
    blocks a decode instance hands out scatter (once its cache has filled)."""

    per_read_s: float = 0.0
    per_run_s: float = 0.0

    def seconds(self, num_reads: int, num_runs: int) -> float:
        return num_reads * self.per_read_s + num_runs * self.per_run_s


def read_runs(local_block_ids: BlockIds, remote_block_ids: BlockIds) -> int:
    """Runs of blocks consecutive on both sides in one read, over its KV cache
    groups. Remote blocks the decode instance already holds (a local prefix
    hit) are not read: the last ones are paired with the local blocks."""
    runs = 0
    for local, remote in zip(local_block_ids, remote_block_ids):
        if not local:
            continue
        remote = remote[len(remote) - len(local) :]
        runs += 1 + sum(
            1
            for i in range(1, len(local))
            if local[i] != local[i - 1] + 1 or remote[i] != remote[i - 1] + 1
        )
    return runs


def nixl_reads(metadata: NixlConnectorMetadata) -> tuple[int, int]:
    """(reads, runs) a NIXL decode rank posts for a step's connector metadata."""
    num_reads = num_runs = 0
    for meta in metadata.reqs_to_recv.values():
        if meta.remote is None or not any(meta.local_block_ids):
            continue
        num_reads += 1
        num_runs += read_runs(meta.local_block_ids, meta.remote.block_ids)
    return num_reads, num_runs


@dataclass(frozen=True)
class TransferRate:
    """Duration of one transfer: `latency_s + num_bytes / bytes_per_s`."""

    latency_s: float
    bytes_per_s: float

    def seconds(self, num_bytes: int) -> float:
        return self.latency_s + num_bytes / self.bytes_per_s


def kv_transfer_rates(
    timing: KVTransferTiming | str,
) -> tuple[TransferRate, TransferRate | None]:
    """(load, store) rates of `SimulationConfig.kv_transfer_timing`; store is
    None if not given."""
    if isinstance(timing, str):
        return transfer_rates_from_log(timing)
    assert timing.load_gbps is not None
    load = TransferRate(timing.load_latency_us * 1e-6, timing.load_gbps * 1e9)
    store = (
        TransferRate(timing.store_latency_us * 1e-6, timing.store_gbps * 1e9)
        if timing.store_gbps is not None
        else None
    )
    return load, store


def kv_post_cost(timing: KVTransferTiming | str) -> PostCost:
    """NIXL read posting cost of `SimulationConfig.kv_transfer_timing`."""
    if isinstance(timing, str):
        return post_cost_from_log(timing)
    return PostCost(timing.post_us_per_read * 1e-6, timing.post_us_per_run * 1e-6)


def post_cost_from_log(path: str) -> PostCost:
    """Least-squares `PostCost` of the read postings recorded in a step log,
    without outliers."""
    rows = [
        (rec["kv_post"]["reads"], rec["kv_post"]["runs"], rec["kv_post"]["seconds"])
        for rec in load_step_log(path, kind="kv_post")
    ]
    if not rows:
        return PostCost()
    a = np.asarray(rows, dtype=np.float64)
    # Both terms grow with reads + runs: screen out the slowest postings
    # before the first fit, which they would otherwise dominate.
    unit_s = a[:, 2] / (a[:, 0] + a[:, 1])
    a = a[unit_s <= _OUTLIER_FACTOR * np.median(unit_s)]
    a, per_read_s, per_run_s = _fit(a)
    if _single_shape(a):
        logger.warning(
            "The reads recorded in %s are all alike, so the posting cost of "
            "reads into scattered blocks is extrapolated. Record until the "
            "decode instance's KV cache has filled and its blocks are reused.",
            path,
        )
    cost = PostCost(per_read_s, per_run_s)
    logger.info(
        "NIXL read posting from %s: %.1f us per read + %.2f us per run (%d records)",
        path,
        per_read_s * 1e6,
        per_run_s * 1e6,
        len(rows),
    )
    return cost


def _single_shape(a: np.ndarray) -> bool:
    units_per_count = a[:, 1] / a[:, 0]
    return units_per_count.max() <= _SAME_SHAPE_RATIO * units_per_count.min()


def _lstsq(a: np.ndarray) -> tuple[float, float]:
    """Least-squares seconds per count and per unit of (count, units,
    seconds) rows; all to the units if the two are not separable."""
    (per_count, per_unit), *_ = np.linalg.lstsq(a[:, :2], a[:, 2], rcond=None)
    if per_count < 0 or per_unit <= 0 or _single_shape(a):
        per_count, per_unit = 0.0, a[:, 2].sum() / a[:, 1].sum()
    return float(per_count), float(per_unit)


def _fit(a: np.ndarray) -> tuple[np.ndarray, float, float]:
    """`_lstsq`, refitted without outliers; also returns the rows kept."""
    per_count, per_unit = _lstsq(a)
    for _ in range(_MAX_REFITS):
        typical = a[:, 2] <= _OUTLIER_FACTOR * (
            a[:, 0] * per_count + a[:, 1] * per_unit
        )
        if typical.all():
            break
        a = a[typical]
        per_count, per_unit = _lstsq(a)
    return a, per_count, per_unit


def transfer_rates_from_log(path: str) -> tuple[TransferRate, TransferRate | None]:
    """(load, store) rates fitted by least squares, without outliers, to the
    transfers of a step log recorded with a KV connector; store is None if
    none were recorded."""
    rows: dict[str, list[tuple[float, float, float]]] = {"load": [], "store": []}
    for rec in load_step_log(path, kind="kv_transfers"):
        for direction, t in rec["kv_transfers"].items():
            rows[direction].append((t["count"], t["bytes"], t["seconds"]))
    rates: list[TransferRate | None] = []
    for direction, direction_rows in rows.items():
        a = np.asarray(direction_rows, dtype=np.float64).reshape(-1, 3)
        if not a[:, 1].any():
            if direction == "store":
                rates.append(None)
                continue
            raise ValueError(f"{path} has no recorded KV {direction}s")
        # Bytes in GB keep the two columns on a similar scale.
        a = a[a[:, 1] > 0] * [1.0, 1e-9, 1.0]
        kept, latency_s, s_per_gb = _fit(a)
        rate = TransferRate(latency_s, 1e9 / s_per_gb)
        rates.append(rate)
        logger.info(
            "KV %ss from %s: %.1f us + %.2f GB/s (%d of %d steps)",
            direction,
            path,
            rate.latency_s * 1e6,
            rate.bytes_per_s * 1e-9,
            len(kept),
            len(direction_rows),
        )
    assert rates[0] is not None
    return rates[0], rates[1]


def completed_transfers(
    kv_output: KVConnectorOutput | None,
) -> dict[str, dict[str, float]] | None:
    """The KV transfers that completed in a step, per direction, as a step log's
    `kv_transfers` record holds them: CPU offloading's loads and stores, or a
    NIXL decode instance's reads."""
    if kv_output is None:
        return None
    meta = kv_output.kv_connector_worker_meta
    stats = kv_output.kv_connector_stats
    if isinstance(meta, OffloadingWorkerMetadata):
        transfer_stats = meta.transfer_stats
        transfers = {
            direction: {"count": len(s.sizes), "bytes": s.bytes, "seconds": s.time}
            for direction, s in (
                ("load", transfer_stats.load),
                ("store", transfer_stats.store),
            )
            if s.sizes
        }
    elif isinstance(stats, NixlKVConnectorStats) and stats.num_successful_transfers:
        transfers = {
            "load": {
                "count": stats.num_successful_transfers,
                "bytes": sum(stats.data["bytes_transferred"]),
                "seconds": sum(stats.data["transfer_duration"]),
            }
        }
    else:
        return None
    return transfers or None


_LOAD, _STORE = 0, 1


@dataclass
class _Transfer:
    job_id: int
    num_bytes: int
    start: float
    end: float


class SimulatedOffloadingWorker(OffloadingWorker):
    """Times KV transfers between device and host memory without moving data.

    As with the CPU offloading worker's copy streams, each direction is a
    serial queue, and a transfer starts only once the device has finished the
    work enqueued before it was submitted (which writes the blocks a store
    reads, or last reads the blocks a load overwrites)."""

    def __init__(
        self,
        clock: Clock,
        load: TransferRate,
        store: TransferRate,
        bytes_per_block: Sequence[int],
        device_free_at: Callable[[], float],
    ):
        self.clock = clock
        self.rates = (load, store)
        # Per offloaded KV cache group, in the order of GPULoadStoreSpec groups.
        self.bytes_per_block = bytes_per_block
        self.device_free_at = device_free_at
        # Indexed by _LOAD and _STORE.
        self._queues: tuple[deque[_Transfer], deque[_Transfer]] = (deque(), deque())
        self._free_at = [0.0, 0.0]

    def _submit(self, direction: int, job_id: int, gpu_spec: GPULoadStoreSpec) -> bool:
        num_bytes = sum(
            n * b for n, b in zip(gpu_spec.group_sizes, self.bytes_per_block)
        )
        start = max(self.clock.now(), self.device_free_at(), self._free_at[direction])
        end = start + self.rates[direction].seconds(num_bytes)
        self._free_at[direction] = end
        self._queues[direction].append(_Transfer(job_id, num_bytes, start, end))
        return True

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._submit(_STORE, job_id, src_spec)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        return self._submit(_LOAD, job_id, dst_spec)

    def get_finished(self) -> list[TransferResult]:
        now = self.clock.now()
        results = []
        for queue in self._queues:
            while queue and queue[0].end <= now:
                t = queue.popleft()
                results.append(
                    TransferResult(
                        job_id=t.job_id,
                        success=True,
                        transfer_size=t.num_bytes,
                        transfer_time=t.end - t.start,
                    )
                )
        return results

    def wait(self, job_ids: set[int]) -> None:
        ends = [t.end for queue in self._queues for t in queue if t.job_id in job_ids]
        if ends:
            self.clock.sleep_until(max(ends))

    def next_completion(self) -> float | None:
        ends = [queue[0].end for queue in self._queues if queue]
        return min(ends, default=None)


class SimulatedTransfers(Protocol):
    def next_completion(self) -> float | None:
        """When the next in-flight transfer completes, if any."""
        ...


class SimulatedKVConnector(ActiveKVConnector):
    """The model runner's KV connector hooks, with the connector's
    worker-side transfers simulated instead of registering KV caches."""

    def __init__(self, vllm_config: VllmConfig, transfers: SimulatedTransfers):
        self.transfers = transfers
        super().__init__(vllm_config, {})

    def register_kv_caches(self, kv_caches_dict: dict[str, torch.Tensor]) -> None:
        connector = self.kv_connector
        if isinstance(connector, OffloadingConnector):
            assert connector.connector_worker is not None
            assert isinstance(self.transfers, SimulatedOffloadingWorker)
            connector.connector_worker.worker = self.transfers


def simulated_kv_connector(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    clock: Clock,
    device_free_at: Callable[[], float],
) -> SimulatedKVConnector:
    from vllm.v1.worker.simulated.kv_nixl import (
        NixlConnector,
        NixlPushConnector,
        SimulatedNixlWorker,
    )

    timing = vllm_config.simulation_config.kv_transfer_timing
    if timing is None:
        raise ValueError(
            "Simulated KV transfer needs simulation_config.kv_transfer_timing: "
            "transfer bandwidths, or the path of a step log recorded with the "
            "connector."
        )
    load, store = kv_transfer_rates(timing)
    connector = get_kv_transfer_group()
    transfers: SimulatedTransfers
    if isinstance(connector, OffloadingConnector):
        if store is None:
            raise ValueError("KV offloading needs kv_transfer_timing store_gbps.")
        groups = kv_cache_config.kv_cache_groups
        transfers = SimulatedOffloadingWorker(
            clock,
            load,
            store,
            bytes_per_block=[
                group_kv_bytes_per_block(groups[group_id])
                for group_id in get_offloading_group_ids(kv_cache_config)
            ],
            device_free_at=device_free_at,
        )
    elif isinstance(connector, NixlConnector | NixlPushConnector):
        assert isinstance(connector.connector_worker, SimulatedNixlWorker)
        transfers = connector.connector_worker
        transfers.configure(clock, load, kv_post_cost(timing))
    else:
        raise ValueError(
            "Simulated execution supports the OffloadingConnector and the "
            f"NixlConnector, not {type(connector).__name__}."
        )
    return SimulatedKVConnector(vllm_config, transfers)
