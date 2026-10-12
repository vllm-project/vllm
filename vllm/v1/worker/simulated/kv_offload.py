# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache offloading under simulated execution.

The `OffloadingConnector` runs unmodified on the scheduler side: the CPU
cache's lookups and evictions, and the load and store jobs it issues, are the
real ones. On the worker, `SimulatedOffloadingWorker` takes the place of the
CPU offloading worker: no data moves, and a transfer completes once its
modeled duration has elapsed on the simulation clock.
"""

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    _group_kv_bytes_per_block,
    get_offloading_group_ids,
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


@dataclass(frozen=True)
class TransferRate:
    """Duration of one transfer: `latency_s + num_bytes / bytes_per_s`."""

    latency_s: float
    bytes_per_s: float

    def seconds(self, num_bytes: int) -> float:
        return self.latency_s + num_bytes / self.bytes_per_s


def kv_offload_rates(
    timing: dict[str, float] | str,
) -> tuple[TransferRate, TransferRate]:
    """(load, store) rates from `SimulationConfig.kv_offload_timing`."""
    if isinstance(timing, str):
        return transfer_rates_from_log(timing)
    load, store = (
        TransferRate(
            latency_s=timing.get(f"{direction}_latency_us", 0.0) * 1e-6,
            bytes_per_s=timing[f"{direction}_gbps"] * 1e9,
        )
        for direction in ("load", "store")
    )
    return load, store


def transfer_rates_from_log(path: str) -> tuple[TransferRate, TransferRate]:
    """(load, store) rates fitted by least squares to the transfers of a step
    log recorded with KV offloading. Each `kv_transfers` record sums, per
    direction, the transfers that completed in one engine step."""
    rows: dict[str, list[tuple[float, float, float]]] = {"load": [], "store": []}
    for rec in load_step_log(path, kv_transfers=True):
        for direction, t in rec["kv_transfers"].items():
            rows[direction].append((t["count"], t["bytes"], t["seconds"]))
    rates = []
    for direction, direction_rows in rows.items():
        a = np.asarray(direction_rows, dtype=np.float64).reshape(-1, 3)
        if not a[:, 1].any():
            raise ValueError(f"{path} has no recorded KV offloading {direction}s")
        # Bytes in GB keep the two columns on a similar scale.
        a[:, 1] *= 1e-9
        (latency_s, s_per_gb), *_ = np.linalg.lstsq(a[:, :2], a[:, 2], rcond=None)
        if latency_s < 0 or s_per_gb <= 0:
            # Latency not separable from bandwidth: attribute all to bandwidth.
            latency_s, s_per_gb = 0.0, a[:, 2].sum() / a[:, 1].sum()
        rates.append(TransferRate(float(latency_s), float(1e9 / s_per_gb)))
        logger.info(
            "KV offloading %ss from %s: %.1f us + %.2f GB/s (%d steps)",
            direction,
            path,
            rates[-1].latency_s * 1e6,
            rates[-1].bytes_per_s * 1e-9,
            len(direction_rows),
        )
    return rates[0], rates[1]


def completed_transfers(
    kv_output: KVConnectorOutput | None,
) -> dict[str, dict[str, float]] | None:
    """The KV offloading transfers that completed in a step, per direction, as
    a step log's `kv_transfers` record holds them."""
    meta = kv_output.kv_connector_worker_meta if kv_output is not None else None
    if not isinstance(meta, OffloadingWorkerMetadata):
        return None
    stats = meta.transfer_stats
    transfers = {
        direction: {"count": len(s.sizes), "bytes": s.bytes, "seconds": s.time}
        for direction, s in (("load", stats.load), ("store", stats.store))
        if s.sizes
    }
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


class SimulatedKVConnector(ActiveKVConnector):
    """The model runner's KV connector hooks, with the offloading connector's
    worker-side transfers simulated instead of registering KV caches."""

    def __init__(self, vllm_config: VllmConfig, transfers: SimulatedOffloadingWorker):
        self.transfers = transfers
        super().__init__(vllm_config, {})

    def register_kv_caches(self, kv_caches_dict: dict[str, torch.Tensor]) -> None:
        connector = self.kv_connector
        if not isinstance(connector, OffloadingConnector):
            raise ValueError(
                "Simulated execution supports KV transfer only with the "
                f"OffloadingConnector, not {type(connector).__name__}."
            )
        assert connector.connector_worker is not None
        connector.connector_worker.worker = self.transfers


def simulated_kv_connector(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    clock: Clock,
    device_free_at: Callable[[], float],
) -> SimulatedKVConnector:
    timing = vllm_config.simulation_config.kv_offload_timing
    if timing is None:
        raise ValueError(
            "Simulated KV offloading needs simulation_config.kv_offload_timing: "
            "transfer bandwidths, or the path of a step log recorded with "
            "offloading."
        )
    load, store = kv_offload_rates(timing)
    groups = kv_cache_config.kv_cache_groups
    transfers = SimulatedOffloadingWorker(
        clock,
        load,
        store,
        bytes_per_block=[
            _group_kv_bytes_per_block(groups[group_id])
            for group_id in get_offloading_group_ids(kv_cache_config)
        ],
        device_free_at=device_free_at,
    )
    return SimulatedKVConnector(vllm_config, transfers)
