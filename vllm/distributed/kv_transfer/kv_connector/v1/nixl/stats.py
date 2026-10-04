# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats and Prometheus metrics for the NIXL connector."""

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine

if TYPE_CHECKING:
    from vllm.distributed.nixl_utils import nixlXferTelemetry


@dataclass
class NixlKVConnectorStats(KVConnectorStats):
    """Container for NIXL KV cache transfer performance metrics.

    Each TP rank maintains its own ``NixlKVConnectorStats`` instance and
    independently records per-transfer telemetry via ``record_transfer()``.
    When metrics reporting is triggered, observations from **all TP ranks**
    are combined via ``aggregate()`` (simple concatenation) and summary
    statistics are computed via ``reduce()`` over the combined pool.

    **Aggregation semantics (important for multi-rank / TP > 1 deployments):**

    Because data from all ranks is concatenated into a single flat list
    before any statistics are computed, the reported metrics have the
    following semantics:

    - ``Num successful transfers``: the **total count across all ranks**,
      not the per-rank count. For example, with TP=4 where each rank
      performs 10 transfers, this reports 40.
    - ``Avg MB per transfer``: the average over all individual
      **rank-level** transfer observations. This is *not* the total
      bytes moved for a single logical KV cache transfer operation.
    - ``Throughput (MB/s)``: computed as
      ``sum(bytes_all_ranks) / sum(durations_all_ranks)``, which yields
      an **average per-rank throughput**, not the aggregate system
      throughput. Aggregate system throughput would require dividing by
      wall-clock time (``max(durations)``) instead.
    - **Percentiles (P90, etc.)**: computed over the combined distribution
      of all ranks' transfer times, not per-rank.

    This is a deliberate design choice (fire-and-forget metric collection
    from workers) but is unintuitive if one expects per-engine totals or
    aggregate system throughput.
    """

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        # Must be serializable
        """Reset all metric accumulators to empty lists.

        Called at initialization and after ``clone_and_reset()`` to prepare
        the container for a fresh reporting interval. Each key in
        ``self.data`` corresponds to a list of per-transfer observations
        recorded by this rank:

        Keys:
            transfer_duration: Per-transfer wall-clock time (seconds).
            post_duration: Per-transfer "post" / notification time (seconds).
                In NIXL, a "post" signals the remote side that data has
                arrived. This is separate from the data transfer itself.
            bytes_transferred: Bytes moved in each individual transfer.
            num_descriptors: Number of memory descriptors per transfer.
                A descriptor is a NIXL concept — it describes a contiguous
                region of GPU memory (address + size) to be transferred.
                A single KV cache transfer may involve many descriptors,
                one per memory region that makes up the cache.
            num_failed_transfers: Counts of failed transfer operations
                (each entry is 1, used as a counter).
            num_failed_notifications: Counts of failed notification
                (send_notif) operations.
            num_kv_expired_reqs: Counts of requests whose KV blocks
                expired before they could be consumed (e.g., the decode
                engine was too slow and the prefilled KV was evicted).
        """
        self.data: dict[str, list[float | int]] = {
            "transfer_duration": [],
            "post_duration": [],
            "bytes_transferred": [],
            "num_descriptors": [],
            "num_failed_transfers": [],
            "num_failed_notifications": [],
            "num_kv_expired_reqs": [],
        }

    def record_transfer(self, res: "nixlXferTelemetry"):
        # Keep metrics units consistent with rest of the code: time us->s
        """Record telemetry for a single successful NIXL transfer.

        Called once per successful KV cache transfer on this rank. The
        raw NIXL telemetry (``nixlXferTelemetry``) provides durations
        in microseconds — this method converts to seconds for consistency
        with the rest of vLLM's metrics stack.

        Args:
            res: NIXL transfer telemetry object containing:
                - ``xferDuration`` (µs): time to complete the data transfer
                - ``postDuration`` (µs): time to complete the post/notification
                  that tells the receiver "data is ready"
                - ``totalBytes`` (bytes): total bytes transferred
                - ``descCount``: number of memory descriptors involved
        """
        self.data["transfer_duration"].append(res.xferDuration / 1e6)
        self.data["post_duration"].append(res.postDuration / 1e6)
        self.data["bytes_transferred"].append(res.totalBytes)
        self.data["num_descriptors"].append(res.descCount)

    def record_failed_transfer(self):
        """Record a failed NIXL transfer operation.

        Increments the failure counter by one. Unlike successful transfer
        metrics (which carry timing/size data), failures only carry a
        count — there is no duration or byte information for a transfer
        that did not complete.
        """
        self.data["num_failed_transfers"].append(1)

    def record_failed_notification(self):
        """Record a failed NIXL notification (send_notif) operation.

        Notifications are the "post" step that signals the remote side
        that KV cache data has arrived. A notification failure means the
        data may have been transferred but the receiver was not informed.
        """
        self.data["num_failed_notifications"].append(1)

    def record_kv_expired_req(self):
        """Record a request whose KV blocks expired before consumption.

        In disaggregated serving, the prefill engine computes KV cache
        and the decode engine must consume it. If the decode engine is
        slow (e.g., under heavy load), the KV blocks on the prefill side
        may be evicted (expire) before they are read. Each such request
        is counted here. This metric is tracked on the prefill (P) instance.
        """
        self.data["num_kv_expired_reqs"].append(1)

    def clone_and_reset(self) -> "NixlKVConnectorStats":
        """Snapshot current stats and reset the container for the next interval.

        Returns a shallow copy of this stats object with the accumulated
        data, then clears this object's data to start collecting fresh
        observations. This is the mechanism that separates reporting
        intervals — the returned copy is passed to ``aggregate()`` and
        ``reduce()``, while ``self`` continues recording new transfers.

        Returns:
            A copy of this object containing all data accumulated since
            the last reset.
        """
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        # Do not discard metrics update that are entirely failures related.
        """Check whether this stats object has any data worth reporting.

        Returns True only if there are zero successful transfers AND
        zero failures of any kind. This prevents discarding a metrics
        update that contains only failure data — even if no transfers
        succeeded, failure counts are still valuable for monitoring.
        """
        return (
            self.num_successful_transfers == 0
            and len(self.data["num_failed_transfers"]) == 0
            and len(self.data["num_failed_notifications"]) == 0
            and len(self.data["num_kv_expired_reqs"]) == 0
        )

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        """Merge another rank's stats into this one by concatenating observations.

        This is the core of the cross-rank aggregation mechanism. In a
        multi-rank (TP > 1) deployment, each rank maintains its own
        ``NixlKVConnectorStats``. When it's time to report metrics,
        stats from all ranks are merged by extending each metric list
        with the other rank's corresponding list.

        After aggregation, the data is a flat pool of observations from
        all ranks — there is no longer any distinction between "rank 0's
        transfer" and "rank 3's transfer". This means downstream
        ``reduce()`` computes statistics over the combined distribution,
        which has specific implications for how metrics should be
        interpreted (see class docstring).

        Args:
            other: Another rank's stats to merge into ``self``. If
                ``other`` is empty (no data), this is a no-op.

        Returns:
            ``self``, now containing the merged data.
        """
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        # Compute compact representative stats suitable for CLI logging
        """Compute summary statistics over the aggregated observation pool.

        This method is called **after** ``aggregate()`` has combined data
        from all TP ranks. It computes compact, human-readable stats
        intended for the CLI log line:

            KV Transfer metrics: Num successful transfers=..., ...

        **Important aggregation semantics:**

        Since observations from all ranks are pooled into flat lists
        (see ``aggregate()``), the computed statistics have these meanings:

        - **Num successful transfers**: total transfer count across all
          ranks. With TP=N, this is roughly N * (per-rank count).
        - **Avg / P90 xfer time**: mean and 90th percentile of transfer
          durations over the combined pool of all ranks' transfers.
        - **Avg / P90 post time**: same, for the NIXL "post" (notification)
          step durations.
        - **Avg MB per transfer**: ``sum(bytes_all_ranks) / count_all_ranks``.
          This is the average per individual rank-level transfer, NOT the
          total bytes of a single logical KV cache operation.
        - **Throughput (MB/s)**: ``sum(bytes_all_ranks) / sum(durations_all_ranks)``.
          Because the denominator sums durations across ranks (rather than
          taking the wall-clock max), this yields an **average per-rank
          throughput**, not the aggregate system throughput.
        - **Avg number of descriptors**: mean descriptor count over all
          transfers from all ranks.

        Returns:
            A dict mapping human-readable metric names to their computed
            values. Returns all-zero dict if there are no successful
            transfers (failure-only data is reported via Prometheus
            counters instead).
        """
        if self.num_successful_transfers == 0:
            # CLI logging only reports successful transfers stats. If all requests in
            # the interval were unsuccessful, Prom will report failures stats instead.
            return {
                "Num successful transfers": 0,
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg post time (ms)": 0,
                "P90 post time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
                "Avg number of descriptors": 0,
            }

        xfer_time = np.asarray(self.data["transfer_duration"])
        post_time = np.asarray(self.data["post_duration"])
        # Convert to MB for CLI logging.
        mb = np.asarray(self.data["bytes_transferred"]) / 2**20
        descs = np.asarray(self.data["num_descriptors"], dtype=np.uint32)
        n = len(descs)
        assert n == self.num_successful_transfers

        total_mb = mb.sum()
        avg_mb = total_mb / n

        total_time_seconds = xfer_time.sum()
        throughput_mb_s = total_mb / total_time_seconds

        return {
            "Num successful transfers": n,
            "Avg xfer time (ms)": round(xfer_time.mean() * 1e3, 3),
            "P90 xfer time (ms)": round(np.percentile(xfer_time, 90).item() * 1e3, 3),
            "Avg post time (ms)": round(post_time.mean() * 1e3, 3),
            "P90 post time (ms)": round(np.percentile(post_time, 90).item() * 1e3, 3),
            "Avg MB per transfer": round(avg_mb, 3),
            "Throughput (MB/s)": round(throughput_mb_s, 3),
            "Avg number of descriptors": round(descs.mean(), 1),
        }

    @property
    def num_successful_transfers(self) -> int:
        """Number of successful transfers recorded so far.

        This counts the length of the ``transfer_duration`` list, which
        is incremented once per successful ``record_transfer()`` call.
        After ``aggregate()``, this reflects the total across all ranks.
        """
        return len(self.data["transfer_duration"])


class NixlPromMetrics(KVConnectorPromMetrics):
    """Feed aggregated transfer stats into Prometheus histograms and counters.

    This is the final step in the metrics pipeline:
    ``observe() (per-rank) → aggregate() → reduce() (CLI) → observe() (Prometheus)``

    Note the dual role of ``observe()``: it is called once per rank
    to record individual data points into ``NixlKVConnectorStats``
    (via ``record_transfer()``), and it is called here on the
    ``NixlPromMetrics`` object to feed the **already-aggregated**
    data into Prometheus metric objects for external monitoring.

    The data arrives **pre-aggregated** across all TP workers — by the
    time this method is called, the lists in ``transfer_stats_data``
    already contain observations from all ranks (merged via
    ``NixlKVConnectorStats.aggregate()``).

    For histogram metrics (transfer duration, post duration, bytes
    transferred, num descriptors), each individual observation is
    recorded as a separate histogram sample. This means Prometheus
    histograms see the same combined distribution as the CLI log
    line — all ranks' observations pooled together.

    For counter metrics (failed transfers, failed notifications,
    expired KV requests), each count is incremented accordingly.

    Args:
        transfer_stats_data: The ``data`` dict from an aggregated
            ``NixlKVConnectorStats`` object, containing lists of
            per-transfer observations from all ranks.
        engine_idx: Index of the engine instance (for multi-engine
            setups). Defaults to 0.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        buckets = [
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.2,
            0.3,
            0.5,
            0.75,
            1.0,
            5.0,
        ]
        nixl_histogram_xfer_time = self._histogram_cls(
            name="vllm:nixl_xfer_time_seconds",
            documentation="Histogram of transfer duration for NIXL KV Cache transfers.",
            buckets=buckets[1:],
            labelnames=labelnames,
        )
        self.nixl_histogram_xfer_time = create_metric_per_engine(
            nixl_histogram_xfer_time, self.per_engine_labelvalues
        )
        nixl_histogram_post_time = self._histogram_cls(
            name="vllm:nixl_post_time_seconds",
            documentation="Histogram of transfer post time for NIXL KV"
            " Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_post_time = create_metric_per_engine(
            nixl_histogram_post_time, self.per_engine_labelvalues
        )
        # uniform 2kb to 16gb range
        buckets = [2 ** (10 + i) for i in range(1, 25, 2)]
        nixl_histogram_bytes_transferred = self._histogram_cls(
            name="vllm:nixl_bytes_transferred",
            documentation="Histogram of bytes transferred per NIXL KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_bytes_transferred = create_metric_per_engine(
            nixl_histogram_bytes_transferred, self.per_engine_labelvalues
        )
        buckets = [
            10,
            20,
            30,
            50,
            75,
            100,
            200,
            400,
            1000,
            2000,
            4000,
            10000,
            20000,
            50000,
        ]
        nixl_histogram_num_descriptors = self._histogram_cls(
            name="vllm:nixl_num_descriptors",
            documentation="Histogram of number of descriptors per NIXL"
            "  KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_num_descriptors = create_metric_per_engine(
            nixl_histogram_num_descriptors, self.per_engine_labelvalues
        )
        counter_nixl_num_failed_transfers = self._counter_cls(
            name="vllm:nixl_num_failed_transfers",
            documentation="Number of failed NIXL KV Cache transfers.",
            labelnames=labelnames,
        )
        self.counter_nixl_num_failed_transfers = create_metric_per_engine(
            counter_nixl_num_failed_transfers, self.per_engine_labelvalues
        )
        counter_nixl_num_failed_notifications = self._counter_cls(
            name="vllm:nixl_num_failed_notifications",
            documentation="Number of failed NIXL KV Cache notifications.",
            labelnames=labelnames,
        )
        self.counter_nixl_num_failed_notifications = create_metric_per_engine(
            counter_nixl_num_failed_notifications, self.per_engine_labelvalues
        )

        counter_nixl_num_kv_expired_reqs = self._counter_cls(
            name="vllm:nixl_num_kv_expired_reqs",
            documentation="Number of requests that had their KV expire. "
            "NOTE: This metric is tracked on the P instance.",
            labelnames=labelnames,
        )
        self.counter_nixl_num_kv_expired_reqs = create_metric_per_engine(
            counter_nixl_num_kv_expired_reqs, self.per_engine_labelvalues
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for prom_obj, list_item_key in zip(
            [
                self.nixl_histogram_xfer_time,
                self.nixl_histogram_post_time,
                self.nixl_histogram_bytes_transferred,
                self.nixl_histogram_num_descriptors,
            ],
            [
                "transfer_duration",
                "post_duration",
                "bytes_transferred",
                "num_descriptors",
            ],
        ):
            for list_item in transfer_stats_data[list_item_key]:
                prom_obj[engine_idx].observe(list_item)
        for counter_obj, counter_item_key in zip(
            [
                self.counter_nixl_num_failed_transfers,
                self.counter_nixl_num_failed_notifications,
                self.counter_nixl_num_kv_expired_reqs,
            ],
            ["num_failed_transfers", "num_failed_notifications", "num_kv_expired_reqs"],
        ):
            for list_item in transfer_stats_data[counter_item_key]:
                counter_obj[engine_idx].inc(list_item)
