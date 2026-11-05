# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Count hot-buffer traffic across periodic worker-stat counter drains."""


class HotTraffic:
    def __init__(self, connector):
        self.connector = connector
        self.original = connector.get_kv_connector_stats
        self.reported = dict(hits=0, misses=0, hot_fetch_bytes=0)
        self.snapshots = 0
        connector.get_kv_connector_stats = self.collect

    def collect(self):
        stats = self.original()
        if stats is not None:
            for target, source in (
                ("hits", "cache_hits"),
                ("misses", "cache_misses"),
                ("hot_fetch_bytes", "host_to_device_bytes"),
            ):
                self.reported[target] += sum(stats.data[source])
            self.snapshots += len(stats.data["cache_hits"])
        return stats

    def read(self):
        """Read only after synchronizing the worker's device and copy streams."""
        totals = dict(self.reported)
        for runtime in self.connector.leader_runtimes:
            group = runtime.index_group
            counters = [group.swap_stats]
            if self.connector._metrics_pending:
                counters.append(group.swap_stats_host)
            for counter in counters:
                hits, misses = counter.cpu().tolist()
                totals["hits"] += hits
                totals["misses"] += misses
                totals["hot_fetch_bytes"] += misses * group.stats_row_bytes
        return dict(
            **totals,
            reported_snapshots=self.snapshots,
            pending_snapshot=self.connector._metrics_pending,
        )


def install(worker):
    from vllm.distributed.kv_transfer.kv_transfer_state import (
        get_kv_transfer_group,
        has_kv_transfer_group,
    )

    assert not hasattr(worker, "agentic_traffic")
    worker.agentic_traffic = (
        HotTraffic(get_kv_transfer_group().connector_worker)
        if has_kv_transfer_group()
        else None
    )
    return dict(rank=worker.rank, installed=worker.agentic_traffic is not None)


def snapshot(worker):
    return worker.agentic_traffic.read() if worker.agentic_traffic is not None else None
