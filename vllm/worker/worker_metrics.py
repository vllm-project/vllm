# SPDX-License-Identifier: Apache-2.0
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union

import prometheus_client
from prometheus_client import Summary

from vllm.logger import init_logger
from vllm.worker.worker_metrics_types import VllmWorkerStats

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

prometheus_client.disable_created_metrics()


@dataclass
class VllmWorkerMetadata:
    """ name of the LLM model """
    model_name: str
    """ world size when running under a distributed setting """
    world_size: int
    """ worker id when running under a distributed setting """
    worker_id: int


class VllmWorkerStatsMonitor:

    def __init__(self):
        self.total_prefill_token = []

    def on_prefill(self, token_count: int):
        self.total_prefill_token.append(token_count)

    def _clear(self):
        """
        Clear all the distribution stats
        """
        self.total_prefill_token = []

    def get_stats_and_clear(self) -> VllmWorkerStats:
        """
        This function should be called with by prometheus adapter with
        a specific interval.
        The function will return the latest states between the current
        call and the previous call.
        """
        ret = VllmWorkerStats(self.total_prefill_token, )
        self._clear()
        return ret

    _instance = None

    @staticmethod
    def GetOrCreate() -> "VllmWorkerStatsMonitor":
        if VllmWorkerStatsMonitor._instance is None:
            VllmWorkerStatsMonitor._instance = VllmWorkerStatsMonitor()
        return VllmWorkerStatsMonitor._instance


class PrometheusLogger:

    def __init__(self, metadata: VllmWorkerMetadata):
        self.metadata = metadata

        self.labels = self._metadata_to_labels(metadata)
        labelnames = list(self.labels.keys())

        self.summary_total_prefill_token = Summary(
            name="vllm_worker:total_prefill_token",
            documentation="Total info of actual prefill token in worker",
            labelnames=labelnames,
        )

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        gauge.labels(**self.labels).set(data)

    def _log_summary(self, summary: Summary, data: Union[List[int],
                                                         List[float]]) -> None:
        for value in data:
            summary.labels(**self.labels).observe(value)

    def log_prometheus(self, stats: VllmWorkerStats):
        self._log_summary(self.summary_total_prefill_token,
                          stats.summary_total_prefill_token)

    @staticmethod
    def _metadata_to_labels(metadata: VllmWorkerMetadata):
        return {
            "model_name": metadata.model_name,
            "worker_id": metadata.worker_id
        }

    _instance = None

    @staticmethod
    def GetOrCreate(metadata: VllmWorkerMetadata) -> "PrometheusLogger":
        if PrometheusLogger._instance is None:
            PrometheusLogger._instance = PrometheusLogger(metadata)
            logger.info("PrometheusLogger created for %s", metadata)
        if PrometheusLogger._instance.metadata != metadata:
            logger.error("PrometheusLogger instance already created with"
                         "different metadata. This should not happen except "
                         "in test")
        return PrometheusLogger._instance


class VllmWorkerStatsLogger:

    def __init__(self, metadata: VllmWorkerMetadata, log_interval: int):
        self.metadata = metadata
        self.log_interval = log_interval
        self.monitor = VllmWorkerStatsMonitor.GetOrCreate()
        self.prometheus_logger = PrometheusLogger.GetOrCreate(metadata)
        self.is_running = True

        self.thread = threading.Thread(target=self.log_worker, daemon=True)
        self.thread.start()

    def log_worker(self):
        while self.is_running:
            stats = self.monitor.get_stats_and_clear()
            self.prometheus_logger.log_prometheus(stats)
            time.sleep(self.log_interval)

    def shutdown(self):
        self.is_running = False
        self.thread.join()
