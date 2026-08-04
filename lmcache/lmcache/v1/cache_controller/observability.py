# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum
from typing import Optional

# Third Party
from prometheus_client import REGISTRY
import prometheus_client

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class SocketType(Enum):
    """Enum for socket types to ensure type safety."""

    PULL = "pull"
    REPLY = "reply"


class PrometheusLogger:
    """
    Prometheus logger for cache controller metrics.
    Provides dynamic metrics for monitoring KV pool and worker registration.
    """

    _instance = None
    _gauge_cls = prometheus_client.Gauge

    def __init__(self, labels: dict):
        self.labels = labels
        labelnames = list(self.labels.keys())

        # Dynamic metrics for cache controller
        self._init_dynamic_metrics(labelnames)

    def _init_dynamic_metrics(self, labelnames):
        """
        Initialize dynamic metrics that will be updated by lambda functions.
        """
        # KV Pool metrics
        self.kv_pool_keys_count = self._gauge_cls(
            name="lmcache:cache_controller_kv_pool_keys_count",
            documentation="The number of keys in the KV pool",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)
        # Registration Controller metrics
        self.registered_workers_count = self._gauge_cls(
            name="lmcache:cache_controller_registered_workers_count",
            documentation="The total number of registered workers",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)
        # Socket message count metrics
        self.pull_socket_message_count = self._gauge_cls(
            name="lmcache:cache_controller_pull_socket_message_count",
            documentation="The total number of messages received on PULL socket",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)
        self.reply_socket_message_count = self._gauge_cls(
            name="lmcache:cache_controller_reply_socket_message_count",
            documentation="The total number of messages received on REPLY socket",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)

        # Socket queue/backlog metrics
        self.pull_socket_has_pending = self._gauge_cls(
            name="lmcache:cache_controller_pull_socket_has_pending",
            documentation="Whether PULL socket has pending messages (1=yes, 0=no)",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)
        self.reply_socket_has_pending = self._gauge_cls(
            name="lmcache:cache_controller_reply_socket_has_pending",
            documentation="Whether REPLY socket has pending messages (1=yes, 0=no)",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)

        # Active request metrics
        self.pull_socket_active_requests = self._gauge_cls(
            name="lmcache:cache_controller_pull_socket_active_requests",
            documentation="Number of requests being processed from PULL socket",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)
        self.reply_socket_active_requests = self._gauge_cls(
            name="lmcache:cache_controller_reply_socket_active_requests",
            documentation="Number of requests being processed from REPLY socket",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)

        # Sequence number discontinuity metrics
        self.kv_op_seq_discontinuity_count = self._gauge_cls(
            name="lmcache:cache_controller_kv_op_seq_discontinuity_count",
            documentation="Total count of KV operation sequence number discontinuities",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)

        # Full sync metrics
        self.full_sync_workers_syncing = self._gauge_cls(
            name="lmcache:cache_controller_full_sync_workers_syncing",
            documentation="Number of workers currently in full sync",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)
        self.full_sync_workers_completed = self._gauge_cls(
            name="lmcache:cache_controller_full_sync_workers_completed",
            documentation="Number of workers that have completed full sync",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)
        self.full_sync_global_progress = self._gauge_cls(
            name="lmcache:cache_controller_full_sync_global_progress",
            documentation="Global full sync progress (0.0 to 1.0)",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)
        self.full_sync_missing_batches_total = self._gauge_cls(
            name="lmcache:cache_controller_full_sync_missing_batches_total",
            documentation="Total count of missing batches across all syncing workers",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)

    @staticmethod
    def GetOrCreate(
        labels: dict,
    ) -> "PrometheusLogger":
        if PrometheusLogger._instance is None:
            PrometheusLogger._instance = PrometheusLogger(labels)
        if PrometheusLogger._instance.labels != labels:
            logger.error(
                "CacheControllerPrometheusLogger instance already created with "
                "different metadata. This should not happen except in test"
            )
        return PrometheusLogger._instance

    @staticmethod
    def GetInstance() -> "PrometheusLogger":
        assert PrometheusLogger._instance is not None, (
            "CacheControllerPrometheusLogger instance not created yet"
        )
        return PrometheusLogger._instance

    @staticmethod
    def GetInstanceOrNone() -> Optional["PrometheusLogger"]:
        """
        Returns the singleton instance of CacheControllerPrometheusLogger if it exists,
        otherwise returns None.
        """
        return PrometheusLogger._instance

    @staticmethod
    def DestroyInstance():
        PrometheusLogger._instance = None

    @staticmethod
    def unregister_all_metrics():
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass


class SocketMetricsContext:
    """Context manager for socket message counting and error handling."""

    def __init__(self, manager, socket_type: SocketType, message_count: int = 1):
        self.manager = manager
        self.socket_type = socket_type
        self.message_count = message_count
        self.counter_attr = f"{socket_type.value}_socket_message_count"
        self.active_attr = f"{socket_type.value}_socket_active_requests"

    def __enter__(self):
        setattr(
            self.manager,
            self.counter_attr,
            getattr(self.manager, self.counter_attr) + self.message_count,
        )
        setattr(
            self.manager,
            self.active_attr,
            getattr(self.manager, self.active_attr) + self.message_count,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        setattr(
            self.manager,
            self.active_attr,
            getattr(self.manager, self.active_attr) - self.message_count,
        )
        if exc_type is not None:
            logger.error(
                "Controller Manager error", exc_info=(exc_type, exc_val, exc_tb)
            )
        return False
