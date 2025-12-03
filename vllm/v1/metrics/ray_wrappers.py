# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorPrometheus
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.spec_decode.metrics import SpecDecodingProm

try:
    from ray import serve as ray_serve
    from ray.util import metrics as ray_metrics
    from ray.util.metrics import Metric
except ImportError:
    ray_metrics = None
    ray_serve = None
import regex as re


def _get_replica_id() -> str | None:
    """Get the current Ray Serve replica ID, or None if not in a Serve context."""
    if ray_serve is None:
        return None
    try:
        return ray_serve.get_replica_context().replica_id.unique_id
    except ray_serve.exceptions.RayServeException:
        return None


class RayPrometheusMetric:
    def __init__(self):
        if ray_metrics is None:
            raise ImportError("RayPrometheusMetric requires Ray to be installed.")
        self.metric: Metric = None

    @staticmethod
    def _get_tag_keys(labelnames: list[str] | None) -> tuple[str, ...]:
        labels = list(labelnames) if labelnames else []
        labels.append("ReplicaId")
        return tuple(labels)

    def labels(self, *labels, **labelskwargs):
        if labels:
            # -1 because ReplicaId was added automatically
            expected = len(self.metric._tag_keys) - 1
            if len(labels) != expected:
                raise ValueError(
                    "Number of labels must match the number of tag keys. "
                    f"Expected {expected}, got {len(labels)}"
                )
            labelskwargs.update(zip(self.metric._tag_keys, labels))

        labelskwargs["ReplicaId"] = _get_replica_id() or ""

        if labelskwargs:
            for k, v in labelskwargs.items():
                if not isinstance(v, str):
                    labelskwargs[k] = str(v)
            self.metric.set_default_tags(labelskwargs)
        return self

    @staticmethod
    def _get_sanitized_opentelemetry_name(name: str) -> str:
        """
        For compatibility with Ray + OpenTelemetry, the metric name must be
        sanitized. In particular, this replaces disallowed character (e.g., ':')
        with '_' in the metric name.
        Allowed characters: a-z, A-Z, 0-9, _

        # ruff: noqa: E501
        Ref: https://github.com/open-telemetry/opentelemetry-cpp/blob/main/sdk/src/metrics/instrument_metadata_validator.cc#L22-L23
        Ref: https://github.com/ray-project/ray/blob/master/src/ray/stats/metric.cc#L107
        """

        return re.sub(r"[^a-zA-Z0-9_]", "_", name)


class RayGaugeWrapper(RayPrometheusMetric):
    """Wraps around ray.util.metrics.Gauge to provide same API as
    prometheus_client.Gauge"""

    def __init__(
        self,
        name: str,
        documentation: str | None = "",
        labelnames: list[str] | None = None,
        multiprocess_mode: str | None = "",
    ):
        # All Ray metrics are keyed by WorkerId, so multiprocess modes like
        # "mostrecent", "all", "sum" do not apply. This logic can be manually
        # implemented at the observability layer (Prometheus/Grafana).
        del multiprocess_mode

        tag_keys = self._get_tag_keys(labelnames)
        name = self._get_sanitized_opentelemetry_name(name)

        self.metric = ray_metrics.Gauge(
            name=name,
            description=documentation,
            tag_keys=tag_keys,
        )

    def set(self, value: int | float):
        return self.metric.set(value)

    def set_to_current_time(self):
        # ray metrics doesn't have set_to_current time, https://docs.ray.io/en/latest/_modules/ray/util/metrics.html
        return self.metric.set(time.time())


class RayCounterWrapper(RayPrometheusMetric):
    """Wraps around ray.util.metrics.Counter to provide same API as
    prometheus_client.Counter"""

    def __init__(
        self,
        name: str,
        documentation: str | None = "",
        labelnames: list[str] | None = None,
    ):
        tag_keys = self._get_tag_keys(labelnames)
        name = self._get_sanitized_opentelemetry_name(name)
        self.metric = ray_metrics.Counter(
            name=name,
            description=documentation,
            tag_keys=tag_keys,
        )

    def inc(self, value: int | float = 1.0):
        if value == 0:
            return
        return self.metric.inc(value)


class RayHistogramWrapper(RayPrometheusMetric):
    """Wraps around ray.util.metrics.Histogram to provide same API as
    prometheus_client.Histogram"""

    def __init__(
        self,
        name: str,
        documentation: str | None = "",
        labelnames: list[str] | None = None,
        buckets: list[float] | None = None,
    ):
        tag_keys = self._get_tag_keys(labelnames)
        name = self._get_sanitized_opentelemetry_name(name)

        boundaries = buckets if buckets else []
        self.metric = ray_metrics.Histogram(
            name=name,
            description=documentation,
            tag_keys=tag_keys,
            boundaries=boundaries,
        )

    def observe(self, value: int | float):
        return self.metric.observe(value)


class RaySpecDecodingProm(SpecDecodingProm):
    """
    RaySpecDecodingProm is used by RayMetrics to log to Ray metrics.
    Provides the same metrics as SpecDecodingProm but uses Ray's
    util.metrics library.
    """

    _counter_cls = RayCounterWrapper


class RayKVConnectorPrometheus(KVConnectorPrometheus):
    """
    RayKVConnectorPrometheus is used by RayMetrics to log Ray
    metrics. Provides the same metrics as KV connectors but
    uses Ray's util.metrics library.
    """

    _gauge_cls = RayGaugeWrapper
    _counter_cls = RayCounterWrapper
    _histogram_cls = RayHistogramWrapper


class RayPrometheusStatLogger(PrometheusStatLogger):
    """RayPrometheusStatLogger uses Ray metrics instead."""

    _gauge_cls = RayGaugeWrapper
    _counter_cls = RayCounterWrapper
    _histogram_cls = RayHistogramWrapper
    _spec_decoding_cls = RaySpecDecodingProm
    _kv_connector_cls = RayKVConnectorPrometheus

    @staticmethod
    def _unregister_vllm_metrics():
        # No-op on purpose
        pass
