# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorPrometheus
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.spec_decode.metrics import SpecDecodingProm

try:
    from ray.util import metrics as ray_metrics
    from ray.util.metrics import Metric
except ImportError:
    ray_metrics = None
import regex as re


class RayPrometheusMetric:
    def __init__(self):
        if ray_metrics is None:
            raise ImportError("RayPrometheusMetric requires Ray to be installed.")

        self.metric: Metric = None

    def labels(self, *labels, **labelskwargs):
        if labelskwargs:
            for k, v in labelskwargs.items():
                if not isinstance(v, str):
                    labelskwargs[k] = str(v)

            self.metric.set_default_tags(labelskwargs)

        if labels:
            if len(labels) != len(self.metric._tag_keys):
                raise ValueError(
                    "Number of labels must match the number of tag keys. "
                    f"Expected {len(self.metric._tag_keys)}, got {len(labels)}"
                )

            self.metric.set_default_tags(dict(zip(self.metric._tag_keys, labels)))

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
        labelnames_tuple = tuple(labelnames) if labelnames else None
        name = self._get_sanitized_opentelemetry_name(name)
        self.metric = ray_metrics.Gauge(
            name=name, description=documentation, tag_keys=labelnames_tuple
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
        labelnames_tuple = tuple(labelnames) if labelnames else None
        name = self._get_sanitized_opentelemetry_name(name)
        self.metric = ray_metrics.Counter(
            name=name, description=documentation, tag_keys=labelnames_tuple
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
        labelnames_tuple = tuple(labelnames) if labelnames else None
        name = self._get_sanitized_opentelemetry_name(name)
        boundaries = buckets if buckets else []
        self.metric = ray_metrics.Histogram(
            name=name,
            description=documentation,
            tag_keys=labelnames_tuple,
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
