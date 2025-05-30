# SPDX-License-Identifier: Apache-2.0
import time
from typing import Optional, Union

from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.spec_decode.metrics import SpecDecodingProm

try:
    from ray.util import metrics as ray_metrics
    from ray.util.metrics import Metric
except ImportError:
    ray_metrics = None


class RayPrometheusMetric:

    def __init__(self):
        if ray_metrics is None:
            raise ImportError(
                "RayPrometheusMetric requires Ray to be installed.")

        self.metric: Metric = None

    def labels(self, *labels, **labelskwargs):
        if labelskwargs:
            for k, v in labelskwargs.items():
                if not isinstance(v, str):
                    labelskwargs[k] = str(v)

            self.metric.set_default_tags(labelskwargs)

        return self


class RayGaugeWrapper(RayPrometheusMetric):
    """Wraps around ray.util.metrics.Gauge to provide same API as
    prometheus_client.Gauge"""

    def __init__(self,
                 name: str,
                 documentation: Optional[str] = "",
                 labelnames: Optional[list[str]] = None):
        labelnames_tuple = tuple(labelnames) if labelnames else None
        self.metric = ray_metrics.Gauge(name=name,
                                        description=documentation,
                                        tag_keys=labelnames_tuple)

    def set(self, value: Union[int, float]):
        return self.metric.set(value)

    def set_to_current_time(self):
        # ray metrics doesn't have set_to_current time, https://docs.ray.io/en/latest/_modules/ray/util/metrics.html
        return self.metric.set(time.time())


class RayCounterWrapper(RayPrometheusMetric):
    """Wraps around ray.util.metrics.Counter to provide same API as
    prometheus_client.Counter"""

    def __init__(self,
                 name: str,
                 documentation: Optional[str] = "",
                 labelnames: Optional[list[str]] = None):
        labelnames_tuple = tuple(labelnames) if labelnames else None
        self.metric = ray_metrics.Counter(name=name,
                                          description=documentation,
                                          tag_keys=labelnames_tuple)

    def inc(self, value: Union[int, float] = 1.0):
        if value == 0:
            return
        return self.metric.inc(value)


class RayHistogramWrapper(RayPrometheusMetric):
    """Wraps around ray.util.metrics.Histogram to provide same API as
    prometheus_client.Histogram"""

    def __init__(self,
                 name: str,
                 documentation: Optional[str] = "",
                 labelnames: Optional[list[str]] = None,
                 buckets: Optional[list[float]] = None):
        labelnames_tuple = tuple(labelnames) if labelnames else None
        boundaries = buckets if buckets else []
        self.metric = ray_metrics.Histogram(name=name,
                                            description=documentation,
                                            tag_keys=labelnames_tuple,
                                            boundaries=boundaries)

    def observe(self, value: Union[int, float]):
        return self.metric.observe(value)


class RaySpecDecodingProm(SpecDecodingProm):
    """
    RaySpecDecodingProm is used by RayMetrics to log to Ray metrics.
    Provides the same metrics as SpecDecodingProm but uses Ray's
    util.metrics library.
    """

    _counter_cls = RayCounterWrapper


class RayPrometheusStatLogger(PrometheusStatLogger):
    """RayPrometheusStatLogger uses Ray metrics instead."""

    _gauge_cls = RayGaugeWrapper
    _counter_cls = RayCounterWrapper
    _histogram_cls = RayHistogramWrapper
    _spec_decoding_cls = RaySpecDecodingProm

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        super().__init__(vllm_config, engine_index)

    @staticmethod
    def _unregister_vllm_metrics():
        # No-op on purpose
        pass
