# SPDX-License-Identifier: Apache-2.0
import time
from typing import Optional, Union

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.spec_decode.metrics import SpecDecodingProm

try:
    from ray.util import metrics as ray_metrics
except ImportError:
    ray_metrics = None


class _RayGaugeWrapper:
    """Wraps around ray.util.metrics.Gauge to provide same API as
    prometheus_client.Gauge"""

    def __init__(self,
                 name: str,
                 documentation: Optional[str] = "",
                 labelnames: Optional[list[str]] = None,
                 multiprocess_mode: str = ""):
        labelnames_tuple = tuple(labelnames) if labelnames else None
        self._gauge = ray_metrics.Gauge(name=name,
                                        description=documentation,
                                        tag_keys=labelnames_tuple)

    def labels(self, **labels):
        self._gauge.set_default_tags(labels)
        return self

    def set(self, value: Union[int, float]):
        return self._gauge.set(value)

    def set_to_current_time(self):
        # ray metrics doesn't have set_to_current time, https://docs.ray.io/en/latest/_modules/ray/util/metrics.html
        return self._gauge.set(time.time())


class _RayCounterWrapper:
    """Wraps around ray.util.metrics.Counter to provide same API as
    prometheus_client.Counter"""

    def __init__(self,
                 name: str,
                 documentation: Optional[str] = "",
                 labelnames: Optional[list[str]] = None):
        labelnames_tuple = tuple(labelnames) if labelnames else None
        self._counter = ray_metrics.Counter(name=name,
                                            description=documentation,
                                            tag_keys=labelnames_tuple)

    def labels(self, **labels):
        self._counter.set_default_tags(labels)
        return self

    def inc(self, value: Union[int, float] = 1.0):
        if value == 0:
            return
        return self._counter.inc(value)


class _RayHistogramWrapper:
    """Wraps around ray.util.metrics.Histogram to provide same API as
    prometheus_client.Histogram"""

    def __init__(self,
                 name: str,
                 documentation: Optional[str] = "",
                 labelnames: Optional[list[str]] = None,
                 buckets: Optional[list[float]] = None):
        labelnames_tuple = tuple(labelnames) if labelnames else None
        boundaries = buckets if buckets else []
        self._histogram = ray_metrics.Histogram(name=name,
                                                description=documentation,
                                                tag_keys=labelnames_tuple,
                                                boundaries=boundaries)

    def labels(self, **labels):
        self._histogram.set_default_tags(labels)
        return self

    def observe(self, value: Union[int, float]):
        return self._histogram.observe(value)


class RaySpecDecodingProm(SpecDecodingProm):
    """
    RaySpecDecodingProm is used by RayMetrics to log to Ray metrics.
    Provides the same metrics as SpecDecodingProm but uses Ray's
    util.metrics library.
    """

    def _create_counter(self, name: str, documentation: Optional[str],
                        labelnames: list[str]):
        return _RayCounterWrapper(name=name,
                                  documentation=documentation,
                                  labelnames=labelnames)


class RayPrometheusStatLogger(PrometheusStatLogger):
    """RayPrometheusStatLogger uses Ray metrics instead."""

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        super().__init__(vllm_config, engine_index)

    def _create_gauge(self,
                      name: str,
                      documentation: Optional[str],
                      labelnames: list[str],
                      multiprocess_mode: str = "all"):
        return _RayGaugeWrapper(name=name,
                                documentation=documentation,
                                labelnames=labelnames,
                                multiprocess_mode=multiprocess_mode)

    def _create_counter(self, name: str, documentation: Optional[str],
                        labelnames: list[str]):
        return _RayCounterWrapper(name=name,
                                  documentation=documentation,
                                  labelnames=labelnames)

    def _create_histogram(self, name: str, documentation: Optional[str],
                          buckets: list[Union[int,
                                              float]], labelnames: list[str]):
        return _RayHistogramWrapper(
            name=name,
            documentation=documentation,
            buckets=buckets,
            labelnames=labelnames,
        )

    def _create_spec_decoding(self, config: SpeculativeConfig,
                              labelnames: list[str], labelvalues: list[str]):
        return RaySpecDecodingProm(config, labelnames, labelvalues)

    @staticmethod
    def _unregister_vllm_metrics():
        # No-op on purpose
        pass
