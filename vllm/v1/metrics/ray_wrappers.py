# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ray-specific implementations of metrics loggers.

This module provides Ray-compatible wrappers for vLLM's metrics logging
infrastructure, using Ray's util.metrics library instead of Prometheus directly.
"""

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorPrometheus
from vllm.v1.metrics.backends.ray_backend import RayBackend
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.perf import PerfMetricsProm
from vllm.v1.spec_decode.metrics import SpecDecodingProm


class RaySpecDecodingProm(SpecDecodingProm):
    """
    RaySpecDecodingProm is used by RayMetrics to log to Ray metrics.
    Provides the same metrics as SpecDecodingProm but uses Ray's
    util.metrics library.
    """

    def __init__(
        self,
        speculative_config,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[str]],
        backend=None,
    ):
        # Use RayBackend if no backend is provided
        backend = backend if backend is not None else RayBackend()
        super().__init__(
            speculative_config, labelnames, per_engine_labelvalues, backend=backend
        )


class RayKVConnectorPrometheus(KVConnectorPrometheus):
    """
    RayKVConnectorPrometheus is used by RayMetrics to log Ray
    metrics. Provides the same metrics as KV connectors but
    uses Ray's util.metrics library.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[str]],
        backend=None,
    ):
        # Use RayBackend if no backend is provided
        backend = backend if backend is not None else RayBackend()
        super().__init__(
            vllm_config, labelnames, per_engine_labelvalues, backend=backend
        )


class RayPerfMetricsProm(PerfMetricsProm):
    """
    RayPerfMetricsProm is used by RayMetrics to log Ray
    metrics. Provides the same MFU metrics as PerfMetricsProm
    uses Ray's util.metrics library.
    """


class RayPrometheusStatLogger(PrometheusStatLogger):
    """RayPrometheusStatLogger uses Ray metrics instead of Prometheus.

    This logger uses a RayBackend to create Ray metrics instead of
    Prometheus metrics, enabling integration with Ray's metrics system.
    """

    _spec_decoding_cls = RaySpecDecodingProm
    _kv_connector_cls = RayKVConnectorPrometheus
    _perf_metrics_cls = RayPerfMetricsProm

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_indexes: list[int] | None = None,
        backend=None,
    ):
        # Use RayBackend if no backend is provided
        backend = backend if backend is not None else RayBackend()
        super().__init__(vllm_config, engine_indexes, backend=backend)

    @staticmethod
    def _unregister_vllm_metrics():
        # No-op on purpose - Ray handles its own metric lifecycle
        pass
