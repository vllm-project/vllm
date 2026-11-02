# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions that create ORCA endpoint load report response headers."""

import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from statistics import fmean

from vllm.logger import init_logger
from vllm.v1.metrics.reader import Gauge, get_metrics_snapshot

logger = init_logger(__name__)


def create_orca_header(
    metrics_format: str, named_metrics: list[tuple[str, float]]
) -> Mapping[str, str] | None:
    """Creates ORCA headers named 'endpoint-load-metrics' in the specified format
    and adds custom metrics to named_metrics.
    ORCA headers format description: https://docs.google.com/document/d/1C1ybMmDKJIVlrbOLbywhu9iRYo4rilR-cT50OTtOFTs/edit?tab=t.0
    ORCA proto https://github.com/cncf/xds/blob/main/xds/data/orca/v3/orca_load_report.proto

    Args:
        metrics_format (str): The format of the header ('TEXT', 'JSON').
        named_metrics (List[Tuple[str, float]]): List of tuples with metric names
            and their corresponding double values.

    Returns:
    - Optional[Mapping[str,str]]: A dictionary with header key as
    'endpoint-load-metrics' and values as the ORCA header strings with
    format prefix and data in  with named_metrics in.

    """
    if metrics_format.lower() not in ["text", "json"]:
        logger.warning(
            "%r format is not supported in the ORCA response header",
            metrics_format,
        )
        return None

    header = {}
    orca_report = {
        "named_metrics": {
            metric_name: value
            for metric_name, value in named_metrics
            if isinstance(metric_name, str) and isinstance(value, float)
        }
    }
    if metrics_format.lower() == "text":
        native_http_header = ", ".join(
            [
                f"named_metrics.{metric_name}={value}"
                for metric_name, value in named_metrics
                if isinstance(metric_name, str) and isinstance(value, float)
            ]
        )
        header["endpoint-load-metrics"] = f"TEXT {native_http_header}"
    elif metrics_format.lower() == "json":
        header["endpoint-load-metrics"] = f"JSON {json.dumps(orca_report)}"

    logger.debug("Created ORCA header %s", header)

    return header


def get_named_metrics_from_prometheus() -> list[tuple[str, float]]:
    """Collects current metrics from Prometheus and returns some of them
    in the form of the `named_metrics` list for `create_orca_header()`.

    With data parallelism each gauge has one sample per engine. Waiting
    requests are summed and KV cache usage is averaged across engines, so the
    header reports the load of the whole endpoint.

    Returns:
        List of tuples of metric names and their values.

    """
    prometheus_to_orca_metrics: dict[str, tuple[str, Callable[..., float]]] = {
        "vllm:kv_cache_usage_perc": ("kv_cache_usage_perc", fmean),
        "vllm:num_requests_waiting": ("num_requests_waiting", sum),
    }
    samples: dict[str, list[float]] = defaultdict(list)
    for metric in get_metrics_snapshot():
        if metric.name in prometheus_to_orca_metrics and isinstance(metric, Gauge):
            samples[metric.name].append(float(metric.value))
    named_metrics = []
    for name, values in samples.items():
        orca_name, combine = prometheus_to_orca_metrics[name]
        named_metrics.append((orca_name, combine(values)))
    return named_metrics


def metrics_header(metrics_format: str) -> Mapping[str, str] | None:
    """Creates ORCA headers named 'endpoint-load-metrics' in the specified format.
    Metrics are collected from Prometheus using `get_named_metrics_from_prometheus()`.

    ORCA headers format description: https://docs.google.com/document/d/1C1ybMmDKJIVlrbOLbywhu9iRYo4rilR-cT50OTtOFTs/edit?tab=t.0
    ORCA proto https://github.com/cncf/xds/blob/main/xds/data/orca/v3/orca_load_report.proto

    Args:
        metrics_format (str): The format of the header ('TEXT', 'JSON').

    Returns:
    - Optional[Mapping[str,str]]: A dictionary with header key as
    'endpoint-load-metrics' and values as the ORCA header strings with
    format prefix and data in  with named_metrics in.

    """
    if not metrics_format:
        return None
    named_metrics = get_named_metrics_from_prometheus()
    return create_orca_header(metrics_format, named_metrics)
