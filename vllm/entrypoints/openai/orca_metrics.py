# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utility functions that create ORCA endpoint load report response headers.
"""

import json
from collections.abc import Mapping

from vllm.logger import init_logger
from vllm.v1.metrics.reader import Gauge, get_metrics_snapshot

logger = init_logger(__name__)


def create_orca_header(
    metrics_format: str, named_metrics: list[tuple[str, float]]
) -> Mapping[str, str] | None:
    """
    Creates ORCA headers named 'endpoint-load-metrics' in the specified format
    and adds custom metrics to named_metrics.
    ORCA headers format description: https://docs.google.com/document/d/1C1ybMmDKJIVlrbOLbywhu9iRYo4rilR-cT50OTtOFTs/edit?tab=t.0
    ORCA proto https://github.com/cncf/xds/blob/main/xds/data/orca/v3/orca_load_report.proto

    Parameters:
    - metrics_format (str): The format of the header ('TEXT', 'JSON').
    - named_metrics (List[Tuple[str, float]]): List of tuples with metric names
    and their corresponding double values.

    Returns:
    - Optional[Mapping[str,str]]: A dictionary with header key as
    'endpoint-load-metrics' and values as the ORCA header strings with
    format prefix and data in  with named_metrics in.
    """

    if metrics_format.lower() not in ["text", "json"]:
        logger.warning(
            "Warning: `%s` format is not supported in the ORCA response header",
            format,
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
    # output example:
    # endpoint-load-metrics: TEXT named_metrics.kv_cache_utilization=0.4
    if metrics_format.lower() == "text":
        native_http_header = ", ".join(
            [
                f"named_metrics.{metric_name}={value}"
                for metric_name, value in named_metrics
                if isinstance(metric_name, str) and isinstance(value, float)
            ]
        )
        header["endpoint-load-metrics"] = f"TEXT {native_http_header}"

    # output example:
    # endpoint-load-metrics: JSON “named_metrics”: {“custom-metric-util”: 0.4}
    elif metrics_format.lower() == "json":
        header["endpoint-load-metrics"] = f"JSON {json.dumps(orca_report)}"

    logger.info("Created ORCA header %s", header)

    return header


def get_named_metrics_from_prometheus() -> list[tuple[str, float]]:
    """
    Collects current metrics from Prometheus and returns some of them
    in the form of the `named_metrics` list for `create_orca_header()`.

    Parameters:
    - None

    Returns:
    - list[tuple[str, float]]: List of tuples of metric names and their values.
    """
    named_metrics: list[tuple[str, float]] = []
    # Map from prometheus metric names to ORCA named metrics.
    prometheus_to_orca_metrics = {
        "vllm_kv_cache_usage_perc": "kv_cache_usage_perc",
        "vllm_num_requests_waiting": "num_requests_waiting",
    }
    metrics = get_metrics_snapshot()
    for metric in metrics:
        orca_name = prometheus_to_orca_metrics.get(metric.name)
        # If this metric is mapped into ORCA, then add it to the report.
        # Note: Only Gauge metrics are currently supported.
        if orca_name is not None and isinstance(metric, Gauge):
            named_metrics.append((str(orca_name), float(metric.value)))
    return named_metrics


def metrics_header(metrics_format: str) -> Mapping[str, str] | None:
    """
    Creates ORCA headers named 'endpoint-load-metrics' in the specified format.
    Metrics are collected from Prometheus using `get_named_metrics_from_prometheus()`.

    ORCA headers format description: https://docs.google.com/document/d/1C1ybMmDKJIVlrbOLbywhu9iRYo4rilR-cT50OTtOFTs/edit?tab=t.0
    ORCA proto https://github.com/cncf/xds/blob/main/xds/data/orca/v3/orca_load_report.proto

    Parameters:
    - metrics_format (str): The format of the header ('TEXT', 'JSON').

    Returns:
    - Optional[Mapping[str,str]]: A dictionary with header key as
    'endpoint-load-metrics' and values as the ORCA header strings with
    format prefix and data in  with named_metrics in.
    """
    if not metrics_format:
        return None
    # Get named metrics from prometheus.
    named_metrics = get_named_metrics_from_prometheus()
    return create_orca_header(metrics_format, named_metrics)
