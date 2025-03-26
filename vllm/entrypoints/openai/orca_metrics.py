# SPDX-License-Identifier: Apache-2.0
"""
This file contains the command line arguments for the vLLM's
OpenAI-compatible server. It is kept in a separate file for documentation
purposes.
"""

import json
from collections.abc import Mapping
from typing import Optional

from vllm.logger import init_logger
from vllm.sequence import InbandEngineStats

logger = init_logger(__name__)


def create_orca_header(metrics_format: str,
                       named_metrics: list[tuple[str, float]],
                       metadata_fields=None) -> Optional[Mapping[str, str]]:
    """
    Creates ORCA headers named 'endpoint-load-metrics' in the specified format 
    and adds custom metrics to named_metrics.
    ORCA headers format description: https://docs.google.com/document/d/1C1ybMmDKJIVlrbOLbywhu9iRYo4rilR-cT50OTtOFTs/edit?tab=t.0
    ORCA proto https://github.com/cncf/xds/blob/main/xds/data/orca/v3/orca_load_report.proto

    Parameters:
    - format (str): The format of the header ('BIN', 'TEXT', 'JSON').
    - named_metrics (List[Tuple[str, float]]): List of tuples with metric names 
    and their corresponding double values.
    - metadata_fields (list): List of additional metadata fields 
    (currently unsupported).

    Returns:
    - Optional[Mapping[str,str]]: A dictionary with header key as 
    'endpoint-load-metrics' and values as the ORCA header strings with 
    format prefix and data in  with named_metrics in.
    """

    if metadata_fields:
        logger.warning("Warning: `metadata_fields` are not supported in the"
                       "ORCA response header yet.")

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
        native_http_header = ", ".join([
            f"named_metrics.{metric_name}={value}"
            for metric_name, value in named_metrics
            if isinstance(metric_name, str) and isinstance(value, float)
        ])
        header["endpoint-load-metrics"] = f"TEXT {native_http_header}"

    # output example:
    # endpoint-load-metrics: JSON “named_metrics”: {“custom-metric-util”: 0.4}
    elif metrics_format.lower() == "json":
        header["endpoint-load-metrics"] = f"JSON {json.dumps(orca_report)}"

    return header


def metrics_header(m: Optional[InbandEngineStats],
                   metrics_format: str) -> Optional[Mapping[str, str]]:
    if not m or not metrics_format:
        return None
    named_metrics: list[tuple[str, float]] = []
    for metric, val in vars(m).items():
        if isinstance(val, float) and metric != "now":
            named_metrics.append((str(metric), float(val)))
    return create_orca_header(metrics_format, named_metrics)
