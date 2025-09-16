"""Utility to fetch and display metrics from the vLLM metrics proxy."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable

import httpx


def fetch_metrics(url: str) -> Dict[str, Any]:
    with httpx.Client(timeout=10.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def print_metric_groups(metrics: Dict[str, Any]) -> None:
    for gauge in metrics.get("gauges", []):
        print(_format_value_metric(gauge, "gauge"))
    for counter in metrics.get("counters", []):
        print(_format_value_metric(counter, "counter"))
    for vector in metrics.get("vectors", []):
        print(_format_vector_metric(vector))
    for histogram in metrics.get("histograms", []):
        print(_format_histogram(histogram))


def _format_value_metric(metric: Dict[str, Any], metric_type: str) -> str:
    labels = json.dumps(metric.get("labels", {}), sort_keys=True)
    value = metric.get("value")
    return f"{metric['name']} ({metric_type}) labels={labels} value={value}"


def _format_vector_metric(metric: Dict[str, Any]) -> str:
    labels = json.dumps(metric.get("labels", {}), sort_keys=True)
    values = metric.get("values", [])
    return f"{metric['name']} (vector) labels={labels} values={values}"


def _format_histogram(metric: Dict[str, Any]) -> str:
    labels = json.dumps(metric.get("labels", {}), sort_keys=True)
    buckets = metric.get("buckets", {})
    count = metric.get("count")
    total = metric.get("sum")
    bucket_lines = ", ".join(f"{le}={value}" for le, value in buckets.items())
    return (f"{metric['name']} (histogram) labels={labels} count={count} "
            f"sum={total} buckets={{ {bucket_lines} }}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print metrics from the proxy")
    parser.add_argument("--url",
                        default="http://localhost:8000/internal/metrics",
                        help="URL of the proxy metrics endpoint")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = fetch_metrics(args.url)
    print_metric_groups(metrics)


if __name__ == "__main__":
    main()
