# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

from prometheus_client import REGISTRY
from prometheus_client import Metric as PromMetric
from prometheus_client.samples import Sample


@dataclass
class Metric:
    """A base class for prometheus metrics.

    Each metric may be associated with key=value labels, and
    in some cases a single vLLM instance may have multiple
    metrics with the same name but different sets of labels.
    """
    name: str
    labels: dict[str, str]


@dataclass
class Counter(Metric):
    """A monotonically increasing integer counter."""
    value: int


@dataclass
class Vector(Metric):
    """An ordered array of integer counters.

    This type - which doesn't exist in Prometheus - models one very
    specific metric, vllm:spec_decode_num_accepted_tokens_per_pos.
    """
    values: list[int]


@dataclass
class Gauge(Metric):
    """A numerical value that can go up or down."""
    value: float


@dataclass
class Histogram(Metric):
    """Observations recorded in configurable buckets.

    Buckets are represented by a dictionary. The key is
    the upper limit of the bucket, and the value is the
    observed count in that bucket. A '+Inf' key always
    exists.

    The count property is the total count across all
    buckets, identical to the count of the '+Inf' bucket.

    The sum property is the total sum of all observed
    values.
    """
    count: int
    sum: float
    buckets: dict[str, float]


def get_metrics_snapshot() -> list[Metric]:
    """An API for accessing in-memory Prometheus metrics.

    Example:
        >>> for metric in llm.get_metrics():
        ...     if isinstance(metric, Counter):
        ...         print(f"{metric} = {metric.value}")
        ...     elif isinstance(metric, Gauge):
        ...         print(f"{metric} = {metric.value}")
        ...     elif isinstance(metric, Histogram):
        ...         print(f"{metric}")
        ...         print(f"    sum = {metric.sum}")
        ...         print(f"    count = {metric.count}")
        ...         for bucket_le, value in metrics.buckets.items():
        ...             print(f"    {bucket_le} = {value}")
    """
    collected: list[Metric] = []
    for metric in REGISTRY.collect():
        if not metric.name.startswith("vllm:"):
            continue
        if metric.type == "gauge":
            sample = _must_get_sample(metric)
            collected.append(
                Gauge(name=metric.name,
                      labels=sample.labels,
                      value=sample.value))
        elif metric.type == "counter":
            samples = _get_samples(metric, "_total")
            if metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                #
                # Ugly vllm:num_accepted_tokens_per_pos special case.
                #
                # This metric is a vector of counters - for each spec
                # decoding token position, we observe the number of
                # accepted tokens using a Counter labeled with 'position'.
                # We convert these into a vector of integer values.
                #
                values: list[int] = [0] * len(samples)
                for s in samples:
                    values[int(s.labels["position"])] = int(s.value)
                collected.append(
                    Vector(name=metric.name,
                           labels=_strip_label(s.labels, "position"),
                           values=values))
                continue

            for s in samples:
                collected.append(
                    Counter(name=metric.name,
                            labels=samples[0].labels,
                            value=int(samples[0].value)))

        elif metric.type == "histogram":
            #
            # A histogram has a number of '_bucket' samples where
            # the 'le' label represents the upper limit of the bucket.
            # We convert these bucketized values into a dict of values
            # indexed by the value of the 'le' label. The 'le=+Inf'
            # label is a special case, catching all values observed.
            #
            count_sample = int(_must_get_sample(metric, "_count").value)
            sum_sample = _must_get_sample(metric, "_sum").value
            buckets: dict[str, float] = dict()
            for s in _get_samples(metric, "_bucket"):
                buckets[s.labels["le"]] = s.value
            collected.append(
                Histogram(name=metric.name,
                          labels=_strip_label(s.labels, "le"),
                          buckets=buckets,
                          count=count_sample,
                          sum=sum_sample))
        else:
            raise AssertionError(f"Unknown metric type {metric.type}")

    return collected


def _get_samples(metric: PromMetric,
                 suffix: Optional[str] = None) -> list[Sample]:
    name = (metric.name + suffix) if suffix is not None else metric.name
    return [s for s in metric.samples if s.name == name]


def _must_get_sample(metric: PromMetric,
                     suffix: Optional[str] = None) -> Sample:
    samples = _get_samples(metric, suffix)
    assert len(samples) == 1
    return samples[0]


def _strip_label(labels: dict[str, str], key_to_remove: str) -> dict[str, str]:
    labels_copy = labels.copy()
    labels_copy.pop(key_to_remove)
    return labels_copy
