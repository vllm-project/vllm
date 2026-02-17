# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from functools import cached_property
from typing import Any, Literal, cast

from packaging.version import parse
from pydantic import Field, field_validator, model_validator

from vllm import version
from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

DetailedTraceModules = Literal["model", "worker", "all"]


@config
class ObservabilityConfig:
    """Configuration for observability - metrics and tracing."""

    show_hidden_metrics_for_version: str | None = None
    """Enable deprecated Prometheus metrics that have been hidden since the
    specified version. For example, if a previously deprecated metric has been
    hidden since the v0.7.0 release, you use
    `--show-hidden-metrics-for-version=0.7` as a temporary escape hatch while
    you migrate to new metrics. The metric is likely to be removed completely
    in an upcoming release."""

    @cached_property
    def show_hidden_metrics(self) -> bool:
        """Check if the hidden metrics should be shown."""
        if self.show_hidden_metrics_for_version is None:
            return False
        return version._prev_minor_version_was(self.show_hidden_metrics_for_version)

    otlp_traces_endpoint: str | None = None
    """Target URL to which OpenTelemetry traces will be sent."""

    collect_detailed_traces: list[DetailedTraceModules] | None = None
    """It makes sense to set this only if `--otlp-traces-endpoint` is set. If
    set, it will collect detailed traces for the specified modules. This
    involves use of possibly costly and or blocking operations and hence might
    have a performance impact.

    Note that collecting detailed timing information for each request can be
    expensive."""

    kv_cache_metrics: bool = False
    """Enable KV cache residency metrics (lifetime, idle time, reuse gaps).
    Uses sampling to minimize overhead.
    Requires log stats to be enabled (i.e., --disable-log-stats not set)."""

    kv_cache_metrics_sample: float = Field(default=0.01, gt=0, le=1)
    """Sampling rate for KV cache metrics (0.0, 1.0]. Default 0.01 = 1% of blocks."""

    cudagraph_metrics: bool = False
    """Enable CUDA graph metrics (number of padded/unpadded tokens, runtime cudagraph
    dispatch modes, and their observed frequencies at every logging interval)."""

    enable_layerwise_nvtx_tracing: bool = False
    """Enable layerwise NVTX tracing. This traces the execution of each layer or
    module in the model and attach informations such as input/output shapes to
    nvtx range markers. Noted that this doesn't work with CUDA graphs enabled."""

    enable_mfu_metrics: bool = False
    """Enable Model FLOPs Utilization (MFU) metrics."""

    enable_mm_processor_stats: bool = False
    """Enable collection of timing statistics for multimodal processor operations.
    This is for internal use only (e.g., benchmarks) and is not exposed as a CLI
    argument."""

    enable_logging_iteration_details: bool = False
    """Enable detailed logging of iteration details.
    If set, vllm EngineCore will log iteration details
    This includes number of context/generation requests and tokens
    and the elapsed cpu time for the iteration."""

    histogram_buckets: str | None = None
    """JSON mapping of Prometheus histogram metric name patterns to custom
    bucket boundaries. Keys are metric name substrings (matched against the
    full metric name), and values are lists of numeric bucket boundaries.
    Example: '{"time_to_first_token": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    "e2e_request_latency": [0.5, 1.0, 5.0, 10.0, 30.0]}'.
    Unmatched metrics use the built-in default buckets."""

    @cached_property
    def parsed_histogram_buckets(self) -> dict[str, list[float]]:
        """Parse the histogram_buckets JSON string into a dict."""
        if self.histogram_buckets is None:
            return {}
        return json.loads(self.histogram_buckets)

    def get_histogram_buckets(
        self, metric_name: str, default: list[float | int]
    ) -> list[float | int]:
        """Return custom buckets for a metric if configured, else default.

        When multiple patterns match, the longest (most specific) pattern wins.
        """
        best_match: list[float | int] | None = None
        best_len = -1
        for pattern, buckets in self.parsed_histogram_buckets.items():
            if pattern in metric_name and len(pattern) > best_len:
                best_len = len(pattern)
                best_match = buckets
        return best_match if best_match is not None else default

    @cached_property
    def collect_model_forward_time(self) -> bool:
        """Whether to collect model forward time for the request."""
        return self.collect_detailed_traces is not None and (
            "model" in self.collect_detailed_traces
            or "all" in self.collect_detailed_traces
        )

    @cached_property
    def collect_model_execute_time(self) -> bool:
        """Whether to collect model execute time for the request."""
        return self.collect_detailed_traces is not None and (
            "worker" in self.collect_detailed_traces
            or "all" in self.collect_detailed_traces
        )

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @field_validator("show_hidden_metrics_for_version")
    @classmethod
    def _validate_show_hidden_metrics_for_version(cls, value: str | None) -> str | None:
        if value is not None:
            # Raises an exception if the string is not a valid version.
            parse(value)
        return value

    @field_validator("otlp_traces_endpoint")
    @classmethod
    def _validate_otlp_traces_endpoint(cls, value: str | None) -> str | None:
        if value is not None:
            from vllm.tracing import is_tracing_available, otel_import_error_traceback

            if not is_tracing_available():
                raise ValueError(
                    "OpenTelemetry is not available. Unable to configure "
                    "'otlp_traces_endpoint'. Ensure OpenTelemetry packages are "
                    f"installed. Original error:\n{otel_import_error_traceback}"
                )
        return value

    @field_validator("collect_detailed_traces")
    @classmethod
    def _validate_collect_detailed_traces(
        cls, value: list[DetailedTraceModules] | None
    ) -> list[DetailedTraceModules] | None:
        """Handle the legacy case where users might provide a comma-separated
        string instead of a list of strings."""
        if value is not None and len(value) == 1 and "," in value[0]:
            value = cast(list[DetailedTraceModules], value[0].split(","))
        return value

    @field_validator("histogram_buckets")
    @classmethod
    def _validate_histogram_buckets(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"histogram_buckets must be valid JSON. Error: {e}"
                ) from e
            if not isinstance(parsed, dict):
                raise ValueError(
                    "histogram_buckets must be a JSON object mapping metric "
                    "name patterns to lists of numeric bucket boundaries."
                )
            for key, buckets in parsed.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"histogram_buckets keys must be strings, got {type(key)}"
                    )
                if not isinstance(buckets, list) or not all(
                    isinstance(b, int | float) for b in buckets
                ):
                    raise ValueError(
                        f"histogram_buckets['{key}'] must be a list of numbers, "
                        f"got {buckets}"
                    )
                if buckets != sorted(buckets):
                    raise ValueError(
                        f"histogram_buckets['{key}'] must be sorted in ascending order."
                    )
        return value

    @model_validator(mode="after")
    def _validate_tracing_config(self):
        if self.collect_detailed_traces and not self.otlp_traces_endpoint:
            raise ValueError(
                "collect_detailed_traces requires `--otlp-traces-endpoint` to be set."
            )
        return self
