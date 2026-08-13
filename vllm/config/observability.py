# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from functools import cached_property
from itertools import pairwise
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

    custom_histogram_buckets: dict[str, list[float]] | None = None
    """Custom Prometheus histogram bucket boundaries, as a JSON mapping from
    bucket-family key to a list of strictly increasing, positive, finite
    upper bounds. When a family key is present, its list replaces the default
    buckets for every histogram in that family; families not listed keep
    their defaults. Known families: `request_latency`, `time_to_first_token`,
    `inter_token_latency`, `iteration_tokens`, `request_params_n`,
    `request_tokens`, `kv_cache_residency`. Example:
    `--custom-histogram-buckets '{"request_latency": [0.01, 0.05, 0.1, 0.5]}'`.
    Note that every extra bucket adds one time series per metric and label
    set."""

    cudagraph_metrics: bool = False
    """Enable CUDA graph metrics (number of padded/unpadded tokens, runtime cudagraph
    dispatch modes, and their observed frequencies at every logging interval)."""

    enable_layerwise_nvtx_tracing: bool = False
    """Enable layerwise NVTX tracing. This traces the execution of each layer or
    module in the model and attach information such as input/output shapes to
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

    jit_monitor_mode: Literal["warn", "error"] = "warn"
    """How to handle post-warmup JIT compilation events."""

    jit_monitor_verbose: bool = False
    """Log every monitored JIT compile with runtime details. This can emit many
    logs and add overhead, so it is intended for debugging."""

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

    @field_validator("custom_histogram_buckets", mode="before")
    @classmethod
    def _reject_bool_histogram_bounds(cls, value: object) -> object:
        """Reject booleans before pydantic silently coerces them to floats."""
        if isinstance(value, dict):
            for family, buckets in value.items():
                if not isinstance(buckets, list):
                    continue
                for bound in buckets:
                    if isinstance(bound, bool):
                        raise ValueError(
                            f"custom_histogram_buckets[{family!r}]: bound "
                            f"{bound!r} must be a number, not a boolean"
                        )
        return value

    @field_validator("custom_histogram_buckets")
    @classmethod
    def _validate_custom_histogram_buckets(
        cls, value: dict[str, list[float]] | None
    ) -> dict[str, list[float]] | None:
        if value is None:
            return value
        from vllm.v1.metrics.buckets import BUCKET_FAMILY_KEYS

        for family, buckets in value.items():
            if family not in BUCKET_FAMILY_KEYS:
                raise ValueError(
                    f"custom_histogram_buckets: unknown bucket family "
                    f"{family!r}; known families: {sorted(BUCKET_FAMILY_KEYS)}"
                )
            if not buckets:
                raise ValueError(
                    f"custom_histogram_buckets[{family!r}]: bucket list "
                    "must not be empty"
                )
            for bound in buckets:
                if not math.isfinite(bound) or bound <= 0:
                    raise ValueError(
                        f"custom_histogram_buckets[{family!r}]: bound "
                        f"{bound!r} must be finite and greater than 0"
                    )
            if any(a >= b for a, b in pairwise(buckets)):
                raise ValueError(
                    f"custom_histogram_buckets[{family!r}]: bounds {buckets} "
                    "must be strictly increasing"
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

    @model_validator(mode="after")
    def _validate_tracing_config(self):
        if self.collect_detailed_traces and not self.otlp_traces_endpoint:
            raise ValueError(
                "collect_detailed_traces requires `--otlp-traces-endpoint` to be set."
            )
        return self
