# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from functools import cached_property
from typing import Any, Literal, Optional, cast

from pydantic import field_validator, model_validator
from pydantic.dataclasses import dataclass

from vllm import version
from vllm.config.utils import config

DetailedTraceModules = Literal["model", "worker", "all"]


@config
@dataclass
class ObservabilityConfig:
    """Configuration for observability - metrics and tracing."""

    show_hidden_metrics_for_version: Optional[str] = None
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

    otlp_traces_endpoint: Optional[str] = None
    """Target URL to which OpenTelemetry traces will be sent."""

    collect_detailed_traces: Optional[list[DetailedTraceModules]] = None
    """It makes sense to set this only if `--otlp-traces-endpoint` is set. If
    set, it will collect detailed traces for the specified modules. This
    involves use of possibly costly and or blocking operations and hence might
    have a performance impact.

    Note that collecting detailed timing information for each request can be
    expensive."""

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
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @field_validator("show_hidden_metrics_for_version", mode="before")
    @classmethod
    def _normalize_version(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.lstrip("v")
        parts = value.split(".")
        if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
            raise ValueError(
                "show_hidden_metrics_for_version must look like '0.7' or '0.7.0'"
            )
        return value

    @field_validator("otlp_traces_endpoint", mode="after")
    @classmethod
    def _validate_endpoint(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not (value.startswith("http://") or value.startswith("https://")):
            raise ValueError(
                "otlp_traces_endpoint must start with http:// or https://"
            )
        return value

    @field_validator("collect_detailed_traces", mode="before")
    @classmethod
    def _parse_collect_traces(cls, value: Any) -> Optional[list[DetailedTraceModules]]:
        if value is None:
            return None

        assert isinstance(value, list)
        value = cast(
            list[DetailedTraceModules], value[0].split(",")
        )
        return value

    @model_validator(mode="after")
    def _require_endpoint_if_traces(self):
        if self.collect_detailed_traces and not self.otlp_traces_endpoint:
            raise ValueError(
                "collect_detailed_traces requires `--otlp-traces-endpoint` to be set."
            )
        return self
