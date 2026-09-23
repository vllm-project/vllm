# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash


@config
class ExpertLoadStatsConfig:
    """Opt-in target-model logical expert assignment statistics.

    Uses the same collection options and output schema with or without EPLB.
    Balancing policy remains controlled by enable_eplb and eplb_config.
    """

    enabled: bool = False
    """Collect routing statistics without enabling or changing EPLB."""
    log_interval: int = Field(default=1000, ge=1)
    """Target forward iterations between interval summaries."""
    scope: Literal["local"] = "local"
    """Rank-local token ownership only; no additional collectives."""
    detail: Literal["summary", "per_expert"] = "per_expert"
    """Include the count vector in interval summaries when per_expert."""
    reset_after_log: bool = True
    """Report interval counts; false reports cumulative counts since startup."""
    output_dir: str | None = None
    """Optional JSONL output directory; required for per-iteration traces."""
    trace: bool = False
    """Also export per-layer, per-target-forward histograms to JSONL."""
    trace_interval: int = Field(default=1, ge=1)
    """Sample one trace iteration out of every N; summaries include all."""
    trace_max_iterations: int = Field(default=1024, ge=1)
    """Stop tracing after this many target iterations (summaries continue)."""
    flush_interval: int = Field(default=64, ge=1, le=4096)
    """Maximum trace samples per asynchronous export chunk."""
    layers: list[int] | None = None
    """Layer IDs to observe; None selects all supported target MoE layers."""

    @model_validator(mode="after")
    def validate_trace(self) -> Self:
        if self.enabled and self.trace and not self.output_dir:
            raise ValueError("Expert-load tracing requires output_dir")
        if self.layers is not None and (
            not self.layers
            or min(self.layers) < 0
            or len(set(self.layers)) != len(self.layers)
        ):
            raise ValueError("Expert-load layers must be unique nonnegative IDs")
        return self

    def compute_hash(self) -> str:
        # Router instrumentation is captured in CUDA graphs. Reporting options
        # do not change the forward graph, but selected layers do.
        factors = (self.enabled, self.layers if self.enabled else None)
        return safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
