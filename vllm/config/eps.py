# ABOUTME: Defines global EPS configuration knobs.
# ABOUTME: Serializes EigenPage Summaries settings for engine wiring.

import hashlib
from typing import Literal

from pydantic import Field
from pydantic.dataclasses import dataclass

from vllm.config.utils import config


@config
@dataclass
class EpsConfig:
    """EigenPage Summaries configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable EigenPage Summaries pre-pass.",
    )
    method: Literal["off", "jl", "1pc"] = Field(
        default="off",
        description="EPS summarization method.",
    )
    heads: Literal["retrieval", "all"] = Field(
        default="retrieval",
        description="Which attention heads apply EPS gating.",
    )
    scope: Literal["union"] = Field(
        default="union",
        description="EPS gating scope. Currently only `union` is supported.",
    )
    group_blocks: int = Field(
        default=8,
        description="Number of contiguous KV blocks per EPS group.",
        ge=1,
    )
    last_n: int = Field(
        default=8,
        description="Always visit the last N groups regardless of scores.",
        ge=0,
    )
    alpha: float = Field(
        default=1.1,
        description="Safety factor for EPS bound comparison.",
        gt=0.0,
    )
    dim: int = Field(
        default=8,
        description="JL sketch dimension for EPS summaries.",
        ge=1,
    )
    top_pages: int | None = Field(
        default=None,
        description="Maximum number of page groups to visit per request.",
        ge=1,
    )
    strict: bool = Field(
        default=False,
        description="Enable strict guardrails (larger safety factor, forced visits).",
    )
    metrics_path: str | None = Field(
        default=None,
        description="Optional JSONL file to append per-step EPS counters.",
    )

    def compute_hash(self) -> str:
        payload = (
            self.enabled,
            self.method,
            self.heads,
            self.scope,
            self.group_blocks,
            self.last_n,
            float(self.alpha),
            self.dim,
            self.top_pages,
            self.strict,
            self.metrics_path,
        )
        return hashlib.md5(str(payload).encode(), usedforsecurity=False).hexdigest()
