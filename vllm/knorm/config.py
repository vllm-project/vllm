# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Knorm configuration, read from registered vLLM environment variables."""

from __future__ import annotations

from dataclasses import dataclass, field

import vllm.envs as envs


@dataclass
class KnormConfig:
    """Knorm compression configuration.

    All values are read from vllm.envs, which lazily evaluates
    environment variables registered in ``environment_variables`` dict.
    """

    compression_ratio: float = field(
        default_factory=lambda: envs.VLLM_KNORM_COMPRESSION_RATIO
    )
    """Fraction of KV cache blocks to KEEP. 1.0 = no compression."""

    warmup_tokens: int = field(default_factory=lambda: envs.VLLM_KNORM_WARMUP_TOKENS)
    """Tokens at the sequence start to always keep (attention sink)."""

    enabled: bool = field(default_factory=lambda: envs.VLLM_KNORM_ENABLED)
    """Whether Knorm compression is enabled. Set to '0' to disable."""

    score_aggregation: str = field(
        default_factory=lambda: envs.VLLM_KNORM_SCORE_AGGREGATION
    )
    """How to aggregate token norms within a block: 'min', 'mean', or 'max'."""

    norm_reduce_op: str = field(default_factory=lambda: envs.VLLM_KNORM_NORM_REDUCE_OP)
    """How to reduce norms across heads: 'mean', 'max', or 'sum'."""

    @property
    def is_active(self) -> bool:
        """Return True if compression should be applied."""
        return self.enabled and self.compression_ratio < 1.0
