# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for Mixture of Experts optimizations."""

from vllm.config.utils import config


@config
class MoEPruningConfig:
    """Batch-aware expert pruning configuration (XShare method)."""

    enable: bool = False
    """Master switch -- must be True for any pruning to happen."""

    expert_budget: int | None = None
    """Absolute number of experts to keep (overrides budget_alpha)."""

    budget_alpha: float = 0.5
    """Fraction of experts to keep when expert_budget is not set."""

    top_per_token: int = 0
    """Per-token top-k pre-selection (0 = disabled)."""

    group_budget: int = 0
    """Per-group expert budget."""

    group_size: int = 0
    """Tokens per group for per-group pruning."""

    min_batch: int = 0
    """Skip pruning when batch size is below this threshold."""

    max_batch: int = 0
    """Skip pruning when batch size is above this threshold (0 = no limit)."""


@config
class MoEConfig:
    """Configuration for Mixture of Experts optimizations.

    Consolidates MoE-related settings under one namespace.
    Currently supports expert pruning; routing, parallelism,
    and kernel selection are planned for future releases.

    Example usage::

        vllm serve model --moe-config '{"pruning":
            {"enable": true, "expert_budget": 24}}'
    """

    pruning: MoEPruningConfig | None = None
    """Expert pruning configuration. None means pruning is disabled."""
    # Future sub-configs (stubs):
    # routing: MoERoutingConfig | None = None
    # parallelism: MoEParallelismConfig | None = None
    # kernel: MoEKernelConfig | None = None

    def compute_hash(self) -> str:
        from vllm.config.utils import get_hash_factors, hash_factors
        factors = get_hash_factors(self)
        return hash_factors(factors)
