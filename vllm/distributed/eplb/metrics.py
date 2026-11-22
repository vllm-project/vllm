# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB metric payloads sent from workers to the metrics frontend."""

from dataclasses import dataclass, field

EPLB_PHASE_PREFILL = "prefill"
EPLB_PHASE_DECODE = "decode"
EPLB_PHASE_MIXED = "mixed"
EPLB_PHASES = (
    EPLB_PHASE_PREFILL,
    EPLB_PHASE_DECODE,
    EPLB_PHASE_MIXED,
)


@dataclass
class EplbExpertTokens:
    """Tokens routed to each physical expert over one EPLB load window.

    ``tokens[layer][physical_expert]`` counts the tokens routed by this EP
    rank. Summing the samples of all EP ranks (via the Prometheus ``engine``
    label) yields the global per-expert load.
    """

    model_name: str
    ep_rank: int
    num_ep_ranks: int
    num_local_experts: int
    tokens: list[list[int]]
    physical_to_logical: list[list[int]] | None = None
    """Placement that was active while these tokens were routed."""
    phase: str = EPLB_PHASE_MIXED
    """Batch phase for this sample: prefill, decode, or mixed."""
    timestamp: float = 0.0
    """Unix timestamp when this load sample was taken."""
    steps: int = 0
    """Number of engine steps covered by this load sample."""
    rearranged: bool = False
    """Whether this sample was taken at an EPLB rearrangement."""


@dataclass
class EplbExpertTopology:
    """Physical-to-logical expert placement.

    ``physical_to_logical[layer][physical_expert]`` is the logical expert held
    by that physical slot. Physical experts are assigned to EP ranks in
    contiguous blocks of ``num_local_experts``.
    """

    model_name: str
    ep_rank: int
    num_ep_ranks: int
    num_local_experts: int
    physical_to_logical: list[list[int]]
    timestamp: float = 0.0
    """Unix timestamp when this topology was committed."""


@dataclass
class EplbRearrangementStall:
    """Blocking interval spent rearranging experts while requests were in flight.

    ``pause_seconds`` is measured on the worker around the synchronous part of
    :meth:`EplbState.rearrange`. ``blocked_requests`` and ``blocked_tokens`` are
    the batch sizes observed for the step that triggered the rearrangement; they
    weight the pause by the work that could not make progress.
    """

    model_name: str
    pause_seconds: float
    blocked_requests: int
    blocked_tokens: int
    phase: str
    is_async: bool
    timestamp: float


@dataclass
class EplbLoadSample:
    """Per-layer load sample used to derive expert- and rank-skew signals."""

    model_name: str
    phase: str
    layer: int
    num_ep_ranks: int
    num_local_experts: int
    expert_tokens: list[int]
    """Tokens routed to each physical expert in this layer."""
    expert_gini: float
    """Gini coefficient of ``expert_tokens``; 0.0 is balanced, 1.0 is concentrated."""
    expert_top_k: int
    """Number of hottest experts used for ``expert_topk_share``."""
    expert_topk_share: float
    """Share of layer tokens routed to the ``expert_top_k`` hottest experts."""
    rank_tokens: list[int]
    """Tokens routed to each EP rank in this layer."""
    active_experts_per_rank: list[int]
    """Physical experts with nonzero load on each EP rank in this layer."""
    expert_skew: float
    """``max(expert_tokens) / mean(expert_tokens)``; 1.0 is balanced."""
    expert_excess_tokens: int
    """``max(expert_tokens) - mean(expert_tokens)``, rounded down."""
    rank_skew: float
    """``max(rank_tokens) / mean(rank_tokens)``; 1.0 is balanced."""
    rank_excess_tokens: int
    """``max(rank_tokens) - mean(rank_tokens)``, rounded down."""
    timestamp: float
    steps: int


@dataclass
class EplbMetricsSnapshot:
    """EPLB expert metrics collected by one worker since the previous snapshot."""

    tokens: list[EplbExpertTokens] = field(default_factory=list)
    topology: list[EplbExpertTopology] = field(default_factory=list)
    stalls: list[EplbRearrangementStall] = field(default_factory=list)
    loads: list[EplbLoadSample] = field(default_factory=list)
