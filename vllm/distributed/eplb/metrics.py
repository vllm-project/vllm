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
class EplbMetricsSnapshot:
    """EPLB expert metrics collected by one worker since the previous snapshot."""

    tokens: list[EplbExpertTokens] = field(default_factory=list)
    topology: list[EplbExpertTopology] = field(default_factory=list)
