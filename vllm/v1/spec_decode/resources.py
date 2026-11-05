# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What speculative decoding holds, shared by the scheduler, the engine core and
the workers."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from vllm.config import VllmConfig

PlanKind = Literal["chain", "tree"]
"""The plans a model verifies: ``"chain"`` (each row follows the one before
it) or ``"tree"`` (each row names its parent row and sees its ancestors)."""


@dataclass(frozen=True)
class SpecResources:
    """What speculation holds, declared by ``SpeculatorConfig.resources`` and
    lowered at startup (``round_contract.md``, Startup)."""

    capture_rows: tuple[int, ...]
    """Rows per request, the root included, of each decode graph (1 = plain
    decode); at most ``num_speculative_tokens + 1``."""
    max_branching: int = 1
    """1 for chains. Above 1 the attention backend and the linear-attention layers must
    take a parent per row."""

    @property
    def plan_kind(self) -> PlanKind:
        """The plans the speculator verifies: trees when it branches."""
        return "tree" if self.max_branching > 1 else "chain"

    @property
    def max_rows(self) -> int:
        """The most rows a request verifies, the root included."""
        return max(self.capture_rows)

    def __post_init__(self) -> None:
        bad = [n for n in self.capture_rows if not isinstance(n, int) or n < 1]
        if not self.capture_rows or bad:
            raise ValueError(
                f"capture_rows must be positive ints, at least one; got "
                f"{self.capture_rows}."
            )
        if self.max_branching < 1:
            raise ValueError(
                f"max_branching ({self.max_branching}) must be >= 1: 1 is a chain."
            )


@dataclass(frozen=True)
class BudgetView:
    """What the scheduler knows when it asks for the step's candidate budget
    (:meth:`SpeculatorConfig.candidate_budget`). Host integers, read once all
    of this step's requests are scheduled and their blocks allocated."""

    num_scheduled_reqs: int
    num_running: int
    num_waiting: int
    num_free_blocks: int
    max_num_seqs: int
    block_size: int
    """Tokens per block of the unified pool (the scheduler's block size)."""


def adapt_spec_resources(
    vllm_config: "VllmConfig", collective_rpc: Callable[..., list[Any]]
) -> SpecResources | None:
    """Lower a branching speculator's declared resources to what every rank's
    layers verify and run every worker with them, before capture. Returns the
    resources for the scheduler, or None to run as declared (a chain, which each
    worker checks against its own layers)."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return None
    speculator_config = speculative_config.speculator_config
    declared = speculator_config.resources(vllm_config.scheduler_config.max_num_seqs)
    if declared.max_branching == 1:
        return None
    capabilities = [c for c in collective_rpc("get_spec_capabilities") if c is not None]
    if not capabilities:
        return None
    from vllm.v1.worker.gpu.spec_decode.capabilities import resolve_spec_resources

    effective = resolve_spec_resources(speculator_config, declared, capabilities)
    collective_rpc("set_spec_resources", args=(effective,))
    return effective
