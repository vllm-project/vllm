# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What speculative decoding holds, shared by the scheduler, the engine core and
the workers."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig

logger = init_logger(__name__)

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


@dataclass(frozen=True)
class SpecLayerKind:
    """What a layer is to speculative decoding, derived from the KV cache spec
    or the layer class (a plugin cannot invent one)."""

    family: Literal["attn", "recurrent", "layout"]
    variant: str
    """attn: the ``KVCacheSpecKind`` value; recurrent: the layer class name;
    layout: ``"paged_token"``, ``"paged_split_scale"`` or ``"recurrent_state"``."""

    def key(self) -> str:
        return f"{self.family}:{self.variant}"


@dataclass(frozen=True)
class SpecLimit:
    """A layer kind or a setting that caps the plan kind, and why."""

    plan_kind: PlanKind
    reason: str


@dataclass(frozen=True)
class SpecCapabilities:
    """What a worker's layers verify; reduced over ranks by the engine core."""

    plan_kind: PlanKind
    """``"chain"`` when any layer or setting caps it, else ``"tree"``."""
    limits: tuple[SpecLimit, ...] = ()
    """Every cap to chains, for the startup log."""

    @classmethod
    def from_limits(cls, limits: Iterable[SpecLimit]) -> "SpecCapabilities":
        limits = tuple(dict.fromkeys(limits))
        return cls(plan_kind="chain" if limits else "tree", limits=limits)


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
    effective = resolve_spec_resources(speculator_config, declared, capabilities)
    collective_rpc("set_spec_resources", args=(effective,))
    return effective


def resolve_spec_resources(
    speculator_config: "SpeculatorConfig",
    declared: SpecResources,
    capabilities: Sequence[SpecCapabilities],
) -> SpecResources:
    """The speculator's declared resources adapted to what every rank verifies
    (``capabilities``, one per worker); logs each lowering."""
    # Pipeline ranks hold different layers: the model verifies the lowest.
    caps = SpecCapabilities.from_limits(
        limit for rank in capabilities for limit in rank.limits
    )
    lowering = caps.limits if declared.plan_kind == "tree" else ()
    for limit in lowering:
        logger.info(
            "spec decode: %s -> %s (%s)",
            declared.plan_kind,
            limit.plan_kind,
            limit.reason,
        )
    effective = speculator_config.adapt_resources(declared, caps)
    validate_effective_resources(declared, effective, caps)
    if effective.max_rows < declared.max_rows:
        logger.info(
            "spec decode: max_rows %d -> %d (adapt_resources)",
            declared.max_rows,
            effective.max_rows,
        )
    return effective


def validate_effective_resources(
    declared: SpecResources, effective: SpecResources, caps: SpecCapabilities
) -> None:
    """Check that ``effective`` takes no more than ``declared`` and ``caps``."""
    problems = []
    if effective.plan_kind == "tree" and "chain" in (
        declared.plan_kind,
        caps.plan_kind,
    ):
        problems.append(
            f"plan_kind {effective.plan_kind!r} is above the declared "
            f"{declared.plan_kind!r} or the model's {caps.plan_kind!r}"
        )
    if effective.max_branching > declared.max_branching:
        problems.append(
            f"max_branching {effective.max_branching} is above the declared "
            f"{declared.max_branching}"
        )
    extra_rows = set(effective.capture_rows) - set(declared.capture_rows)
    if extra_rows:
        problems.append(f"capture_rows {sorted(extra_rows)} not declared")
    if problems:
        raise ValueError(
            "SpeculatorConfig.adapt_resources returned resources the model cannot "
            "run: " + "; ".join(problems)
        )
