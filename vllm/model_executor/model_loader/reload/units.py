# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shard coverage tracking for streaming, per-unit weight reload.

A *quantizable unit* is the smallest group of incoming shards that can be
converted into serving format without any further input. For per-tensor FP8
MoE that group is one expert's ``w1``/``w3`` weights plus their two scales:
the runtime scale is the max of the pair, so neither half can be finalized
alone. For block, group and channel granularities every shard is a unit by
itself.

``ReloadUnit`` declares that group; ``ShardCoverageTracker`` watches shards
arrive, stages the ones that cannot be written straight into runtime storage,
and runs the unit's commit as soon as its last shard lands -- releasing the
staging buffer immediately. Completion is a set-coverage decision, never an
element count.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "ReloadUnit",
    "StagingSpec",
    "ShardCoverageTracker",
    "ShardKey",
    "SlotKey",
]

# (parameter name, local expert id, shard id). Non-expert layers use 0 for the
# expert id; ``shard_id`` distinguishes the fused halves ("w1"/"w3", "q"/"k"/"v").
ShardKey = tuple[str, int, str]

# (parameter name, local expert id). One staging slab covers all shards that the
# original weight loader writes into the same per-expert slice.
SlotKey = tuple[str, int]


@dataclass(frozen=True)
class StagingSpec:
    """Checkpoint-format description of one expert's slice of a parameter.

    Staging cannot be derived from the runtime parameter: post-load processing
    may have reduced it (per-tensor FP8 keeps ``[E, 2]`` scales on disk but a
    ``[E]`` scale at runtime), so the incoming shape is declared explicitly.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class ReloadUnit:
    """A set of shards that becomes committable only once all of them arrive.

    Args:
        name: Identifier used in diagnostics.
        keys: Shard keys that must all arrive before ``commit`` may run.
        commit: Called with the staged slabs, keyed by slot, once the unit is
            covered. Implementations write serving-format values into the live
            runtime storage in place.
        staged: Per-parameter staging descriptions. Parameters absent from this
            mapping are written in place by the original loader.
        deferred: Commit at ``finish`` rather than on coverage. Used by values
            that must be reduced across the whole layer (per-tensor input
            scales take the max over experts).
    """

    name: str
    keys: frozenset[ShardKey]
    commit: Callable[[dict[SlotKey, torch.Tensor]], None]
    staged: Mapping[str, StagingSpec] = field(default_factory=dict)
    deferred: bool = False
    slots: frozenset[SlotKey] = field(init=False)

    def __post_init__(self) -> None:
        slots = frozenset(
            (name, expert) for name, expert, _ in self.keys if name in self.staged
        )
        object.__setattr__(self, "slots", slots)


class ShardCoverageTracker:
    """Route expert weight writes to staging slabs and commit covered units."""

    def __init__(
        self,
        layer: torch.nn.Module,
        units: Iterable[ReloadUnit],
        num_experts: int | None = None,
    ) -> None:
        self.layer = layer
        self.units = list(units)
        if num_experts is None:
            num_experts = int(getattr(layer, "local_num_experts", 1))
        self.num_experts = num_experts
        self._unit_of_key: dict[ShardKey, ReloadUnit] = {}
        for unit in self.units:
            for key in unit.keys:
                if key in self._unit_of_key:
                    raise ValueError(
                        f"Shard {key} is claimed by both "
                        f"{self._unit_of_key[key].name} and {unit.name}"
                    )
                self._unit_of_key[key] = unit
        self._arrived: dict[str, set[ShardKey]] = {u.name: set() for u in self.units}
        self._committed: set[str] = set()
        self._slabs: dict[SlotKey, torch.Tensor] = {}
        self._proxies: dict[SlotKey, torch.nn.Parameter] = {}
        # Every shard key seen this session, including ones outside any unit.
        self.observed: set[ShardKey] = set()

    def target(self, key: ShardKey, param: torch.nn.Parameter) -> torch.nn.Parameter:
        """Return the parameter the original loader should write into."""
        unit = self._unit_of_key.get(key)
        if unit is None:
            return param
        spec = unit.staged.get(key[0])
        if spec is None:
            return param
        return self._proxy((key[0], key[1]), param, spec)

    def _proxy(
        self, slot: SlotKey, param: torch.nn.Parameter, spec: StagingSpec
    ) -> torch.nn.Parameter:
        proxy = self._proxies.get(slot)
        if proxy is not None:
            return proxy

        # The slab holds one expert's checkpoint-format slice. Exposing it
        # through an expert-dimension broadcast lets the original loader keep
        # indexing `param.data[expert]` and narrowing for TP unchanged.
        slab = torch.zeros(spec.shape, dtype=spec.dtype, device=param.device)
        expanded = slab.reshape(1, *slab.shape).expand(self.num_experts, *slab.shape)
        proxy = torch.nn.Parameter(expanded, requires_grad=False)
        # Loader behaviour is driven by parameter attributes (`quant_method`,
        # `is_transposed`, `load_full_w2`, ...); the staging proxy must present
        # exactly the same ones.
        proxy.__dict__.update(param.__dict__)
        self._slabs[slot] = slab
        self._proxies[slot] = proxy
        return proxy

    def record(self, key: ShardKey) -> None:
        """Note that a shard was loaded, committing its unit once covered."""
        self.observed.add(key)
        unit = self._unit_of_key.get(key)
        if unit is None:
            return

        arrived = self._arrived[unit.name]
        if unit.name in self._committed:
            # A second full round for the same unit: re-arm rather than
            # committing again from a mix of old and new shards.
            self._committed.discard(unit.name)
            arrived.clear()
        arrived.add(key)

        if not unit.deferred and arrived >= unit.keys:
            self._commit(unit)

    def _commit(self, unit: ReloadUnit) -> None:
        pieces = {slot: self._slabs[slot] for slot in unit.slots if slot in self._slabs}
        missing = unit.slots - pieces.keys()
        if missing:
            raise RuntimeError(
                f"{unit.name} is covered but staging slabs {sorted(missing)} "
                "were never allocated"
            )
        unit.commit(pieces)
        for slot in unit.slots:
            self._slabs.pop(slot, None)
            self._proxies.pop(slot, None)
        self._arrived[unit.name].clear()
        self._committed.add(unit.name)

    def finish(self, *, fail_on_partial: bool = True) -> None:
        """Commit deferred units and optionally tolerate partial units.

        Args:
            fail_on_partial: Raise for a partially received unit when true.
                Modelwise streaming reload passes false so incomplete modules
                can be discarded while complete modules are committed.
        """
        problems: list[str] = []
        try:
            for unit in self.units:
                arrived = self._arrived[unit.name]
                if not arrived:
                    continue
                if arrived >= unit.keys:
                    self._commit(unit)
                    continue
                problems.append(
                    f"{unit.name}: missing {sorted(unit.keys - arrived)[:4]}"
                )
        finally:
            self.release()

        if problems and fail_on_partial:
            raise ValueError(
                f"{self.layer.__class__.__name__} received a partial weight "
                "update; these units never completed:\n  " + "\n  ".join(problems[:10])
            )

    def release(self) -> None:
        """Drop staging buffers without committing them."""
        self._slabs.clear()
        self._proxies.clear()

    @property
    def staged_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._slabs.values())


TRACKER_ATTR = "_reload_shard_tracker"


def install_trackers(
    layers: Iterable[tuple[str, torch.nn.Module, ShardCoverageTracker]],
) -> None:
    for _, module, tracker in layers:
        setattr(module, TRACKER_ATTR, tracker)


def uninstall_trackers(
    layers: Iterable[tuple[str, torch.nn.Module, ShardCoverageTracker]],
) -> None:
    for _, module, tracker in layers:
        tracker.release()
        setattr(module, TRACKER_ATTR, None)
