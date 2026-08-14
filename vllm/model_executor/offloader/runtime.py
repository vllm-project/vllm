# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime helpers for prefetch offload unit scheduling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeOffloadUnit:
    """Execution-time metadata for one offload unit."""

    unit_idx: int
    slot_idx: int


class PrefetchRuntimeController:
    """Controls offload-unit prefetch sequencing and slot reuse."""

    def __init__(self, unit_count: int, prefetch_step: int):
        if prefetch_step < 1:
            raise ValueError("prefetch_step must be >= 1")
        self.unit_count = unit_count
        self.prefetch_step = prefetch_step
        self.units = tuple(
            RuntimeOffloadUnit(
                unit_idx=unit_idx,
                slot_idx=unit_idx % prefetch_step,
            )
            for unit_idx in range(unit_count)
        )
        self._prefetch_after = self._build_prefetch_after()
        self._pending_capture_units: set[int] = set()
        self._slot_owners: list[int | None] = [None] * prefetch_step

    def _build_prefetch_after(self) -> tuple[RuntimeOffloadUnit | None, ...]:
        """Return the next unit to prefetch after each unit executes.

        Units sharing a slot form a circular sequence. A slot is only reused
        after its current owner has executed, avoiding copies into a slot still
        needed by a not-yet-executed unit when the unit count is not divisible
        by the prefetch window.
        """
        slot_units: list[list[RuntimeOffloadUnit]] = [
            [] for _ in range(self.prefetch_step)
        ]
        for unit in self.units:
            slot_units[unit.slot_idx].append(unit)

        prefetch_after: list[RuntimeOffloadUnit | None] = [None] * self.unit_count
        for units_in_slot in slot_units:
            if len(units_in_slot) <= 1:
                continue
            for index, unit in enumerate(units_in_slot):
                prefetch_after[unit.unit_idx] = units_in_slot[
                    (index + 1) % len(units_in_slot)
                ]

        return tuple(prefetch_after)

    def get_unit(self, unit_idx: int) -> RuntimeOffloadUnit:
        """Return metadata for one runtime unit."""
        return self.units[unit_idx]

    def begin_prefetch(self, unit_idx: int) -> RuntimeOffloadUnit | None:
        """Assign the destination slot to one unit and return the previous owner."""
        if self.unit_count == 0:
            return None

        runtime_unit = self.units[unit_idx]
        previous_owner = self._slot_owners[runtime_unit.slot_idx]
        self._slot_owners[runtime_unit.slot_idx] = unit_idx
        if previous_owner is None or previous_owner == unit_idx:
            return None
        return self.units[previous_owner]

    def is_unit_resident(self, unit_idx: int) -> bool:
        """Return whether one unit currently owns its runtime buffer slot."""
        if self.unit_count == 0:
            return False
        runtime_unit = self.units[unit_idx]
        return self._slot_owners[runtime_unit.slot_idx] == unit_idx

    def reset(self) -> None:
        """Clear runtime ownership and capture bookkeeping."""
        self._pending_capture_units.clear()
        self._slot_owners = [None] * self.prefetch_step

    def initial_prefetches(self) -> tuple[RuntimeOffloadUnit, ...]:
        """Return the units prefetched before the first execution."""
        return self.units[: min(self.prefetch_step, self.unit_count)]

    def prefetch_after(self, unit_idx: int) -> RuntimeOffloadUnit | None:
        """Return the unit that should be prefetched after executing one unit."""
        if self.unit_count == 0:
            return None
        return self._prefetch_after[unit_idx]

    def mark_prefetch_started(self, unit_idx: int, *, in_capture: bool) -> None:
        """Track whether a unit's latest prefetch started during capture."""
        if in_capture:
            self._pending_capture_units.add(unit_idx)
        else:
            self._pending_capture_units.discard(unit_idx)

    def mark_waited(self, unit_idx: int) -> None:
        """Mark one unit's capture-started prefetch as joined."""
        self._pending_capture_units.discard(unit_idx)

    def is_pending_in_capture(self, unit_idx: int) -> bool:
        """Return whether a unit has an unjoined capture-started prefetch."""
        return unit_idx in self._pending_capture_units

    def pending_capture_prefetches(self) -> tuple[RuntimeOffloadUnit, ...]:
        """Return units whose capture-started prefetch still needs joining."""
        return tuple(
            unit for unit in self.units if unit.unit_idx in self._pending_capture_units
        )
