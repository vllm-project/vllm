# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded reusable staging storage for NCCL M2N destinations."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class M2NStagingRequirement:
    """One planned use of a staging slot."""

    group: int
    slot: int
    dtype: torch.dtype
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.group < 0 or self.slot < 0:
            raise ValueError("staging group and slot must be non-negative")
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise ValueError("staging shape must contain positive dimensions")


class M2NStagingPool:
    """Reuse high-watermark buffers while honoring retained input lifetimes."""

    def __init__(
        self,
        requirements: Sequence[M2NStagingRequirement],
        device: torch.device,
    ) -> None:
        self._requirements = frozenset(requirements)
        self._expected_groups = frozenset(
            requirement.group for requirement in requirements
        )
        capacities: dict[tuple[int, torch.dtype], int] = {}
        for requirement in requirements:
            key = (requirement.slot, requirement.dtype)
            capacities[key] = max(capacities.get(key, 0), math.prod(requirement.shape))
        self._device = device
        self._capacities = capacities
        self._buffers = {
            key: [torch.empty(capacity, dtype=key[1], device=device)]
            for key, capacity in capacities.items()
        }
        self._active_slots: dict[tuple[int, int], torch.Tensor] = {}
        self._leased_buffer_ids: set[int] = set()
        self._active_groups: dict[int, set[int]] = {}
        self._completed_groups: set[int] = set()
        self._updating = False

    def start_update(self) -> None:
        if self._updating:
            raise RuntimeError("M2N staging update is already active")
        if self._active_slots or self._active_groups:
            raise RuntimeError("M2N staging pool retained buffers between updates")
        self._completed_groups.clear()
        self._updating = True

    def acquire(
        self,
        *,
        group: int,
        slot: int,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Lease one exact planned view for the active update."""
        if not self._updating:
            raise RuntimeError("M2N staging update is not active")
        requirement = M2NStagingRequirement(group, slot, dtype, shape)
        if requirement not in self._requirements:
            raise ValueError(f"unplanned M2N staging request: {requirement}")
        if group in self._completed_groups:
            raise RuntimeError(f"M2N staging group {group} was consumed more than once")
        lease = (group, slot)
        if lease in self._active_slots:
            raise RuntimeError(
                f"M2N staging group {group} slot {slot} was acquired twice"
            )
        key = (slot, dtype)
        buffer = next(
            (
                candidate
                for candidate in self._buffers[key]
                if id(candidate) not in self._leased_buffer_ids
            ),
            None,
        )
        if buffer is None:
            buffer = torch.empty(
                self._capacities[key], dtype=dtype, device=self._device
            )
            self._buffers[key].append(buffer)
        numel = math.prod(shape)
        if numel > buffer.numel():
            raise AssertionError("planned M2N staging capacity is too small")
        self._active_slots[lease] = buffer
        self._leased_buffer_ids.add(id(buffer))
        self._active_groups.setdefault(group, set()).add(slot)
        return buffer[:numel].view(shape)

    def release_group(self, group: int) -> None:
        """Release every lease retained by one completed semantic consumer."""
        slots = self._active_groups.pop(group, None)
        if not slots:
            raise RuntimeError(f"M2N staging group {group} has no active leases")
        for slot in slots:
            buffer = self._active_slots.pop((group, slot))
            self._leased_buffer_ids.remove(id(buffer))
        self._completed_groups.add(group)

    def finish_update(self) -> None:
        """Require all planned groups to have completed before reuse."""
        if not self._updating:
            raise RuntimeError("M2N staging update is not active")
        if self._active_groups:
            pending = sorted(self._active_groups)
            raise RuntimeError(
                f"M2N staging groups still retain input buffers: {pending}"
            )
        missing = sorted(self._expected_groups - self._completed_groups)
        if missing:
            raise RuntimeError(f"M2N staging groups were not consumed: {missing}")
        self._updating = False

    def release_retained_after_finalize(self) -> None:
        """Release inputs after layerwise finalization drops loader references."""
        if not self._updating:
            raise RuntimeError("M2N staging update is not active")
        for group in sorted(tuple(self._active_groups)):
            self.release_group(group)

    def discard_update(self) -> None:
        """Forget leases after a poisoned update; their data is not reusable."""
        self._active_slots.clear()
        self._leased_buffer_ids.clear()
        self._active_groups.clear()
        self._completed_groups.clear()
        self._updating = False
