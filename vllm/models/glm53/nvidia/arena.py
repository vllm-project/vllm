# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The eight-rank peer-memory arena of GLM-5.3's decode kernels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.models.glm53.nvidia.locality import (
    AllocationMark,
    LocalityDomains,
    domains_for,
)
from vllm.models.glm53.nvidia.ops.cute_dsl.fused_moe import (
    ROUTED_CTAS,
    Flag,
    FusedMoe,
)
from vllm.models.glm53.nvidia.ops.cute_dsl.mla_worker import Handoff, MlaWorker
from vllm.models.glm53.nvidia.ops.launch import Region, sizes_of

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


@dataclass(frozen=True)
class Slice:
    """One named mailbox slice: ``size`` bytes at a multiple of ``alignment``."""

    MAX_ALIGNMENT: ClassVar[int] = 256

    name: str
    size: int
    alignment: int = 128


def layout(slices: Sequence[Slice]) -> tuple[dict[str, int], int]:
    """Return each slice's offset and the arena's size."""
    offsets: dict[str, int] = {}
    offset = 0
    for item in sorted(slices, key=lambda entry: entry.name):
        if item.name in offsets:
            raise ValueError(f"slice {item.name!r} is reserved twice")
        if (
            item.alignment & (item.alignment - 1)
            or not 0 < item.alignment <= Slice.MAX_ALIGNMENT
        ):
            raise ValueError(f"slice {item.name!r} has alignment {item.alignment}")
        offset = -(-offset // item.alignment) * item.alignment
        offsets[item.name] = offset
        offset += item.size
    alignment = max((item.alignment for item in slices), default=1)
    return offsets, -(-offset // alignment) * alignment


class PeerArena:
    """This rank's arena in symmetric memory and every rank's base address."""

    def __init__(
        self,
        slices: Sequence[Slice],
        group: ProcessGroup,
        contents: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        self.slices: tuple[Slice, ...] = tuple(slices)
        self.offsets, self.bytes_per_rank = layout(self.slices)
        device = torch.device("cuda", torch.cuda.current_device())
        self._buffer: torch.Tensor | None = symm_mem.empty(
            self.bytes_per_rank, dtype=torch.uint8, device=device
        )
        self._buffer.zero_()
        for name, tensor in (contents or {}).items():
            data = tensor.contiguous().view(-1).view(torch.uint8)
            self.view(name)[: data.numel()].copy_(data)
        torch.cuda.synchronize(device)
        self._handle: symm_mem._SymmetricMemory | None = symm_mem.rendezvous(
            self._buffer, group
        )
        self.rank: int = self._handle.rank
        self.peer_addresses: tuple[int, ...] = tuple(
            int(pointer) for pointer in self._handle.buffer_ptrs
        )
        if (
            self.peer_addresses[self.rank] != self._buffer.data_ptr()
            or self._buffer.data_ptr() % Slice.MAX_ALIGNMENT
        ):
            raise RuntimeError(
                "symmetric memory: the arena base is not this rank's mapped buffer"
            )
        self._table: torch.Tensor | None = torch.tensor(
            self.peer_addresses, dtype=torch.int64, device=device
        )
        self.table: int = self._table.data_ptr()

    def view(self, name: str) -> torch.Tensor:
        """Return this rank's bytes of slice ``name`` (a uint8 view of the arena)."""
        if self._buffer is None:
            raise RuntimeError("the peer arena is closed")
        offset = self.offsets[name]
        size = next(item.size for item in self.slices if item.name == name)
        return self._buffer[offset : offset + size]

    def regions(self) -> dict[str, Region]:
        """Return ``peer<p>:<slice>`` for every rank and slice, and ``peer_table``."""
        regions = {
            f"peer{peer}:{item.name}": Region(base + self.offsets[item.name], item.size)
            for peer, base in enumerate(self.peer_addresses)
            for item in self.slices
        }
        regions["peer_table"] = Region(
            self.table, self._table.element_size() * len(self.peer_addresses)
        )
        return regions

    def close(self) -> None:
        """Release this rank's arena (after every graph using it is destroyed and no
        peer stores into it).
        """
        self._buffer = self._table = self._handle = None
        self.table = 0


GROUP_A = tuple(range(0, 64))
GROUP_B = tuple(range(64, 132))


class WorkerPlacement:
    """The role table and the two-pool handoff of one worker layer."""

    def __init__(self, device: torch.device, domains: LocalityDomains, by_sm: bool):
        """Assign worker roles and allocate the handoff."""
        grid = MlaWorker.GRID
        sm_domains = domains.sm_domain
        if by_sm and len(sm_domains) != grid:
            raise RuntimeError(
                f"domain placement: the {grid}-CTA grid must cover the device's "
                f"{len(sm_domains)} SMs"
            )
        block_sms = list(range(grid))
        block_domains = [0] * grid
        if domains.active and by_sm:
            block_sms = domains.grid_sms(grid)
            if sorted(block_sms) != list(range(grid)):
                raise RuntimeError("domain placement: the grid does not cover the SMs")
            block_domains = [sm_domains[sm] for sm in block_sms]
        elif domains.active:
            block_domains = domains.grid_domains(grid)
        counts = [block_domains.count(0), block_domains.count(1)]

        def spill(a: int) -> int:
            """Count the CTAs that miss their group."""
            return max(0, len(GROUP_A) - counts[a]) + max(
                0, len(GROUP_B) - counts[1 - a]
            )

        self.domain_a: int = min((0, 1), key=lambda a: (spill(a), counts[a]))
        roles = list(range(grid))
        if domains.active:
            roles = self._assign_roles(block_domains)
        else:
            self.domain_a = 0
        entries = [
            role | (int(domain != self.domain_a) << Handoff.POOL_SHIFT)
            for role, domain in zip(roles, block_domains)
        ]
        if by_sm:
            by_index = [0] * grid
            for block, sm in enumerate(block_sms):
                by_index[sm] = entries[block]
            entries = by_index
        self.role_table: tuple[int, ...] = tuple(entries)
        self.handoff: torch.Tensor = domains.mirrored_words(
            self.domain_a, Handoff.MIRROR, Handoff.WORDS
        )
        self.mirror: torch.Tensor = domains.mirror_of(self.handoff, Handoff.MIRROR)
        self.handoff[Handoff.ROLE_TABLE : Handoff.ROLE_TABLE + grid] = torch.tensor(
            self.role_table, dtype=torch.int32, device=device
        )

    def _assign_roles(self, block_domains: list[int]) -> list[int]:
        """Return each CTA's role."""
        grid = MlaWorker.GRID
        wanted = {self.domain_a: list(GROUP_A), 1 - self.domain_a: list(GROUP_B)}
        placed = set(GROUP_A + GROUP_B)
        rest = [cta for cta in range(grid) if cta not in placed]
        roles = [-1] * grid
        for domain, own in wanted.items():
            slots = [b for b in range(grid) if block_domains[b] == domain]
            rest += own[len(slots) :]
            for block, cta in zip(slots, own):
                roles[block] = cta
        for block in range(grid):
            if roles[block] < 0:
                roles[block] = rest.pop(0)
        if sorted(roles) != list(range(grid)):
            raise AssertionError("domain placement: the roles are not a permutation")
        return roles


class Glm53Placement:
    """One rank's optional locality-aware placement."""

    def __init__(self, device: torch.device) -> None:
        """Detect locality support (a plain fallback where it is absent)."""
        self._device: torch.device = device
        self._flag_words: int = sizes_of("fused_moe_q4_ckpt_shared")["kFlagWords"]
        self.domains: LocalityDomains = domains_for(device)
        # Everything this placement allocates is given back by release().
        self._mark: AllocationMark = self.domains.mark()
        self.preferred_domain: int = 0
        if self.domains.active:
            cta_domains = self.domains.grid_domains(FusedMoe.GRID)[:ROUTED_CTAS]
            self.preferred_domain = self.domains.majority(cta_domains)

    def release(self) -> None:
        """Return the worker handoffs and MoE flags to their pools (the segment is
        closed).
        """
        self.domains.release(self._mark)

    def worker_handoff(self, by_sm: bool) -> WorkerPlacement:
        """Place one worker layer."""
        return WorkerPlacement(self._device, self.domains, by_sm)

    def moe_flags(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate one zeroed four-token MoE ``flags`` tensor and its second copy."""
        flags = self.domains.mirrored_words(
            self.preferred_domain, Flag.MIRROR, self._flag_words
        )
        bits = self.domains.copy_bits(self.preferred_domain)
        flags[Flag.SM_DOMAIN : Flag.SM_DOMAIN + len(bits)] = torch.tensor(
            bits, dtype=torch.int32, device=self._device
        )
        return flags, self.domains.mirror_of(flags, Flag.MIRROR)
