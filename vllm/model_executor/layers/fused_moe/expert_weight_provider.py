# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# How a forward too wide for the cache is broken up.
#   "token"  -- split the rows; every token's full sum still happens inside one
#               kernel call, so output matches the uncached path exactly. Costs
#               one launch per chunk, which approaches one per token as
#               capacity approaches top_k.
#   "expert" -- split the experts; one launch per ceil(experts/capacity)
#               regardless of batch size, and each expert is fetched at most
#               once per forward. Each group's partial sum is rounded to the
#               model dtype before being accumulated, so results differ from
#               the uncached path at rounding level.
MoECacheSplit = Literal["token", "expert"]

# Every row: what the expert split passes, since it never cuts the batch.
_ALL_ROWS = slice(None)


@dataclass
class ExpertWeightResult:
    """GPU-resident expert weights ready for kernel consumption.

    ``expert_map`` follows the expert-parallel convention the kernels already
    understand: global expert id to buffer slot, or -1 for experts that are not
    resident. Pairs routed to -1 are dropped during alignment, which is what
    lets one forward be evaluated a group of experts at a time.
    """

    w1: torch.Tensor
    w2: torch.Tensor
    expert_map: torch.Tensor
    w1_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None


class CachedWeightProvider:
    """GPU LRU cache backed by CPU pinned memory.

    Keeps capacity expert weight tensors in a fixed-size GPU scratch
    buffer. All expert weights reside in CPU pinned memory; only the N
    hottest experts are mirrored into the GPU buffer.

    Uses LFRU (frequency-weighted LRU) eviction: score = freq / age.
    This prevents early layers from monopolizing the cache — a known
    problem with pure LRU in sequential MoE execution where early
    layers always appear "recently used."

    prepare() copies any missing experts from CPU to GPU, evicting the
    lowest-scored resident entry when the buffer is full, and returns an
    ExpertWeightResult whose expert_map selects them. Forwards needing more
    experts than fit are handled by run_with_expert_cache(), which splits
    them according to ``split``.
    """

    def __init__(
        self,
        capacity: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        split: MoECacheSplit = "token",
    ) -> None:
        num_experts = w13_weight.size(0)

        self.capacity = capacity
        self.split: MoECacheSplit = split
        self._num_experts = num_experts
        self.hits = 0
        self.misses = 0

        if w13_weight.device.type == "cpu":
            cuda_device = torch.accelerator.current_accelerator()
            self._cpu_w13: torch.Tensor = (
                w13_weight if w13_weight.is_pinned() else w13_weight.pin_memory()
            )
            self._cpu_w2: torch.Tensor = (
                w2_weight if w2_weight.is_pinned() else w2_weight.pin_memory()
            )
        else:
            cuda_device = w13_weight.device
            self._cpu_w13 = w13_weight.cpu().pin_memory()
            self._cpu_w2 = w2_weight.cpu().pin_memory()

        self._buf_w13: torch.Tensor = torch.empty(
            capacity,
            *w13_weight.shape[1:],
            dtype=w13_weight.dtype,
            device=cuda_device,
        )
        self._buf_w2: torch.Tensor = torch.empty(
            capacity,
            *w2_weight.shape[1:],
            dtype=w2_weight.dtype,
            device=cuda_device,
        )

        if w13_scale is not None and w2_scale is not None:
            # Pinned for the same reason the weights are: these are copied on
            # every miss, and pageable source memory forces a staging copy.
            self._cpu_w13_scale: torch.Tensor | None = w13_scale.cpu().pin_memory()
            self._cpu_w2_scale: torch.Tensor | None = w2_scale.cpu().pin_memory()
            self._buf_w13_scale: torch.Tensor | None = torch.empty(
                capacity,
                *w13_scale.shape[1:],
                dtype=w13_scale.dtype,
                device=cuda_device,
            )
            self._buf_w2_scale: torch.Tensor | None = torch.empty(
                capacity,
                *w2_scale.shape[1:],
                dtype=w2_scale.dtype,
                device=cuda_device,
            )
        else:
            self._cpu_w13_scale = None
            self._cpu_w2_scale = None
            self._buf_w13_scale = None
            self._buf_w2_scale = None

        # LFRU state: {expert_id: [slot, freq, last_access_clock]}
        # Eviction score = freq / (clock - last_access + 1). Lower = evict first.
        self._lru: dict[int, list] = {}
        self._clock: int = 0
        self._free_slots: list[int] = list(range(capacity))

        # Expert map handed to the kernel: expert id to slot for the group
        # being evaluated, -1 for everything else. Rebuilt each prepare() --
        # it must expose exactly the requested group, not whatever else
        # happens to still be resident, or experts already summed in an
        # earlier group would be counted twice. Residency itself lives in
        # _lru; this is only the view the kernel gets. Built in a pinned host
        # mirror and uploaded once, rather than a transfer per entry.
        self._mapping: torch.Tensor = torch.full(
            (num_experts,), -1, dtype=torch.int32, device=cuda_device
        )
        self._mapping_host: torch.Tensor = torch.full(
            (num_experts,), -1, dtype=torch.int32
        ).pin_memory()

    @property
    def buf_w13(self) -> torch.Tensor:
        return self._buf_w13

    @property
    def buf_w2(self) -> torch.Tensor:
        return self._buf_w2

    @property
    def buf_w13_scale(self) -> torch.Tensor | None:
        return self._buf_w13_scale

    @property
    def buf_w2_scale(self) -> torch.Tensor | None:
        return self._buf_w2_scale

    def invalidate(self, expert_id: int) -> None:
        """Remove *expert_id* from the cache, returning its slot to the free
        list.  No-op if the expert is not currently cached."""
        if expert_id in self._lru:
            entry = self._lru.pop(expert_id)
            self._free_slots.append(entry[0])

    @torch.compiler.disable
    def plan_chunks(self, topk_ids: torch.Tensor) -> list[tuple[slice, list[int]]]:
        """Row ranges whose combined unique expert count fits the cache.

        Routing is per token, so evaluating a subset of rows and concatenating
        the results is equivalent to evaluating the whole batch -- and every
        token's sum still happens in a single kernel call, which is why this
        split reproduces the uncached output exactly.

        Returns a single full-width slice when the batch already fits, which is
        the common case, so callers pay nothing extra for it. Each slice comes
        with the expert ids it needs, computed from host data this method
        already has, so ``prepare()`` need not synchronize again per chunk.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs.

        Returns:
            ``(row slice, unique expert ids)`` pairs covering ``topk_ids``.

        Raises:
            RuntimeError: if one token alone routes to more experts than the
                cache can hold, which no amount of splitting can fix.
        """
        num_rows = topk_ids.size(0)
        if num_rows == 0:
            return [(slice(0, 0), [])]

        # The common case only needs the distinct ids, which the device can
        # reduce far more cheaply than transferring every row.
        unique = topk_ids.unique()
        if unique.numel() <= self.capacity:
            return [(slice(0, num_rows), unique.tolist())]

        rows = topk_ids.tolist()
        chunks: list[tuple[slice, list[int]]] = []
        start = 0
        seen: set[int] = set()
        for i, row in enumerate(rows):
            row_ids = set(row)
            if len(seen | row_ids) > self.capacity:
                if i == start:
                    raise RuntimeError(
                        f"CachedWeightProvider: one token routes to "
                        f"{len(row_ids)} experts but "
                        f"--moe-expert-cache-size={self.capacity}. "
                        f"Set --moe-expert-cache-size >= {len(row_ids)}."
                    )
                chunks.append((slice(start, i), sorted(seen)))
                start = i
                seen = row_ids
            else:
                seen |= row_ids
        chunks.append((slice(start, num_rows), sorted(seen)))
        return chunks

    @torch.compiler.disable
    def plan_expert_groups(self, topk_ids: torch.Tensor) -> list[list[int]]:
        """Split the forward's experts into groups the cache can hold at once.

        A token's output is the weighted sum of its experts' outputs, so the
        sum can be taken a few experts at a time and accumulated: run the
        kernel once per group with an ``expert_map`` that hides the others,
        then add the results. Every (token, expert) pair is still computed
        exactly once, in whichever group owns its expert.

        Cost is ``ceil(experts_used / capacity)`` kernel launches, independent
        of how many tokens are in the batch, and each expert is fetched at most
        once per forward. Splitting the token axis instead costs one launch per
        chunk -- one per token once capacity approaches ``top_k`` -- and
        refetches experts as the cache thrashes.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs.

        Returns:
            Groups of global expert ids, each no larger than ``capacity``. A
            single group when everything already fits, which is the common
            case.
        """
        unique = topk_ids.unique().tolist()
        if len(unique) <= self.capacity:
            return [unique]
        return [
            unique[i : i + self.capacity] for i in range(0, len(unique), self.capacity)
        ]

    @torch.compiler.disable
    def prepare(
        self, topk_ids: torch.Tensor, unique_ids: list[int] | None = None
    ) -> ExpertWeightResult:
        """Make a set of experts resident and return the map selecting them.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs. Only
                read when ``unique_ids`` is not supplied.
            unique_ids: The experts to make resident. Both planners already
                know this, and passing it avoids a device synchronization --
                which matters, because a split forward calls this once per
                piece.

        Returns:
            ExpertWeightResult holding the GPU buffers and an ``expert_map``
            exposing exactly these experts, everything else -1.

        Raises:
            RuntimeError: if more experts are requested than the cache holds.
        """
        if unique_ids is None:
            unique_ids = topk_ids.unique().tolist()
        if len(unique_ids) > self.capacity:
            raise RuntimeError(
                f"CachedWeightProvider: {len(unique_ids)} unique experts "
                f"requested but --moe-expert-cache-size={self.capacity}. "
                f"Set --moe-expert-cache-size >= {len(unique_ids)}."
            )

        # Experts requested here must never be evicted to make room for
        # another one in the same call -- their slot would be handed to a
        # different expert while they are still expected to be resident. A
        # freshly loaded expert has freq=1, exactly the lowest LFRU score, so
        # it is the first eviction candidate. The map built at the end reads
        # every requested expert back out of _lru, so violating this raises
        # rather than corrupting silently, but it must not happen at all.
        needed = set(unique_ids)

        for expert_id in unique_ids:
            if expert_id in self._lru:
                # Cache hit: update frequency and recency
                self._clock += 1
                entry = self._lru[expert_id]
                entry[1] += 1  # freq
                entry[2] = self._clock  # last access
                self.hits += 1
            else:
                # Cache miss: need to load expert
                if self._free_slots:
                    slot = self._free_slots.pop()
                else:
                    # Evict entry with lowest freq/age score
                    best_key = None
                    best_score = float("inf")
                    for k, (s, freq, last) in self._lru.items():
                        if k in needed:
                            continue
                        age = self._clock - last + 1
                        score = freq / age
                        if score < best_score:
                            best_score = score
                            best_key = k
                    # len(unique_ids) <= capacity is enforced above, so at least
                    # one cached expert is outside `needed` whenever the buffer
                    # is full and a miss remains to be served.
                    assert best_key is not None
                    slot = self._lru.pop(best_key)[0]

                # Copy expert weights from CPU to GPU slot
                self._buf_w13[slot].copy_(self._cpu_w13[expert_id], non_blocking=True)
                self._buf_w2[slot].copy_(self._cpu_w2[expert_id], non_blocking=True)
                if self._buf_w13_scale is not None:
                    assert self._cpu_w13_scale is not None
                    assert self._cpu_w2_scale is not None
                    assert self._buf_w2_scale is not None
                    self._buf_w13_scale[slot].copy_(
                        self._cpu_w13_scale[expert_id], non_blocking=True
                    )
                    self._buf_w2_scale[slot].copy_(
                        self._cpu_w2_scale[expert_id], non_blocking=True
                    )

                self._clock += 1
                self._lru[expert_id] = [slot, 1, self._clock]
                self.misses += 1

        total = self.hits + self.misses
        if total > 0:
            logger.debug(
                "Expert cache: %d hits, %d misses (%.1f%% hit rate)",
                self.hits,
                self.misses,
                100.0 * self.hits / total,
            )

        # Expose exactly this group. Blocking on purpose: the host mirror is
        # rewritten by the next group, so an async copy could still be reading
        # it when that happens.
        self._mapping_host.fill_(-1)
        for expert_id in unique_ids:
            self._mapping_host[expert_id] = self._lru[expert_id][0]
        self._mapping.copy_(self._mapping_host)

        return ExpertWeightResult(
            w1=self._buf_w13,
            w2=self._buf_w2,
            expert_map=self._mapping,
            w1_scale=self._buf_w13_scale,
            w2_scale=self._buf_w2_scale,
        )


def run_with_expert_cache(
    provider: CachedWeightProvider,
    topk_ids: torch.Tensor,
    run: Callable[[ExpertWeightResult, slice, bool], torch.Tensor],
) -> torch.Tensor:
    """Evaluate a MoE forward through the expert cache.

    ``run`` receives the resident weights with the ``expert_map`` selecting
    them, the rows it should evaluate, and whether it should include work that
    belongs to the forward as a whole rather than to this call -- shared
    experts, most importantly. It sees the original ``topk_ids``; the map is
    what restricts each call to the resident experts.

    The two splits differ only in what is cut. ``"token"`` cuts rows and
    concatenates, so each token's sum stays inside one kernel call and the
    result matches the uncached path exactly. ``"expert"`` cuts the expert set
    and sums, which costs far fewer launches when capacity is small but rounds
    each group's partial sum to the model dtype.

    When everything fits -- the common case for both splits -- ``run`` is
    called exactly once and its result returned untouched.
    """
    if provider.split == "expert":
        groups = provider.plan_expert_groups(topk_ids)
        if len(groups) == 1:
            return run(provider.prepare(topk_ids, groups[0]), _ALL_ROWS, True)

        # Accumulate in fp32. It does not recover what each group already lost
        # rounding to the model dtype, but it keeps the sum from losing more.
        accumulator: torch.Tensor | None = None
        out_dtype: torch.dtype | None = None
        for i, expert_ids in enumerate(groups):
            part = run(provider.prepare(topk_ids, expert_ids), _ALL_ROWS, i == 0)
            if accumulator is None:
                accumulator, out_dtype = part.float(), part.dtype
            else:
                accumulator += part.float()
        assert accumulator is not None and out_dtype is not None
        return accumulator.to(out_dtype)

    plan = provider.plan_chunks(topk_ids)
    if len(plan) == 1:
        rows, unique_ids = plan[0]
        return run(provider.prepare(topk_ids, unique_ids), rows, True)
    return torch.cat(
        [
            run(provider.prepare(topk_ids[rows], unique_ids), rows, True)
            for rows, unique_ids in plan
        ],
        dim=0,
    )
