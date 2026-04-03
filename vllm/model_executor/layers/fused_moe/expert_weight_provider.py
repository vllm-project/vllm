# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class ExpertWeightResult:
    """GPU-resident expert weights ready for kernel consumption."""

    w1: torch.Tensor
    w2: torch.Tensor
    topk_ids: torch.Tensor
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

    On each forward pass, prepare() identifies which experts are needed,
    copies any misses from CPU to GPU (evicting the lowest-scored entry
    when the buffer is full), and returns an ExpertWeightResult with
    remapped topk_ids whose values are GPU-buffer slot indices.
    """

    def __init__(
        self,
        capacity: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
    ) -> None:
        num_experts = w13_weight.size(0)

        self.capacity = capacity
        self._num_experts = num_experts
        self.hits = 0
        self.misses = 0
        self._overflow_warned = False

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
            self._cpu_w13_scale: torch.Tensor | None = w13_scale.cpu()
            self._cpu_w2_scale: torch.Tensor | None = w2_scale.cpu()
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

        # Persistent GPU mapping tensor: _mapping[expert_id] = slot.
        self._mapping: torch.Tensor = torch.zeros(
            num_experts, dtype=torch.int32, device=cuda_device
        )

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
    def prepare(self, topk_ids: torch.Tensor) -> ExpertWeightResult:
        """Populate the GPU buffer and return slot-remapped expert IDs.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs.

        Returns:
            ExpertWeightResult with remapped topk_ids and GPU buffer refs.

        Note:
            If unique experts exceed capacity (common during prefill),
            truncates to the last ``capacity`` experts and logs a warning.
        """
        unique_ids = topk_ids.unique().tolist()
        if len(unique_ids) > self.capacity:
            # Prefill overflow: more unique experts than cache slots.
            # Process only the last `capacity` experts (most likely to be
            # needed in upcoming decode steps). Warn once.
            if not self._overflow_warned:
                logger.warning(
                    "CachedWeightProvider.prepare() called with %d unique "
                    "experts but capacity is only %d. Truncating to last %d. "
                    "This is expected during prefill with large batches.",
                    len(unique_ids), self.capacity, self.capacity,
                )
                self._overflow_warned = True
            unique_ids = unique_ids[-self.capacity:]

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
                        age = self._clock - last + 1
                        score = freq / age
                        if score < best_score:
                            best_score = score
                            best_key = k
                    slot = self._lru.pop(best_key)[0]

                # Copy expert weights from CPU to GPU slot
                self._buf_w13[slot].copy_(self._cpu_w13[expert_id])
                self._buf_w2[slot].copy_(self._cpu_w2[expert_id])
                if self._buf_w13_scale is not None:
                    assert self._cpu_w13_scale is not None
                    assert self._cpu_w2_scale is not None
                    assert self._buf_w2_scale is not None
                    self._buf_w13_scale[slot].copy_(self._cpu_w13_scale[expert_id])
                    self._buf_w2_scale[slot].copy_(self._cpu_w2_scale[expert_id])

                self._clock += 1
                self._lru[expert_id] = [slot, 1, self._clock]
                self._mapping[expert_id] = slot
                self.misses += 1

        total = self.hits + self.misses
        if total > 0:
            logger.debug(
                "Expert cache: %d hits, %d misses (%.1f%% hit rate)",
                self.hits,
                self.misses,
                100.0 * self.hits / total,
            )

        remapped_ids = self._mapping[topk_ids.long()].to(dtype=topk_ids.dtype)

        return ExpertWeightResult(
            w1=self._buf_w13,
            w2=self._buf_w2,
            topk_ids=remapped_ids,
            w1_scale=self._buf_w13_scale,
            w2_scale=self._buf_w2_scale,
        )
