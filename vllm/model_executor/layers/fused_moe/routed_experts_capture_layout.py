# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from vllm.distributed.parallel_state import GroupCoordinator


@dataclass(frozen=True)
class RoutedExpertsCaptureLayout:
    """Map physical routing rows to this DP rank's token order.
    Built per forward/microbatch; metadata stays fixed across graph replay."""

    # Ordered (start, length) spans after gathering, excluding interior padding.
    source_ranges: tuple[tuple[int, int], ...]
    # Physical rows on this rank before gathering.
    input_rows: int
    # All group members participate, even when source_ranges is empty.
    gather_group: GroupCoordinator | None = None
    # Sizes in group-rank order, identical across peers; the backend must support
    # these sizes, including uneven and empty shards.
    shard_sizes: tuple[int, ...] = ()
    num_rows: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_rows", sum(n for _, n in self.source_ranges))
        if self.input_rows < 0:
            raise ValueError("capture input_rows must be non-negative")
        if self.gather_group is not None:
            if len(self.shard_sizes) != self.gather_group.world_size:
                raise ValueError("capture shard_sizes must match gather group size")
            if any(size < 0 for size in self.shard_sizes):
                raise ValueError("capture shard_sizes must be non-negative")
            if self.input_rows != self.shard_sizes[self.gather_group.rank_in_group]:
                raise ValueError("capture local rows do not match shard_sizes")
            available_rows = sum(self.shard_sizes)
        else:
            if self.shard_sizes:
                raise ValueError("capture shard_sizes require a gather group")
            available_rows = self.input_rows
        if any(
            start < 0 or length < 0 or start + length > available_rows
            for start, length in self.source_ranges
        ):
            raise ValueError("capture source range exceeds available rows")


@runtime_checkable
class RoutedExpertsCaptureLayoutProvider(Protocol):
    """Describe routing rows before each forward, without running collectives.
    The runner activates its context, including backend metadata, first."""

    def get_routed_experts_capture_layouts(
        self, num_tokens_across_dp: tuple[int, ...], dp_rank: int
    ) -> dict[int, RoutedExpertsCaptureLayout]: ...


def prepare_routed_experts_capture_layouts(
    model: nn.Module,
    num_tokens: int,
    num_tokens_across_dp: torch.Tensor | None = None,
    dp_rank: int = 0,
) -> dict[int, RoutedExpertsCaptureLayout] | None:
    """Collect and validate the model's per-layer layouts for this forward.
    The caller must activate its forward/microbatch context first."""
    providers: tuple[RoutedExpertsCaptureLayoutProvider, ...] | None = getattr(
        model, "routed_experts_capture_layout_providers", None
    )
    if providers is None:
        return None
    counts = (
        (num_tokens,)
        if num_tokens_across_dp is None
        else tuple(num_tokens_across_dp.tolist())
    )
    if any(n < 0 for n in counts):
        raise ValueError("capture token counts must be non-negative")
    if not 0 <= dp_rank < len(counts) or counts[dp_rank] != num_tokens:
        raise ValueError("capture local count disagrees with DP metadata")
    layouts: dict[int, RoutedExpertsCaptureLayout] = {}
    for provider in providers:
        provider_layouts = provider.get_routed_experts_capture_layouts(counts, dp_rank)
        for layer_id, layout in provider_layouts.items():
            if layer_id in layouts:
                raise ValueError(f"Duplicate capture layout for layer {layer_id}")
            layouts[layer_id] = layout
    if missing := model.routed_experts_capture_layers - layouts.keys():
        raise ValueError(f"Missing routed-experts capture layouts: {missing}")
    return layouts


def make_routed_experts_capture_layout(
    num_tokens_across_dp: tuple[int, ...],
    dp_rank: int,
    *,
    sp_size: int = 1,
    is_gathered: bool = False,
) -> RoutedExpertsCaptureLayout:
    """Build one DP-major / evenly padded SP layout for a built-in provider."""
    if sp_size < 1 or any(n < 0 for n in num_tokens_across_dp):
        raise ValueError("capture requires positive SP size and non-negative counts")
    local_rows = num_tokens_across_dp[dp_rank]
    shard_rows = tuple((n + sp_size - 1) // sp_size for n in num_tokens_across_dp)
    if is_gathered:
        # DP-major, then SP rank; padding follows each DP rank's valid rows.
        return RoutedExpertsCaptureLayout(
            ((sum(shard_rows[:dp_rank]) * sp_size, local_rows),),
            sum(shard_rows) * sp_size,
        )
    if sp_size == 1:
        return RoutedExpertsCaptureLayout(((0, local_rows),), local_rows)
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    if group.world_size != sp_size:
        raise ValueError("capture SP shards must match the TP group")
    return RoutedExpertsCaptureLayout(
        ((0, local_rows),),
        shard_rows[dp_rank],
        group,
        (shard_rows[dp_rank],) * sp_size,
    )
