# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GDNHeadPartition:
    tp_size: int
    tp_rank: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    k_start: int
    k_count: int
    v_start: int
    v_count: int
    max_k_count: int
    max_v_count: int

    @property
    def v_per_k(self) -> int:
        return self.num_v_heads // self.num_k_heads

    @property
    def local_key_dim(self) -> int:
        return self.k_count * self.head_k_dim

    @property
    def local_value_dim(self) -> int:
        return self.v_count * self.head_v_dim

    @property
    def padded_key_dim(self) -> int:
        return self.max_k_count * self.head_k_dim

    @property
    def padded_value_dim(self) -> int:
        return self.max_v_count * self.head_v_dim

    @property
    def local_conv_dim(self) -> int:
        return self.local_key_dim * 2 + self.local_value_dim

    @property
    def padded_conv_dim(self) -> int:
        return self.padded_key_dim * 2 + self.padded_value_dim

    @property
    def padded_qkvz_output_sizes(self) -> list[int]:
        padded_key_global = self.padded_key_dim * self.tp_size
        padded_value_global = self.padded_value_dim * self.tp_size
        return [
            padded_key_global,
            padded_key_global,
            padded_value_global,
            padded_value_global,
        ]

    @property
    def padded_ba_output_sizes(self) -> list[int]:
        padded_value_heads_global = self.max_v_count * self.tp_size
        return [padded_value_heads_global, padded_value_heads_global]


def _balanced_range(total: int, parts: int, rank: int) -> tuple[int, int]:
    base = total // parts
    extra = total % parts
    count = base + (1 if rank < extra else 0)
    start = rank * base + min(rank, extra)
    return start, count


def make_gdn_head_partition(
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    tp_size: int,
    tp_rank: int,
) -> GDNHeadPartition:
    if num_v_heads % num_k_heads != 0:
        raise ValueError(
            "GDN non-uniform TP requires an integral value/key head ratio. "
            f"Got {num_v_heads=} and {num_k_heads=}."
        )
    if not (0 <= tp_rank < tp_size):
        raise ValueError(f"Invalid TP rank {tp_rank} for TP size {tp_size}.")

    v_per_k = num_v_heads // num_k_heads
    k_start, k_count = _balanced_range(num_k_heads, tp_size, tp_rank)
    max_k_count = (num_k_heads + tp_size - 1) // tp_size
    v_start = k_start * v_per_k
    v_count = k_count * v_per_k
    max_v_count = max_k_count * v_per_k

    return GDNHeadPartition(
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        k_start=k_start,
        k_count=k_count,
        v_start=v_start,
        v_count=v_count,
        max_k_count=max_k_count,
        max_v_count=max_v_count,
    )


def explicit_gdn_conv_weight_loader(
    partition: GDNHeadPartition,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """Load packed [q, k, v] conv rows using GDN head-group spans."""

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        param.data.zero_()
        key_dim = partition.num_k_heads * partition.head_k_dim
        value_offset = key_dim * 2
        specs = [
            (
                partition.k_start * partition.head_k_dim,
                partition.local_key_dim,
                0,
            ),
            (
                key_dim + partition.k_start * partition.head_k_dim,
                partition.local_key_dim,
                partition.padded_key_dim,
            ),
            (
                value_offset + partition.v_start * partition.head_v_dim,
                partition.local_value_dim,
                partition.padded_key_dim * 2,
            ),
        ]
        for source_start, copy_size, dest_start in specs:
            if copy_size == 0:
                continue
            param.data[dest_start : dest_start + copy_size, ...].copy_(
                loaded_weight[source_start : source_start + copy_size, ...]
            )

    return loader


def explicit_vector_weight_loader(
    start: int,
    size: int,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        param.data.zero_()
        if size == 0:
            return
        param.data[:size].copy_(loaded_weight[start : start + size])

    return loader
