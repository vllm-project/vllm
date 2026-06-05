# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline-parallel layer map helpers for NIXL metadata."""

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)


@dataclass(frozen=True)
class PPLayerMap:
    pp_size: int
    boundaries: tuple[tuple[int, int], ...]
    registered_layer_indices: tuple[tuple[int, ...], ...]
    total_num_hidden_layers: int

    @classmethod
    def from_metadata_shards(
        cls, metas: list[NixlAgentMetadata], total_num_hidden_layers: int
    ) -> "PPLayerMap":
        assert metas, "cannot build PPLayerMap without metadata shards"
        pp_size = metas[0].pp_size

        boundaries_by_rank: dict[int, tuple[int, int]] = {}
        registered_by_rank: dict[int, tuple[int, ...]] = {}
        for meta in metas:
            boundary = (meta.start_layer, meta.end_layer)
            registered_layers = tuple(meta.registered_layer_indices)
            prior_boundary = boundaries_by_rank.get(meta.pp_rank)
            if prior_boundary is not None:
                if (
                    prior_boundary != boundary
                    or registered_by_rank[meta.pp_rank] != registered_layers
                ):
                    raise ValueError(
                        f"conflicting metadata shards for pp_rank {meta.pp_rank}"
                    )
                continue
            boundaries_by_rank[meta.pp_rank] = boundary
            registered_by_rank[meta.pp_rank] = registered_layers

        assert len(boundaries_by_rank) == pp_size, (
            "missing metadata shards for pp_rank(s): "
            f"{sorted(set(range(pp_size)) - boundaries_by_rank.keys())}"
        )

        expected_start = 0
        for pp_rank in range(pp_size):
            start, end = boundaries_by_rank[pp_rank]
            if start != expected_start:
                raise ValueError(
                    "PP layer boundaries must be contiguous; pp_rank "
                    f"{pp_rank} starts at {start}, expected {expected_start}"
                )
            expected_start = end
        if expected_start != total_num_hidden_layers:
            raise ValueError(
                "PP layer boundaries must cover all hidden layers; last end "
                f"{expected_start}, expected {total_num_hidden_layers}"
            )

        return cls(
            pp_size=pp_size,
            boundaries=tuple(boundaries_by_rank[rank] for rank in range(pp_size)),
            registered_layer_indices=tuple(
                registered_by_rank[rank] for rank in range(pp_size)
            ),
            total_num_hidden_layers=total_num_hidden_layers,
        )
