# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The topology seedable: a daemon reports its group's reachable endpoints."""

from typing import TYPE_CHECKING, Any

from vllm.model_executor.model_loader.weight_cache.protocol import (
    dataclass_to_json,
)
from vllm.model_executor.model_loader.weight_cache.transport import Seedable

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        WeightCacheKey,
    )


class ReplicaTopology(Seedable):
    """Every rank's dialable D2D address, gathered on the source's world group."""

    kind = "topology"

    def __init__(self, world_size: int, is_draft: bool, cache_key: "WeightCacheKey"):
        self._world_size = world_size
        self._is_draft = is_draft
        self._cache_key = cache_key
        self._ranks: dict[int, tuple[str, int]] | None = None

    def set_ranks(self, mapping: dict[int, tuple[str, int]]) -> None:
        """Record the group's global-rank -> (host, port) endpoints."""
        self._ranks = dict(mapping)

    def resolve(self, request: dict[str, Any]) -> dict[str, Any]:
        """The full replica topology, once the listener map was gathered."""
        if self._ranks is None:
            raise ValueError(
                "This daemon has no remote listener or has not gathered its "
                "group's topology"
            )
        return {
            "world_size": self._world_size,
            "is_draft": self._is_draft,
            "cache_config": dataclass_to_json(self._cache_key),
            "ranks": {
                str(rank): [host, port] for rank, (host, port) in self._ranks.items()
            },
        }

    def request_params(self, ctx: Any) -> dict[str, Any]:
        """The whole topology needs no key fields."""
        return {}

    def validate_payload(self, payload: Any) -> dict[str, Any]:
        """Structural checks on a fetched replica topology."""
        if not isinstance(payload, dict):
            raise ValueError(f"Topology payload is not a dict: {type(payload)}")
        world_size = payload.get("world_size")
        if (
            not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or world_size <= 0
        ):
            raise ValueError(f"Topology world_size is not a positive int: {world_size}")
        is_draft = payload.get("is_draft")
        if not isinstance(is_draft, bool):
            raise ValueError(f"Topology is_draft is not a bool: {is_draft}")
        if not isinstance(payload.get("cache_config"), dict):
            raise ValueError("Topology cache_config is not a dict")
        ranks = payload.get("ranks")
        if not isinstance(ranks, dict):
            raise ValueError(f"Topology ranks is not a dict: {type(ranks)}")
        for rank, endpoint in ranks.items():
            if not isinstance(rank, str) or not rank.isdigit():
                raise ValueError(f"Topology rank key is not a numeric string: {rank}")
            if (
                not isinstance(endpoint, (list, tuple))
                or len(endpoint) != 2
                or not isinstance(endpoint[0], str)
                or not isinstance(endpoint[1], int)
                or isinstance(endpoint[1], bool)
            ):
                raise ValueError(
                    f"Topology rank {rank} endpoint is not [host, port]: {endpoint}"
                )
        return payload
