# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
import torch

from vllm.model_executor.model_loader.sharded_weight import (
    ShardedWeightRequest,
    ShardedWeightSpec,
    ShardedWeightTarget,
    resolve_sharded_weight_target,
)


def _target(
    *,
    semantic_id: str = "test.weight.v1",
    local_shape: tuple[int, ...] = (2, 3),
    consume: Any = None,
) -> ShardedWeightTarget:
    if consume is None:
        consume = lambda _: True
    return ShardedWeightTarget(
        spec=ShardedWeightSpec(
            semantic_id=semantic_id,
            dtype=torch.float16,
            shard_dim=0,
            local_shape=local_shape,
            shard_index=0,
            num_shards=2,
        ),
        retention_key=semantic_id,
        consume=consume,
    )


class _Provider(torch.nn.Module):
    def __init__(self, target: ShardedWeightTarget | None):
        super().__init__()
        self.target = target
        self.calls: list[str] = []

    def resolve_sharded_weight_target(
        self,
        relative_name: str,
        request: ShardedWeightRequest,
    ) -> ShardedWeightTarget | None:
        self.calls.append(relative_name)
        return self.target


def test_resolver_uses_nearest_provider_and_validates_before_loading():
    consumed: list[torch.Tensor] = []

    def consume(weight: torch.Tensor) -> bool:
        consumed.append(weight)
        return True

    root = _Provider(_target(semantic_id="root"))
    child_target = _target(consume=consume)
    root.child = _Provider(child_target)
    request = ShardedWeightRequest(
        name="child.weight",
        dtype=torch.float16,
        global_shape=(4, 3),
    )

    resolved = resolve_sharded_weight_target(root, request)

    assert resolved is child_target
    assert root.child.calls == ["weight"]
    assert root.calls == []
    with pytest.raises(ValueError, match="local shard shape"):
        resolved.load(torch.empty((1, 3), dtype=torch.float16))
    assert resolved.load(torch.ones((2, 3), dtype=torch.float16))
    assert len(consumed) == 1


def test_resolver_rejects_invalid_targets_and_supports_aliases():
    request = ShardedWeightRequest(
        name="block.weight",
        dtype=torch.float16,
        global_shape=(4, 3),
    )
    invalid = torch.nn.Module()
    invalid.block = _Provider(_target(local_shape=(3, 3)))
    with pytest.raises(ValueError, match="incompatible local shape"):
        resolve_sharded_weight_target(invalid, request)

    target = _target()
    no_retention = torch.nn.Module()
    no_retention.block = _Provider(
        ShardedWeightTarget(
            spec=target.spec,
            retention_key=None,
            consume=target.consume,
        )
    )
    with pytest.raises(ValueError, match="has no retention key"):
        resolve_sharded_weight_target(no_retention, request)

    aliased = _Provider(_target(semantic_id="alias"))
    aliases = torch.nn.Module()
    aliases.canonical = aliased
    aliases.block = aliased
    assert resolve_sharded_weight_target(aliases, request) is aliased.target
    assert aliased.calls == ["weight"]

    with pytest.raises(ValueError, match="retention_group_size"):
        ShardedWeightTarget(
            spec=target.spec,
            retention_key="invalid",
            consume=lambda _: True,
            retention_group_size=0,
        )
