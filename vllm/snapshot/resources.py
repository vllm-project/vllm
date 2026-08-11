# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

WeightResourcePolicy = Literal["cuda_image", "host_backup", "discard"]
KVResourcePolicy = Literal["cuda_image", "discard"]
RuntimeResourcePolicy = Literal["cuda_image"]


class SnapshotResourcePolicyWire(TypedDict):
    weights: WeightResourcePolicy
    kv: KVResourcePolicy
    runtime: RuntimeResourcePolicy


@dataclass(frozen=True)
class SnapshotResourcePolicy:
    weights: WeightResourcePolicy
    kv: KVResourcePolicy
    runtime: RuntimeResourcePolicy

    @property
    def requires_allocator(self) -> bool:
        return self.weights != "cuda_image" or self.kv == "discard"

    def to_wire(self) -> SnapshotResourcePolicyWire:
        return {
            "weights": self.weights,
            "kv": self.kv,
            "runtime": self.runtime,
        }


_RESOURCE_POLICIES = {
    "full": SnapshotResourcePolicy(
        weights="cuda_image", kv="cuda_image", runtime="cuda_image"
    ),
    "discard_kv": SnapshotResourcePolicy(
        weights="cuda_image", kv="discard", runtime="cuda_image"
    ),
    "l1_prepared": SnapshotResourcePolicy(
        weights="host_backup", kv="discard", runtime="cuda_image"
    ),
    "l2_prepared": SnapshotResourcePolicy(
        weights="discard", kv="discard", runtime="cuda_image"
    ),
}


def parse_snapshot_resource_policy(name: str) -> SnapshotResourcePolicy:
    try:
        return _RESOURCE_POLICIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown snapshot resource policy: {name}") from exc


def decode_snapshot_resource_policy(
    value: Mapping[str, object],
) -> SnapshotResourcePolicy:
    expected_fields = {"weights", "kv", "runtime"}
    if set(value) != expected_fields or not all(
        isinstance(value[field], str) for field in expected_fields
    ):
        raise ValueError("invalid snapshot resource policy")

    policy = SnapshotResourcePolicy(
        weights=cast(WeightResourcePolicy, value["weights"]),
        kv=cast(KVResourcePolicy, value["kv"]),
        runtime=cast(RuntimeResourcePolicy, value["runtime"]),
    )
    if policy not in _RESOURCE_POLICIES.values():
        raise ValueError("unsupported snapshot resource policy")
    return policy
