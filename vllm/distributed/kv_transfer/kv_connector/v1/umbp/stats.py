# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared UMBP connector transfer statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats


@dataclass
class UMBPStoreConnectorStats(KVConnectorStats):
    """Serializable interval counters shared by every UMBP runtime."""

    data: dict[str, Any] = field(default_factory=dict)

    def record(
        self,
        operation: str,
        *,
        submitted: int = 0,
        completed: int = 0,
        failed: int = 0,
        num_bytes: int = 0,
    ) -> None:
        entry = self.data.setdefault(
            operation,
            {
                "submitted": 0,
                "completed": 0,
                "failed": 0,
                "num_bytes": 0,
            },
        )
        entry["submitted"] += submitted
        entry["completed"] += completed
        entry["failed"] += failed
        entry["num_bytes"] += num_bytes

    def reset(self) -> None:
        self.data.clear()

    def aggregate(self, other: KVConnectorStats) -> "UMBPStoreConnectorStats":
        if not isinstance(other, UMBPStoreConnectorStats):
            raise TypeError("cannot aggregate incompatible UMBP stats")
        result = UMBPStoreConnectorStats()
        for operation, values in [*self.data.items(), *other.data.items()]:
            entry = result.data.setdefault(
                operation,
                {
                    "submitted": 0,
                    "completed": 0,
                    "failed": 0,
                    "num_bytes": 0,
                },
            )
            for key in entry:
                entry[key] += values.get(key, 0)
        return result

    def reduce(self) -> dict[str, int | float]:
        result: dict[str, int | float] = {}
        for operation, values in self.data.items():
            for key, value in values.items():
                result[f"{operation}_{key}"] = value
        return result

    def is_empty(self) -> bool:
        return not self.data
