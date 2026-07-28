# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Learned expert-usage heat map persistence (.vllm_expert_usage)."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)


class ExpertUsageStore:
    """Records which experts a workload routes to and persists heat maps."""

    def __init__(self, path: str | None = None):
        self.path = Path(path) if path else None
        # (layer_id, expert_id) -> count
        self._counts: dict[tuple[int, int], int] = defaultdict(int)
        self._dirty = False
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        try:
            data = json.loads(self.path.read_text())
            for key, count in data.get("counts", {}).items():
                layer_s, expert_s = key.split(":")
                self._counts[(int(layer_s), int(expert_s))] = int(count)
            logger.info(
                "Loaded expert usage heat map from %s (%d entries)",
                self.path,
                len(self._counts),
            )
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to load expert usage from %s: %s", self.path, e)

    def record(self, layer_id: int, expert_ids: list[int]) -> None:
        for e in expert_ids:
            if e < 0:
                continue
            self._counts[(layer_id, e)] += 1
            self._dirty = True

    def hottest(
        self, layer_id: int, limit: int, num_experts: int
    ) -> list[int]:
        """Return up to ``limit`` hottest expert ids for a layer."""
        scored = [
            (self._counts.get((layer_id, e), 0), e) for e in range(num_experts)
        ]
        scored.sort(reverse=True)
        hot = [e for count, e in scored[:limit] if count > 0]
        return hot or list(range(min(limit, num_experts)))

    def flush(self) -> None:
        if not self._dirty or self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "counts": {
                    f"{layer}:{expert}": count
                    for (layer, expert), count in self._counts.items()
                }
            }
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, self.path)
            self._dirty = False
        except OSError as e:
            logger.warning("Failed to flush expert usage to %s: %s", self.path, e)


def default_usage_path(disk_path: str | None, model_path: str | None) -> str | None:
    """Resolve default usage file path."""
    if disk_path:
        return str(Path(disk_path) / ".vllm_expert_usage")
    if model_path:
        return str(Path(model_path) / ".vllm_expert_usage")
    return None
