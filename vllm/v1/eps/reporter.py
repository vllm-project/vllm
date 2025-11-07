# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ABOUTME: Emits EPS counters to log files and aggregates totals.
# ABOUTME: Simple JSONL-based reporting for EigenPage Summaries metrics.

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from vllm.v1.eps.telemetry import EpsStepCounters


class EpsReporter:
    def __init__(self, jsonl_path: str | None = None) -> None:
        self._jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._logger = logging.getLogger("vllm.eps")
        self._totals: dict[str, float] = {}

    def add_step(
        self,
        request_ids: Sequence[str],
        counters: EpsStepCounters,
    ) -> None:
        payload = {
            "ts": time.time(),
            "requests": list(request_ids),
            "eps": asdict(counters),
        }

        for key, value in payload["eps"].items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            self._totals[key] = self._totals.get(key, 0.0) + numeric

        if self._jsonl_path:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._logger.debug("Writing EPS metrics to %s", self._jsonl_path)
            with self._jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        else:
            self._logger.info("EPS counters: %s", payload)

    @property
    def totals(self) -> dict[str, float]:
        return dict(self._totals)


class EpsAggregator:
    """Aggregates EPS counters over multiple decode steps."""

    def __init__(self) -> None:
        self.steps = 0
        self.cumulative = defaultdict(float)

    def ingest(self, counters: EpsStepCounters) -> None:
        self.steps += 1
        for key in [
            "pages_total",
            "pages_visited",
            "pages_skipped",
            "pages_unique_total",
            "pages_unique_kept",
            "blocks_total",
            "blocks_kept",
            "groups_total",
            "groups_kept",
            "kv_bytes_total",
            "kv_bytes_kept",
            "eps_prepass_ms",
            "decode_ms",
        ]:
            self.cumulative[key] += float(getattr(counters, key, 0.0))

    def snapshot(self) -> dict[str, float]:
        steps = max(1, self.steps)
        pages_total = max(1.0, self.cumulative["pages_total"])
        bytes_total = max(1.0, self.cumulative["kv_bytes_total"])
        return {
            "eps.pages_kept_ratio": self.cumulative["pages_visited"] / pages_total,
            "eps.pages_skipped_ratio": self.cumulative["pages_skipped"] / pages_total,
            "eps.bytes_kept_ratio": self.cumulative["kv_bytes_kept"] / bytes_total,
            "eps.prepass_ms_avg": self.cumulative["eps_prepass_ms"] / steps,
            "eps.decode_ms_avg": self.cumulative["decode_ms"] / steps,
        }
