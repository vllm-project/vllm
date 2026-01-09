# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from dataclasses import dataclass
from typing import Any

from vllm.v1.engine import EngineCoreEvent, EngineCoreOutput
from vllm.v1.outputs import IterStats

NONES = itertools.repeat(None)


@dataclass
class Event:
    name: str
    timestamp: float
    attributes: dict[str, Any] | None = None


@dataclass
class ObservableContext:
    token_related_events: list[Event]
    engine_core_events: list[EngineCoreEvent]
    num_cached_tokens: int = 0
    not_empty: bool = False

    @classmethod
    def from_new_request(cls):
        return cls(
            token_related_events=[],
            engine_core_events=[],
        )

    def _update_iter_stats(
        self, iter_stats: IterStats, new_token_ids: list[int]
    ) -> None:
        if not new_token_ids:
            return
        self.not_empty = True

        self.token_related_events.append(
            Event(
                timestamp=iter_stats.token_scheduled_time,
                name="token_scheduled",
                attributes={
                    "iter_batch_size": iter_stats.iter_batch_size,
                    "iter_waiting_size": iter_stats.iter_waiting_size,
                },
            )
        )

        self.token_related_events.append(
            Event(
                timestamp=iter_stats.token_output_time,
                name="token_generated",
                attributes={
                    "new_token_ids": new_token_ids,
                    "iter_waiting_size": iter_stats.iter_waiting_size,
                    "iter_total_tokens_count": iter_stats.iter_total_tokens_count,
                },
            )
        )

        if self.num_cached_tokens is None:
            self.num_cached_tokens = iter_stats.num_cached_tokens

    def _update_events(self, events: list[EngineCoreEvent]) -> None:
        self.engine_core_events.extend(events)

    def update_from_output(self, output: EngineCoreOutput) -> None:
        if output.iter_stats is not None:
            self._update_iter_stats(output.iter_stats, output.new_token_ids)
        if output.events is not None:
            self._update_events(output.events)
