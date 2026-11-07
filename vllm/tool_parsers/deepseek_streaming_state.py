# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class DeepSeekStreamingTokenState:
    tool_calls_start_token_id: int
    tool_call_start_token_id: int | None
    tool_call_end_token_id: int | None
    tool_calls_started: bool = False
    tool_start_count: int = 0
    tool_end_count: int = 0
    initialized: bool = False

    def update(
        self,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> tuple[int, int, int, int] | None:
        """Return previous/current inner tool-tag counts after a delta."""
        if not self.tool_calls_started:
            start_in_delta = self.tool_calls_start_token_id in delta_token_ids
            if not self.initialized:
                self.initialized = True
                if (
                    not start_in_delta
                    and self.tool_calls_start_token_id not in current_token_ids
                ):
                    return None
            elif not start_in_delta:
                return None

            self.tool_calls_started = True
            previous_start_count = previous_token_ids.count(
                self.tool_call_start_token_id
            )
            previous_end_count = previous_token_ids.count(self.tool_call_end_token_id)
            self.tool_start_count = current_token_ids.count(
                self.tool_call_start_token_id
            )
            self.tool_end_count = current_token_ids.count(self.tool_call_end_token_id)
        else:
            previous_start_count = self.tool_start_count
            previous_end_count = self.tool_end_count
            self.tool_start_count += delta_token_ids.count(
                self.tool_call_start_token_id
            )
            self.tool_end_count += delta_token_ids.count(self.tool_call_end_token_id)

        return (
            previous_start_count,
            previous_end_count,
            self.tool_start_count,
            self.tool_end_count,
        )
