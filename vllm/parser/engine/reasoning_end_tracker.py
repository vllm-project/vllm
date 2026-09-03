# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-scoped reasoning-end tracking for parser engines."""

from collections.abc import Sequence


class ReasoningEndTracker:
    """Incrementally match token sequences that end reasoning."""

    def __init__(
        self,
        boundaries: tuple[tuple[int, ...], ...],
        reasoning_ended: bool,
        prefix_token_ids: Sequence[int] = (),
    ) -> None:
        self._boundaries = boundaries
        self._reasoning_ended = reasoning_ended
        self._matched_prefixes = self._suffix_prefix_lengths(prefix_token_ids)

    @property
    def reasoning_ended(self) -> bool:
        return self._reasoning_ended

    def _suffix_prefix_lengths(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        prefixes = []
        for boundary in self._boundaries:
            max_length = min(len(token_ids), len(boundary) - 1)
            for length in range(max_length, 0, -1):
                if tuple(token_ids[-length:]) == boundary[:length]:
                    prefixes.append(length)
                    break
            else:
                prefixes.append(0)
        return tuple(prefixes)

    def _scan(
        self,
        token_ids: Sequence[int],
        matched_prefixes: tuple[int, ...],
    ) -> tuple[int | None, tuple[int, ...]]:
        prefixes = list(matched_prefixes)
        for offset, token_id in enumerate(token_ids):
            for index, boundary in enumerate(self._boundaries):
                matched = prefixes[index]
                candidate = (*boundary[:matched], token_id)
                if candidate == boundary:
                    return offset, tuple(prefixes)

                next_matched = min(len(candidate), len(boundary) - 1)
                while (
                    next_matched
                    and candidate[-next_matched:] != boundary[:next_matched]
                ):
                    next_matched -= 1
                prefixes[index] = next_matched
        return None, tuple(prefixes)

    def preview(self, token_ids: Sequence[int]) -> int | None:
        if self._reasoning_ended:
            return None
        offset, _ = self._scan(token_ids, self._matched_prefixes)
        return offset

    def commit(self, token_ids: Sequence[int]) -> int | None:
        if self._reasoning_ended:
            return None
        offset, prefixes = self._scan(token_ids, self._matched_prefixes)
        if offset is not None:
            self._reasoning_ended = True
        else:
            self._matched_prefixes = prefixes
        return offset
