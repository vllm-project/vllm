# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import regex as re


class DynamicSpeculativeDecodingManager:
    """Chooses speculative-token counts from a batch-size schedule."""

    _RANGE_PATTERN = re.compile(r"^\s*(\d+)(?:\s*-\s*(\d+))?\s*$")
    _optimal_num_speculative_tokens: dict[int, int]

    def __init__(
        self,
        num_speculative_tokens_per_batch_size: dict[str, int] | None,
        vllm_max_batch_size: int,
        vllm_num_speculative_tokens: int,
    ):
        self.vllm_max_batch_size = vllm_max_batch_size
        self.vllm_num_speculative_tokens = vllm_num_speculative_tokens

        if num_speculative_tokens_per_batch_size is None:
            raise ValueError(
                "num_speculative_tokens_per_batch_size is required for "
                "dynamic speculative decoding."
            )
        if vllm_max_batch_size <= 0:
            raise ValueError("vllm_max_batch_size must be > 0.")
        if vllm_num_speculative_tokens <= 0:
            raise ValueError("vllm_num_speculative_tokens must be > 0.")

        self._schedule = self._parse_schedule(num_speculative_tokens_per_batch_size)
        self._optimal_num_speculative_tokens = self._build_dense_schedule()

    def step(self, batch_size: int) -> int:
        return self.get_optimal_num_speculative_tokens(batch_size)

    def get_optimal_num_speculative_tokens(self, batch_size: int) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if batch_size > self.vllm_max_batch_size:
            raise ValueError("batch_size must be <= vllm_max_batch_size.")
        return min(
            self.vllm_num_speculative_tokens,
            self._optimal_num_speculative_tokens[batch_size],
        )

    def _parse_schedule(self, schedule: dict[str, int]) -> list[tuple[int, int, int]]:
        """Validate and normalize schedule ranges into sorted integer tuples.

        Accepts keys like ``"16"`` or ``"1-16"`` and returns
        ``(range_start, range_end, num_speculative_tokens)`` entries.
        """
        parsed_schedule: list[tuple[int, int, int]] = []

        for batch_size_range, num_speculative_tokens in schedule.items():
            match = self._RANGE_PATTERN.fullmatch(str(batch_size_range))
            if match is None:
                raise ValueError(
                    "Invalid batch-size range "
                    f"{batch_size_range!r}. Expected 'N' or 'N-M'."
                )

            range_start = int(match.group(1))
            range_end = int(match.group(2) or match.group(1))

            if range_start <= 0 or range_end <= 0:
                raise ValueError(
                    f"Batch-size range {batch_size_range!r} must be positive."
                )
            if range_start > range_end:
                raise ValueError(
                    f"Batch-size range start must be <= end for {batch_size_range!r}."
                )
            if num_speculative_tokens < 0:
                raise ValueError(
                    "num_speculative_tokens_per_batch_size values must be >= 0."
                )
            parsed_schedule.append((range_start, range_end, num_speculative_tokens))

        if not parsed_schedule:
            raise ValueError("num_speculative_tokens_per_batch_size must not be empty.")

        parsed_schedule.sort(key=lambda entry: entry[0])

        previous_end = 0
        for range_start, range_end, _ in parsed_schedule:
            if range_start <= previous_end:
                raise ValueError(
                    "Batch-size ranges must be non-overlapping and sorted."
                )
            previous_end = range_end

        first_range_start = parsed_schedule[0][0]
        if first_range_start != 1:
            raise ValueError(
                "The first batch-size range must start at 1 so every runtime "
                "batch size has a defined schedule."
            )

        return parsed_schedule

    def _build_dense_schedule(self) -> dict[int, int]:
        """Expand parsed ranges into per-batch-size values up to the runtime max.

        Gaps between configured ranges inherit the previous K so lookups can use
        a dense ``batch_size -> num_speculative_tokens`` map at runtime. For
        example, ``{"1-16": 3, "32-128": 2}`` maps batch sizes ``17-31`` to
        ``3``.
        """
        dense_schedule: dict[int, int] = {}
        next_batch_size = 1
        last_num_speculative_tokens: int | None = None

        for range_start, range_end, num_speculative_tokens in self._schedule:
            if (
                range_start > next_batch_size
                and last_num_speculative_tokens is not None
            ):
                for batch_size in range(
                    next_batch_size,
                    min(range_start, self.vllm_max_batch_size + 1),
                ):
                    dense_schedule[batch_size] = last_num_speculative_tokens

            for batch_size in range(
                max(range_start, next_batch_size),
                min(range_end, self.vllm_max_batch_size) + 1,
            ):
                dense_schedule[batch_size] = num_speculative_tokens

            next_batch_size = max(next_batch_size, range_end + 1)
            last_num_speculative_tokens = num_speculative_tokens

            if next_batch_size > self.vllm_max_batch_size:
                break

        if last_num_speculative_tokens is None:
            raise ValueError(
                "num_speculative_tokens_per_batch_size must contain at least "
                "one valid batch-size range."
            )
        for batch_size in range(next_batch_size, self.vllm_max_batch_size + 1):
            dense_schedule[batch_size] = last_num_speculative_tokens

        return dense_schedule
