# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

DynamicSDSchedule = list[tuple[int, int, int]]


def validate_and_normalize_dynamic_sd_schedule(
    num_speculative_tokens_per_batch_size: object,
) -> DynamicSDSchedule:
    """Validate and normalize a Dynamic SD batch-size schedule.

    The schedule is expressed as a list of inclusive ranges:

    ``[(range_start, range_end, num_speculative_tokens), ...]``
    """
    if num_speculative_tokens_per_batch_size is None:
        raise ValueError(
            "num_speculative_tokens_per_batch_size is required for "
            "dynamic speculative decoding."
        )
    if not isinstance(num_speculative_tokens_per_batch_size, list):
        raise ValueError(
            "num_speculative_tokens_per_batch_size must be a non-empty list of "
            "(range_start, range_end, num_speculative_tokens) entries."
        )
    if not num_speculative_tokens_per_batch_size:
        raise ValueError("num_speculative_tokens_per_batch_size must not be empty.")

    parsed_schedule: DynamicSDSchedule = []
    for entry in num_speculative_tokens_per_batch_size:
        if not isinstance(entry, list | tuple) or len(entry) != 3:
            raise ValueError(
                "Each num_speculative_tokens_per_batch_size entry must be a "
                "3-item sequence: (range_start, range_end, num_speculative_tokens)."
            )

        range_start, range_end, num_speculative_tokens = (
            int(entry[0]),
            int(entry[1]),
            int(entry[2]),
        )

        if range_start <= 0 or range_end <= 0:
            raise ValueError(
                f"Batch-size range ({range_start}, {range_end}) must be positive."
            )
        if range_start > range_end:
            raise ValueError(
                "Batch-size range start must be <= end for "
                f"({range_start}, {range_end}, {num_speculative_tokens})."
            )
        if num_speculative_tokens < 0:
            raise ValueError(
                "num_speculative_tokens_per_batch_size values must be >= 0."
            )

        parsed_schedule.append((range_start, range_end, num_speculative_tokens))

    parsed_schedule.sort(key=lambda entry: entry[0])

    previous_end = 0
    for range_start, range_end, _ in parsed_schedule:
        if range_start <= previous_end:
            raise ValueError("Batch-size ranges must be non-overlapping and sorted.")
        previous_end = range_end

    first_range_start = parsed_schedule[0][0]
    if first_range_start != 1:
        raise ValueError(
            "The first batch-size range must start at 1 so every runtime "
            "batch size has a defined schedule."
        )

    return parsed_schedule


def build_dynamic_sd_schedule_lookup(
    num_speculative_tokens_per_batch_size: object,
    vllm_max_batch_size: int,
    vllm_num_speculative_tokens: int,
) -> list[int]:
    """Expand the configured schedule into a dense batch_size -> K lookup.

    "dense_schedule" means a 1-indexed lookup table where index ``batch_size``
    stores the exact K to use for that runtime batch size. This lets the
    scheduler do a simple array lookup instead of searching the configured
    ranges on every scheduling step.
    """
    if vllm_max_batch_size <= 0:
        raise ValueError("vllm_max_batch_size must be > 0.")
    if vllm_num_speculative_tokens <= 0:
        raise ValueError("vllm_num_speculative_tokens must be > 0.")

    parsed_schedule = validate_and_normalize_dynamic_sd_schedule(
        num_speculative_tokens_per_batch_size
    )

    # Index 0 is intentionally unused so that valid runtime batch sizes can be
    # looked up directly as dense_schedule[batch_size].
    dense_schedule = [0] * (vllm_max_batch_size + 1)
    next_batch_size = 1
    last_num_speculative_tokens: int | None = None

    for range_start, range_end, num_speculative_tokens in parsed_schedule:
        if range_start > next_batch_size and last_num_speculative_tokens is not None:
            # Fill any gap before the next configured range by carrying forward
            # the previous K. For example, [(1, 16, 3), (32, 128, 2)] should map
            # batch sizes 17-31 to K=3.
            for batch_size in range(
                next_batch_size,
                min(range_start, vllm_max_batch_size + 1),
            ):
                dense_schedule[batch_size] = min(
                    vllm_num_speculative_tokens,
                    last_num_speculative_tokens,
                )

        # Fill the current configured inclusive range with its K value.
        for batch_size in range(
            max(range_start, next_batch_size),
            min(range_end, vllm_max_batch_size) + 1,
        ):
            dense_schedule[batch_size] = min(
                vllm_num_speculative_tokens,
                num_speculative_tokens,
            )

        next_batch_size = max(next_batch_size, range_end + 1)
        last_num_speculative_tokens = num_speculative_tokens

        if next_batch_size > vllm_max_batch_size:
            break

    if last_num_speculative_tokens is None:
        raise ValueError(
            "num_speculative_tokens_per_batch_size must contain at least "
            "one valid batch-size range."
        )

    # Fill the tail after the final configured range by carrying forward the
    # last K through vllm_max_batch_size.
    for batch_size in range(next_batch_size, vllm_max_batch_size + 1):
        dense_schedule[batch_size] = min(
            vllm_num_speculative_tokens,
            last_num_speculative_tokens,
        )

    return dense_schedule
