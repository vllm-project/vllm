# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import NamedTuple

_CTX_INFINITY = 2**31 - 1

DynamicSDEntry = tuple[int, int, int, int, int]
DynamicSDSchedule = list[DynamicSDEntry]


class DynamicSDLookup(NamedTuple):
    """Dense ``(batch_size, context_bucket) -> K`` lookup with context-axis
    bucket boundaries."""

    dense: list[list[int]]
    ctx_boundaries: list[tuple[int, int]]

    def bucket_of(self, ctx_value: int) -> int:
        for i, (_, hi) in enumerate(self.ctx_boundaries):
            if ctx_value <= hi:
                return i
        return len(self.ctx_boundaries) - 1


def validate_and_normalize_dynamic_sd_schedule(
    num_speculative_tokens_per_batch_size: object,
) -> DynamicSDSchedule:
    """Validate and normalize a Dynamic SD schedule.

    Accepts either 3-item ``(bs_lo, bs_hi, K)`` entries (context-independent)
    or 5-item ``(bs_lo, bs_hi, ctx_lo, ctx_hi, K)`` entries forming a
    rectangular ``(batch_size x context)`` grid. The two forms cannot be
    mixed in a single schedule.
    """
    if num_speculative_tokens_per_batch_size is None:
        raise ValueError(
            "num_speculative_tokens_per_batch_size is required for "
            "dynamic speculative decoding."
        )
    if not isinstance(num_speculative_tokens_per_batch_size, list):
        raise ValueError(
            "num_speculative_tokens_per_batch_size must be a non-empty list "
            "of 3-item or 5-item entries."
        )
    if not num_speculative_tokens_per_batch_size:
        raise ValueError("num_speculative_tokens_per_batch_size must not be empty.")

    arities: set[int] = set()
    parsed: list[DynamicSDEntry] = []
    for entry in num_speculative_tokens_per_batch_size:
        if not isinstance(entry, list | tuple) or len(entry) not in (3, 5):
            raise ValueError(
                "Each num_speculative_tokens_per_batch_size entry must be a "
                "3-item sequence (bs_lo, bs_hi, num_speculative_tokens) or a "
                "5-item sequence "
                "(bs_lo, bs_hi, ctx_lo, ctx_hi, num_speculative_tokens)."
            )
        arities.add(len(entry))
        if len(entry) == 3:
            bs_lo, bs_hi, k = int(entry[0]), int(entry[1]), int(entry[2])
            ctx_lo, ctx_hi = 1, _CTX_INFINITY
        else:
            bs_lo, bs_hi, ctx_lo, ctx_hi, k = (
                int(entry[0]),
                int(entry[1]),
                int(entry[2]),
                int(entry[3]),
                int(entry[4]),
            )

        if bs_lo <= 0 or bs_hi <= 0:
            raise ValueError(
                f"Batch-size range ({bs_lo}, {bs_hi}) must be positive."
            )
        if bs_lo > bs_hi:
            raise ValueError(
                "Batch-size range start must be <= end for "
                f"({bs_lo}, {bs_hi}, ..., {k})."
            )
        if ctx_lo <= 0 or ctx_hi <= 0:
            raise ValueError(
                f"Context range ({ctx_lo}, {ctx_hi}) must be positive."
            )
        if ctx_lo > ctx_hi:
            raise ValueError(
                "Context range start must be <= end for "
                f"({bs_lo}, {bs_hi}, {ctx_lo}, {ctx_hi}, {k})."
            )
        if k < 0:
            raise ValueError(
                "num_speculative_tokens_per_batch_size values must be >= 0."
            )

        parsed.append((bs_lo, bs_hi, ctx_lo, ctx_hi, k))

    if len(arities) > 1:
        raise ValueError(
            "num_speculative_tokens_per_batch_size entries must all be "
            "3-item or all 5-item; mixing the two forms is not supported."
        )

    bs_ranges = sorted({(bs_lo, bs_hi) for bs_lo, bs_hi, *_ in parsed})
    prev_bs_end = 0
    for bs_lo, bs_hi in bs_ranges:
        if bs_lo <= prev_bs_end:
            raise ValueError(
                "Batch-size ranges must be non-overlapping and sorted."
            )
        prev_bs_end = bs_hi
    if bs_ranges[0][0] != 1:
        raise ValueError(
            "The first batch-size range must start at 1 so every runtime "
            "batch size has a defined schedule."
        )

    ctx_ranges = sorted({(ctx_lo, ctx_hi) for _, _, ctx_lo, ctx_hi, _ in parsed})

    expected_cells = len(bs_ranges) * len(ctx_ranges)
    if len(parsed) != expected_cells:
        raise ValueError(
            "num_speculative_tokens_per_batch_size must form a rectangular "
            f"(batch_size x context) grid: got {len(parsed)} entries but "
            f"expected {expected_cells} = {len(bs_ranges)} bs ranges x "
            f"{len(ctx_ranges)} context ranges. Every batch-size range must "
            "share the same set of context boundaries."
        )

    if ctx_ranges[0][0] != 1:
        raise ValueError(
            "The first context range must start at 1 so every runtime "
            "context length has a defined bucket."
        )
    for i in range(1, len(ctx_ranges)):
        if ctx_ranges[i][0] != ctx_ranges[i - 1][1] + 1:
            raise ValueError(
                "Context ranges must be contiguous with no gaps; "
                f"({ctx_ranges[i - 1][0]}, {ctx_ranges[i - 1][1]}) is "
                f"followed by ({ctx_ranges[i][0]}, {ctx_ranges[i][1]})."
            )
    last_ctx_lo, _ = ctx_ranges[-1]
    ctx_ranges[-1] = (last_ctx_lo, _CTX_INFINITY)

    ctx_hi_by_lo = {lo: hi for lo, hi in ctx_ranges}
    normalized: DynamicSDSchedule = []
    seen: set[tuple[int, int, int]] = set()
    for bs_lo, bs_hi, ctx_lo, _, k in parsed:
        key = (bs_lo, bs_hi, ctx_lo)
        if key in seen:
            raise ValueError(
                f"Duplicate schedule cell for (bs=[{bs_lo},{bs_hi}], "
                f"ctx_lo={ctx_lo})."
            )
        seen.add(key)
        normalized.append((bs_lo, bs_hi, ctx_lo, ctx_hi_by_lo[ctx_lo], k))

    normalized.sort(key=lambda e: (e[0], e[2]))
    return normalized


def build_dynamic_sd_schedule_lookup(
    num_speculative_tokens_per_batch_size: object,
    vllm_max_batch_size: int,
    vllm_num_speculative_tokens: int,
) -> DynamicSDLookup:
    """Expand the configured schedule into a dense ``(bs, ctx) -> K`` lookup.

    Index 0 on the batch-size axis is intentionally unused so runtime batch
    sizes look up directly. Legacy 3-item schedules produce a single-bucket
    lookup where every ``dense[bs]`` row has length 1.
    """
    if vllm_max_batch_size <= 0:
        raise ValueError("vllm_max_batch_size must be > 0.")
    if vllm_num_speculative_tokens <= 0:
        raise ValueError("vllm_num_speculative_tokens must be > 0.")

    normalized = validate_and_normalize_dynamic_sd_schedule(
        num_speculative_tokens_per_batch_size
    )

    ctx_boundaries = sorted(
        {(ctx_lo, ctx_hi) for _, _, ctx_lo, ctx_hi, _ in normalized}
    )
    ctx_index = {ctx_range: i for i, ctx_range in enumerate(ctx_boundaries)}
    num_ctx = len(ctx_boundaries)

    per_ctx: list[list[tuple[int, int, int]]] = [[] for _ in range(num_ctx)]
    for bs_lo, bs_hi, ctx_lo, ctx_hi, k in normalized:
        per_ctx[ctx_index[(ctx_lo, ctx_hi)]].append((bs_lo, bs_hi, k))

    dense: list[list[int]] = [
        [0] * num_ctx for _ in range(vllm_max_batch_size + 1)
    ]

    for c, entries in enumerate(per_ctx):
        entries.sort(key=lambda e: e[0])
        next_bs = 1
        last_k: int | None = None
        for bs_lo, bs_hi, k in entries:
            if bs_lo > next_bs and last_k is not None:
                for bs in range(next_bs, min(bs_lo, vllm_max_batch_size + 1)):
                    dense[bs][c] = min(vllm_num_speculative_tokens, last_k)
            for bs in range(
                max(bs_lo, next_bs),
                min(bs_hi, vllm_max_batch_size) + 1,
            ):
                dense[bs][c] = min(vllm_num_speculative_tokens, k)
            next_bs = max(next_bs, bs_hi + 1)
            last_k = k
            if next_bs > vllm_max_batch_size:
                break
        if last_k is not None:
            for bs in range(next_bs, vllm_max_batch_size + 1):
                dense[bs][c] = min(vllm_num_speculative_tokens, last_k)

    return DynamicSDLookup(dense=dense, ctx_boundaries=ctx_boundaries)
