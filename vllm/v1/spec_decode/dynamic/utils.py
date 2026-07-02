# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger

logger = init_logger(__name__)

DynamicSDSchedule = list[tuple[int, int, int]]

# Prefix-survival prior seeding dynamic-SD derivation until realized
# acceptance is measured (shape from DSpark-V4-Flash/Qwen3 measurements).
DEFAULT_SURVIVAL_PRIOR = [0.78, 0.55, 0.37, 0.24, 0.16, 0.11, 0.08]


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


def derive_dynamic_sd_schedule(
    r_grid: list[int],
    times_by_l: list[list[float]],
    survival: list[float],
    max_num_reqs: int,
    overhead_c0: float = 0.003,
    overhead_c1: float = 5e-5,
) -> DynamicSDSchedule:
    """Derive a batch-size -> K schedule from a profiled cost table.

    ``times_by_l[k][i]`` is the step time at ``r_grid[i]`` requests for
    verification width ``1+k``; ``survival`` is the per-position prefix
    acceptance probability (a startup prior, or measured live). For each
    request bucket pick the K maximizing expected tokens/sec, then collapse
    adjacent equal-K buckets into inclusive ranges covering 1..max_num_reqs.
    """
    gamma = len(times_by_l) - 1
    accept = [0.0]
    run = 0.0
    for s in survival[:gamma]:
        run += s
        accept.append(run)
    while len(accept) < gamma + 1:
        accept.append(accept[-1])
    table: list[list[int]] = []
    start = 1
    for i, r in enumerate(r_grid):
        best_k, best = gamma, -1.0
        for k in range(gamma + 1):
            t = times_by_l[k][i]
            if t <= 0.0:
                continue
            tokens = r * (1.0 + accept[k])
            tput = tokens / (t + overhead_c0 + overhead_c1 * tokens)
            if tput > best:
                best, best_k = tput, k
        end = r if i < len(r_grid) - 1 else max_num_reqs
        if table and table[-1][2] == best_k:
            table[-1][1] = end
        else:
            table.append([start, end, best_k])
        start = end + 1
    return [tuple(row) for row in table]


class DSparkLiveRederivation:
    """Re-derives the dynamic-SD schedule from realized acceptance.

    The worker's startup cost profile is static (hardware); the acceptance
    curve is measured from live traffic, so the batch-size -> K schedule
    tracks the actual workload with no prior/tuning dependence.
    """

    # Re-derive after this many observed drafts, updating positions with at
    # least this many observations.
    REDERIVE_DRAFTS = 16384
    MIN_POSITION_OBS = 512.0

    def __init__(
        self,
        r_grid: list[int],
        times_by_l: list[list[float]],
        max_num_seqs: int,
        num_spec_tokens: int,
    ) -> None:
        gamma = len(times_by_l) - 1
        self._cost_profile = (r_grid, times_by_l)
        self._max_num_seqs = max_num_seqs
        self._num_spec_tokens = num_spec_tokens
        self._acc_counts = [0.0] * gamma
        self._obs_counts = [0.0] * gamma
        self._rederive_ct = 0
        self._rederived_once = False
        self._survival = list(DEFAULT_SURVIVAL_PRIOR[:gamma])
        while len(self._survival) < gamma:
            self._survival.append(self._survival[-1] * 0.7)

    def observe(self, num_draft_tokens: int, num_accepted: int) -> list[int] | None:
        """Account one verified draft; returns a new dense lookup on re-derive."""
        obs, acc = self._obs_counts, self._acc_counts
        gamma = len(obs)
        for j in range(min(num_draft_tokens, gamma)):
            obs[j] += 1.0
        for j in range(min(num_accepted, gamma)):
            acc[j] += 1.0
        self._rederive_ct += 1
        if self._rederive_ct < self.REDERIVE_DRAFTS:
            return None
        self._rederive_ct = 0
        for j in range(gamma):
            if obs[j] >= self.MIN_POSITION_OBS:
                self._survival[j] = acc[j] / obs[j]
            # Decay so the estimate keeps tracking workload drift.
            obs[j] *= 0.5
            acc[j] *= 0.5
        r_grid, times_by_l = self._cost_profile
        table = derive_dynamic_sd_schedule(
            r_grid, times_by_l, self._survival, self._max_num_seqs
        )
        log = logger.debug if self._rederived_once else logger.info
        self._rederived_once = True
        log(
            "DSpark dynamic-SD re-derived from live acceptance %s: %s",
            [round(s, 2) for s in self._survival],
            table,
        )
        return build_dynamic_sd_schedule_lookup(
            table,
            vllm_max_batch_size=self._max_num_seqs,
            vllm_num_speculative_tokens=self._num_spec_tokens,
        )
