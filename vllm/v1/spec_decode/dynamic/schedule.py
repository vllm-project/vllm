# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Schedule computation shared by the sweep tuner and the Dynamic SD controller.

Pure functions, no vLLM imports, so both build tables with the same code and
the rules are unit-testable in CI:

    tokens_per_step(K) = 1 + p * sum_{i<=K} a_i   a_i = P(first i drafts accepted)
    goodput(B, K)      = B * tokens_per_step(K) / step_time(B, K)
    K*(B)              = argmax_K goodput(B, K), then two rules against switching:
      1. ties keep the current K (another K must be more than `eps` better);
      2. a range narrower than `min_width_frac` of its start batch size merges
         into the neighbour that loses the least goodput.

`p` is the proposal rate, the share of decode steps with drafts: 1 for
autoregressive drafters, lower for n-gram. Step time stays the profiled time
at the full K even when p < 1.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

EPS = 0.005  # rule 1: change K only if another K is >0.5% better
MIN_WIDTH_FRAC = 0.25  # rule 2: a range must span >=25% of its start batch size
PROPOSAL_RATE_FULL = 0.98  # at or above: treat as 1 (counting noise)

Costs = dict[int, tuple[list[int], list[float]]]  # K -> (batch sizes, step ms)
Schedule = list[list[int]]  # [[start_bs, end_bs, K], ...]


def interp(xs: list[int], ys: list[float], x: float) -> float:
    """Linear interpolation; flat below the first point, slope >= 0 above the
    last (step time does not fall as the batch grows)."""
    if x <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    if len(xs) < 2:
        return ys[-1]
    slope = max(ys[-1] - ys[-2], 0.0) / (xs[-1] - xs[-2])
    return ys[-1] + slope * (x - xs[-1])


def step_ms(costs: Costs, k: int, b: float) -> float:
    """Profiled step time at batch size b for K."""
    return interp(*costs[k], b)


def tokens_per_step(
    accept_by_pos: list[float], proposal_rate: float = 1.0
) -> dict[int, float]:
    """Expected output tokens per verification step for K = 0..len(accept_by_pos)."""
    p = min(proposal_rate, 1.0)
    return {k: 1 + p * sum(accept_by_pos[:k]) for k in range(len(accept_by_pos) + 1)}


def goodput_table(
    costs: Costs, tps: dict[int, float], max_b: int, ks: list[int] | None = None
) -> dict[int, dict[int, float]]:
    """goodput[B][K] in output tokens/s for B = 1..max_b."""
    ks = sorted(k for k in costs if k in tps) if ks is None else ks
    if not ks:
        raise ValueError(
            "no K value has both a cost profile and an acceptance estimate"
        )
    return {
        b: {k: b * tps[k] / (step_ms(costs, k, b) / 1000) for k in ks}
        for b in range(1, max_b + 1)
    }


def compress(best: dict[int, int]) -> Schedule:
    """{B: K} -> [[start, end, K], ...] with runs of equal K merged."""
    ranges: Schedule = []
    for b in sorted(best):
        if ranges and ranges[-1][2] == best[b]:
            ranges[-1][1] = b
        else:
            ranges.append([b, b, best[b]])
    return ranges


def expand(schedule: Schedule, max_b: int) -> list[int]:
    """[[start, end, K], ...] -> dense lookup[B] = K (index 0 unused).

    Gaps and the tail carry the previous K forward, like vLLM's
    build_dynamic_sd_schedule_lookup.
    """
    if not schedule:
        raise ValueError("empty schedule")
    lookup = [0] * (max_b + 1)
    k = schedule[0][2]
    for b in range(1, max_b + 1):
        for s, e, kk in sorted(schedule):
            if s <= b <= e:
                k = kk
                break
        lookup[b] = k
    return lookup


def build_schedule(
    goodput: dict[int, dict[int, float]],
    eps: float = EPS,
    min_width_frac: float = MIN_WIDTH_FRAC,
) -> dict[int, int]:
    """Best K per batch size. Rule 1 (eps): keep the current K unless another
    is > eps better. Rule 2 (min_width_frac): merge a range narrower than
    min_width_frac of its start batch size into the neighbour that loses the
    least goodput."""
    best: dict[int, int] = {}
    prev = None
    for b in sorted(goodput):
        top = max(goodput[b], key=lambda k: goodput[b][k])
        if prev is not None and goodput[b][prev] >= goodput[b][top] * (1 - eps):
            best[b] = prev
        else:
            best[b] = top
        prev = best[b]
    while True:
        ranges = compress(best)
        narrow = [
            i
            for i, (s, e, _) in enumerate(ranges)
            if len(ranges) > 1 and e - s + 1 < min_width_frac * s
        ]
        if not narrow:
            break
        i = narrow[0]
        s, e, _ = ranges[i]
        neighbours = [ranges[j][2] for j in (i - 1, i + 1) if 0 <= j < len(ranges)]
        pick = max(
            neighbours, key=lambda n: sum(goodput[b][n] for b in range(s, e + 1))
        )
        for b in range(s, e + 1):
            best[b] = pick
    return best


def make_schedule(
    costs: Costs,
    accept_by_pos: list[float],
    max_b: int,
    allowed_ks: list[int] | None = None,
    eps: float = EPS,
    min_width_frac: float = MIN_WIDTH_FRAC,
    proposal_rate: float = 1.0,
) -> Schedule:
    """End to end: costs + acceptance -> schedule. `allowed_ks` restricts the
    choice to K values that have captured CUDA graphs."""
    if not (isinstance(max_b, int) and max_b >= 1):
        raise ValueError(f"max batch size must be an integer >= 1, got {max_b!r}")
    validate_costs(costs)
    validate_acceptance(accept_by_pos, proposal_rate)
    tps = tokens_per_step(accept_by_pos, proposal_rate=proposal_rate)
    ks = sorted(
        k for k in costs if k in tps and (allowed_ks is None or k in allowed_ks)
    )
    if not ks:
        raise ValueError(
            "no K value has both a cost profile and an acceptance estimate"
        )
    return compress(
        build_schedule(goodput_table(costs, tps, max_b, ks), eps, min_width_frac)
    )


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_costs(costs: Costs) -> None:
    """Raise ValueError unless every K has increasing batch sizes and positive,
    finite step times."""
    if not costs:
        raise ValueError("no step-cost profiles")
    for k, (xs, ys) in costs.items():
        where = f"step costs for K={k}"
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise ValueError(f"{where}: K must be an integer >= 0")
        if not xs or len(xs) != len(ys):
            raise ValueError(
                f"{where}: need as many step times as batch sizes, and at least one"
            )
        if not all(
            isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in xs
        ):
            raise ValueError(f"{where}: batch sizes must be integers >= 1, got {xs}")
        if any(x1 <= x0 for x0, x1 in zip(xs, xs[1:])):
            raise ValueError(
                f"{where}: batch sizes must be strictly increasing, got {xs}"
            )
        if not all(_finite(y) and y > 0 for y in ys):
            raise ValueError(f"{where}: step times must be finite and > 0 ms, got {ys}")


def validate_acceptance(accept_by_pos: list[float], proposal_rate: float = 1.0) -> None:
    """Raise ValueError unless acceptance is a list of probabilities and the
    proposal rate is a positive number."""
    if not isinstance(accept_by_pos, (list, tuple)):
        raise ValueError(
            f"acceptance by position must be a list, got {accept_by_pos!r}"
        )
    if not all(_finite(a) and 0.0 <= a <= 1.0 for a in accept_by_pos):
        raise ValueError(
            f"acceptance by position must be in [0, 1], got {accept_by_pos}"
        )
    if not (_finite(proposal_rate) and proposal_rate > 0):
        raise ValueError(f"proposal rate must be > 0, got {proposal_rate!r}")


def load_costs(pattern: str) -> Costs:
    """Read costs_k*.json files written by `vllm bench sweep tune`.

    Rows are sorted by batch size. Malformed files, two files for the same K,
    and repeated batch sizes raise ValueError naming the file.
    """
    costs: Costs = {}
    origin: dict[int, str] = {}
    for f in sorted(glob.glob(pattern)):
        try:
            rows = sorted(
                json.loads(Path(f).read_text())["rows"], key=lambda r: r["batch_size"]
            )
            if not rows:
                raise ValueError("no rows")
            ks = {r["k"] for r in rows}
            if len(ks) != 1:
                raise ValueError(f"rows for several K values {sorted(ks)}")
            k = rows[0]["k"]
            entry = ([r["batch_size"] for r in rows], [r["step_ms"] for r in rows])
            validate_costs({k: entry})
        except (
            ValueError,
            KeyError,
            TypeError,
        ) as e:  # JSONDecodeError is a ValueError
            detail = f"missing field {e}" if isinstance(e, KeyError) else str(e)
            raise ValueError(f"bad cost file {f}: {detail}") from e
        if k in costs:
            raise ValueError(f"cost files {origin[k]} and {f} are both for K={k}")
        costs[k], origin[k] = entry, f
    if not costs:
        raise FileNotFoundError(f"no cost files match {pattern!r}")
    return costs


def load_acceptance(path: str | Path) -> tuple[list[float], float]:
    """(per-draft acceptance by position, proposal rate) from the tuner's
    acceptance.json. A malformed file raises ValueError."""
    try:
        acc = json.loads(Path(path).read_text())
        accept, rate = acc["accept_by_pos"], acc.get("proposal_rate", 1.0)
        validate_acceptance(accept, rate)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        detail = f"missing field {e}" if isinstance(e, KeyError) else str(e)
        raise ValueError(f"bad acceptance file {path}: {detail}") from e
    rate = float(rate)
    return list(accept), (1.0 if rate >= PROPOSAL_RATE_FULL else rate)
