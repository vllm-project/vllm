# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import gc
import time
from collections.abc import Callable

from vllm.logprobs import FlatLogprobs, Logprob


def make_flat_logprobs(num_positions: int, num_logprobs: int) -> FlatLogprobs:
    flat = FlatLogprobs()
    for pos in range(num_positions):
        start = len(flat.logprobs)
        count = num_logprobs + 1
        flat.start_indices.append(start)
        flat.end_indices.append(start + count)
        flat.token_ids.extend(pos * 1000 + i for i in range(count))
        flat.logprobs.extend(-(i + 1.0) for i in range(count))
        flat.ranks.extend([0] + list(range(1, count)))
        flat.decoded_tokens.extend(f"tok-{pos}-{i}" for i in range(count))
    return flat


def old_slice(flat: FlatLogprobs, index: slice) -> FlatLogprobs:
    min_index = flat.start_indices[index][0]
    max_index = flat.end_indices[index][-1]
    return FlatLogprobs(
        start_indices=[i - min_index for i in flat.start_indices[index]],
        end_indices=[i - min_index for i in flat.end_indices[index]],
        token_ids=flat.token_ids[min_index:max_index],
        logprobs=flat.logprobs[min_index:max_index],
        ranks=flat.ranks[min_index:max_index],
        decoded_tokens=flat.decoded_tokens[min_index:max_index],
    )


def old_getitem(flat: FlatLogprobs, index: int | slice):
    if isinstance(index, int):
        return {
            flat.token_ids[i]: Logprob(
                logprob=flat.logprobs[i],
                rank=flat.ranks[i],
                decoded_token=flat.decoded_tokens[i],
            )
            for i in range(flat.start_indices[index], flat.end_indices[index])
        }
    if isinstance(index, slice):
        return old_slice(flat, index)
    raise TypeError(f"Invalid index type: {type(index)}")


def assert_equal(left: FlatLogprobs, right: FlatLogprobs) -> None:
    assert left.start_indices == right.start_indices
    assert left.end_indices == right.end_indices
    assert left.token_ids == right.token_ids
    assert left.logprobs == right.logprobs
    assert left.ranks == right.ranks
    assert left.decoded_tokens == right.decoded_tokens


def assert_correctness(flat: FlatLogprobs) -> None:
    cases = [
        slice(-1, None),
        slice(-5, None),
        slice(1, None),
        slice(None, None),
        slice(2, 20, 2),
        slice(None, None, -1),
    ]
    for index in cases:
        assert_equal(old_slice(flat, index), flat[index])

    for index in (slice(0, 0), slice(len(flat), len(flat))):
        old_error = None
        new_error = None
        try:
            old_slice(flat, index)
        except Exception as exc:
            old_error = type(exc)
        try:
            flat[index]
        except Exception as exc:
            new_error = type(exc)
        assert old_error is new_error

    old_error = None
    new_error = None
    empty = FlatLogprobs()
    try:
        old_slice(empty, slice(-1, None))
    except Exception as exc:
        old_error = type(exc)
    try:
        empty[-1:]
    except Exception as exc:
        new_error = type(exc)
    assert old_error is new_error


def measure(fn: Callable[[], object], repeats: int, trials: int) -> float:
    for _ in range(min(repeats, 1000)):
        fn()

    timings: list[float] = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(trials):
            start = time.perf_counter()
            for _ in range(repeats):
                fn()
            timings.append((time.perf_counter() - start) * 1e6 / repeats)
    finally:
        if was_enabled:
            gc.enable()
    return min(timings)


def bench_pair(
    old_fn: Callable[[], object],
    new_fn: Callable[[], object],
    repeats: int,
    trials: int,
) -> tuple[float, float]:
    return (
        measure(old_fn, repeats, trials),
        measure(new_fn, repeats, trials),
    )


def bench_case(
    flat: FlatLogprobs,
    index: slice,
    repeats: int,
    trials: int,
) -> tuple[float, float]:
    return bench_pair(
        lambda: old_getitem(flat, index),
        lambda: flat[index],
        repeats,
        trials,
    )


def old_delta_logprobs(logprobs: FlatLogprobs, token_ids: list[int]):
    return old_getitem(logprobs, slice(-len(token_ids), None))


def new_delta_logprobs(logprobs: FlatLogprobs, token_ids: list[int]):
    return logprobs[-len(token_ids) :]


def bench_delta_selector(
    logprobs: FlatLogprobs,
    token_ids: list[int],
    repeats: int,
    trials: int,
) -> tuple[float, float]:
    old_result = old_delta_logprobs(logprobs, token_ids)
    new_result = new_delta_logprobs(logprobs, token_ids)
    assert isinstance(old_result, FlatLogprobs)
    assert isinstance(new_result, FlatLogprobs)
    assert_equal(old_result, new_result)

    return bench_pair(
        lambda: old_delta_logprobs(logprobs, token_ids),
        lambda: new_delta_logprobs(logprobs, token_ids),
        repeats,
        trials,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=2000)
    parser.add_argument("--num-logprobs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20000)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--include-broad-slices", action="store_true")
    args = parser.parse_args()

    flat = make_flat_logprobs(args.positions, args.num_logprobs)
    assert_correctness(flat)

    scenarios = [
        ("last_5", slice(-5, None)),
        ("last_20", slice(-20, None)),
    ]
    if args.include_broad_slices:
        scenarios.extend(
            [
                ("prompt_tail", slice(1, None)),
                ("full", slice(None, None)),
                ("step_2", slice(2, 200, 2)),
            ]
        )

    print("correctness: old and new FlatLogprobs slices match")
    print(f"timing: best of {args.trials} trials, {args.repeats} repeats each")
    print("case         old_us  new_us  speedup")
    print("-----------  ------  ------  -------")
    old_us, new_us = bench_case(flat, slice(-1, None), args.repeats, args.trials)
    print(f"{'last_1':11s}  {old_us:6.2f}  {new_us:6.2f}  "
          f"{old_us / new_us:7.2f}x")
    for name, index in scenarios:
        old_us, new_us = bench_case(flat, index, args.repeats, args.trials)
        print(f"{name:11s}  {old_us:6.2f}  {new_us:6.2f}  "
              f"{old_us / new_us:7.2f}x")

    selector_scenarios = [
        ("flat_1", flat, [1]),
        ("flat_5", flat, [1, 2, 3, 4, 5]),
    ]

    print()
    print("flat delta selection")
    print("case         old_us  new_us  speedup")
    print("-----------  ------  ------  -------")
    for name, logprobs, token_ids in selector_scenarios:
        old_us, new_us = bench_delta_selector(
            logprobs, token_ids, args.repeats, args.trials
        )
        print(f"{name:11s}  {old_us:6.2f}  {new_us:6.2f}  "
              f"{old_us / new_us:7.2f}x")


if __name__ == "__main__":
    main()
