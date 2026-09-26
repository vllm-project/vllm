# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.engine.coordinator import _is_stale_engine_stats


@pytest.mark.parametrize(
    ("last_stats", "stats_wave", "stats_step", "expected"),
    [
        (None, 1, 10, False),
        ((1, 10), 1, 11, False),
        ((1, 10), 1, 10, False),
        ((1, 100), 2, 0, False),
        ((1, 10), 1, 9, True),
        ((2, 0), 1, 100, True),
    ],
)
def test_is_stale_engine_stats(
    last_stats: tuple[int, int] | None,
    stats_wave: int,
    stats_step: int,
    expected: bool,
) -> None:
    assert _is_stale_engine_stats(last_stats, stats_wave, stats_step) is expected


def test_delayed_stats_from_another_engine_are_not_stale() -> None:
    last_engine_stats: list[tuple[int, int] | None] = [None, None]

    last_engine_stats[0] = (1, 11)

    assert not _is_stale_engine_stats(last_engine_stats[1], 1, 10)
