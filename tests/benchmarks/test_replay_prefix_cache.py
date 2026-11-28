# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest

from benchmarks.multi_turn.replay_prefix_cache import replay


@pytest.mark.parametrize("seed", [1, 7, 35])
def test_replay_matches_explicit_published_prefixes(seed):
    rng = random.Random(seed)
    requests = []
    histories: list[list[int]] = [[]]
    for _ in range(100):
        history = rng.choice(histories)
        prompt = history[: rng.randint(0, len(history))] + [rng.randrange(3)]
        output = [rng.randrange(3) for _ in range(rng.randrange(16))]
        histories.append(prompt + output)
        requests.append((prompt, output))

    alignments = {1, 2, 4, 6, 12}
    count, tokens, hits, saved = replay(requests, alignments)
    assert count == len(requests)
    assert tokens == sum(len(prompt) for prompt, _ in requests)
    for size in alignments:
        published: set[tuple[int, ...]] = set()
        expected = []
        for prompt, output in requests:
            expected.append(
                max(
                    (
                        end
                        for end in range(size, len(prompt), size)
                        if tuple(prompt[:end]) in published
                    ),
                    default=0,
                )
            )
            computed = prompt + output[:-1]
            published.update(
                tuple(computed[:end]) for end in range(size, len(computed) + 1, size)
            )
        assert hits[size] == sum(n > 0 for n in expected)
        assert saved[size] == sum(expected)
