# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The reference consumer against a real `KVCacheManager`, on CPU.

The engine tests in this directory need a GPU and a model. These feed the
events of the scheduler's own cache manager to the reference consumer and
check, before every request, that the consumer predicts the hit the manager
finds. A change to what `BlockStored`, `BlockRemoved` or `AllBlocksCleared`
mean therefore shows up without a GPU.
"""

from collections.abc import Iterator

import pytest

from tests.v1.core.test_prefix_caching import (
    make_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash

from .reference_consumer import ReferenceIndex, expected_cached_tokens

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
# Four full blocks and one more token, so all four blocks can be hits.
PROMPT = list(range(4 * BLOCK_SIZE + 1))


class Cache:
    """A `KVCacheManager` whose events go to a reference consumer."""

    def __init__(self, num_blocks: int) -> None:
        self.manager = make_kv_cache_manager(
            make_kv_cache_config(BLOCK_SIZE, num_blocks),
            max_model_len=8192,
            enable_caching=True,
            enable_kv_cache_events=True,
            hash_block_size=BLOCK_SIZE,
        )
        self.index = ReferenceIndex()
        self.num_requests = 0

    def run(self, token_ids: list[int]) -> int:
        """Prefill a request and finish it in one scheduler step.

        Returns:
            The number of tokens the manager found cached, after checking
            that the consumer predicted it from the events so far.

        """
        self.manager.new_step_starts()
        request = make_request(str(self.num_requests), token_ids, BLOCK_SIZE, sha256)
        self.num_requests += 1
        computed, num_computed, _ = self.manager.get_computed_blocks(request)
        assert expected_cached_tokens(self.index, token_ids) == num_computed
        assert (
            self.manager.allocate_slots(
                request, len(token_ids) - num_computed, num_computed, computed
            )
            is not None
        )
        self.manager.free(request)
        self.index.apply_batch(self.manager.take_events())
        return num_computed

    def reset(self) -> None:
        assert self.manager.reset_prefix_cache()
        self.index.apply_batch(self.manager.take_events())


@pytest.fixture(autouse=True)
def _init_hash_fn() -> None:
    init_none_hash(sha256)


@pytest.fixture
def cache() -> Iterator[Cache]:
    cache = Cache(num_blocks=64)
    yield cache
    assert not cache.index.anomalies, cache.index.anomalies


def test_store_then_hit(cache: Cache) -> None:
    assert cache.run(PROMPT) == 0
    assert cache.run(PROMPT) == 4 * BLOCK_SIZE


def test_block_aligned_prompt_recomputes_last_block(cache: Cache) -> None:
    prompt = PROMPT[: 4 * BLOCK_SIZE]
    cache.run(prompt)
    assert cache.run(prompt) == 3 * BLOCK_SIZE


def test_partial_prefix_hit_then_blocks_under_a_cached_parent(cache: Cache) -> None:
    """A prompt that shares two blocks with a cached one hits those two; its
    own blocks are then stored under the second shared block, and the
    consumer has to chain them to it to predict the full hit."""
    cache.run(PROMPT)
    other = PROMPT[: 2 * BLOCK_SIZE] + [10_000 + t for t in PROMPT[2 * BLOCK_SIZE :]]
    assert cache.run(other) == 2 * BLOCK_SIZE
    assert cache.run(other) == 4 * BLOCK_SIZE


def test_eviction() -> None:
    """With room for 8 blocks, a second 5-block prompt evicts some of the
    first prompt's cached blocks."""
    cache = Cache(num_blocks=1 + 8)  # block 0 is the null block
    cache.run(PROMPT)
    cache.run([10_000 + t for t in PROMPT])
    assert cache.index.stats["removed"] > 0
    assert cache.run(PROMPT) < 4 * BLOCK_SIZE
    assert not cache.index.anomalies, cache.index.anomalies


def test_reset(cache: Cache) -> None:
    cache.run(PROMPT)
    cache.reset()
    assert cache.index.stats["cleared"] == 1
    assert cache.run(PROMPT) == 0
