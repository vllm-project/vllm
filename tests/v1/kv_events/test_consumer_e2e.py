# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end checks of the KV event stream against the engine's real cache.

Every test runs a real `vllm serve` with KV events enabled, feeds the
published events to the reference consumer in `reference_consumer.py` and
compares the consumer's prediction with the `cached_tokens` the engine reports
for the actual request. That agreement is the contract external routers and
indexers (llm-d, Dynamo, Mooncake Conductor, vLLM Router) depend on.
"""

import threading
from collections.abc import Iterator

import pytest

from vllm.utils.network_utils import get_open_ports_list

from .engine_harness import (
    REQUEST_TIMEOUT_S,
    ServedEngine,
    ServerConfig,
    cached_tokens,
    serve,
    subscribe,
)
from .reference_consumer import EventStream, expected_cached_tokens

DENSE = ServerConfig(
    name="dense",
    model="Qwen/Qwen3-0.6B",
    description="dense GQA model, no speculative decoding",
)
BLOCK_SIZE = DENSE.block_size
PROMPT_BLOCKS = 10


@pytest.fixture(scope="module")
def kv_ports() -> tuple[int, int]:
    pub_port, replay_port = get_open_ports_list(2)
    return pub_port, replay_port


@pytest.fixture(scope="module")
def engine(kv_ports: tuple[int, int]) -> Iterator[ServedEngine]:
    with serve(DENSE, *kv_ports) as engine:
        assert engine.block_size == BLOCK_SIZE, engine.groups
        yield engine


@pytest.fixture
def stream(engine: ServedEngine) -> Iterator[EventStream]:
    with engine.subscribe() as stream:
        engine.fresh_state(stream)
        yield stream
        assert not stream.index.anomalies, stream.index.anomalies


def test_store_then_hit(engine: ServedEngine, stream: EventStream) -> None:
    prompt = engine.prompt(1, PROMPT_BLOCKS)
    assert engine.check(stream, prompt, "cold prompt") == 0
    assert stream.index.cached_prefix_blocks(prompt, BLOCK_SIZE) == PROMPT_BLOCKS
    warm = engine.check(stream, prompt, "same prompt again")
    assert warm == PROMPT_BLOCKS * BLOCK_SIZE
    shared = 5
    divergent = prompt[: shared * BLOCK_SIZE] + engine.prompt(2, PROMPT_BLOCKS - shared)
    assert engine.check(stream, divergent, "shared prefix") == shared * BLOCK_SIZE


def test_block_aligned_prompt_recomputes_last_block(
    engine: ServedEngine, stream: EventStream
) -> None:
    prompt = engine.prompt(3, PROMPT_BLOCKS, extra_tokens=0)
    assert engine.check(stream, prompt, "cold aligned prompt") == 0
    # All ten blocks are cached, but the engine must recompute the last token
    # to produce logits, which costs the whole last block.
    assert (
        engine.check(stream, prompt, "aligned prompt again")
        == (PROMPT_BLOCKS - 1) * BLOCK_SIZE
    )


def test_full_report_mode_republishes_the_hit(
    engine: ServedEngine, stream: EventStream
) -> None:
    prompt = engine.prompt(4, PROMPT_BLOCKS)
    engine.check(stream, prompt, "cold prompt")
    resident_before = set(stream.index.resident)
    duplicates_before = stream.index.stats["stored_duplicate"]
    completion = engine.complete(prompt, vllm_xargs={"kv_cache_report_mode": "full"})
    assert cached_tokens(completion) == PROMPT_BLOCKS * BLOCK_SIZE
    stream.drain()
    # The reused prefix is reported again as one BlockStored rooted at None.
    assert stream.index.stats["stored_duplicate"] - duplicates_before == PROMPT_BLOCKS
    assert stream.index.resident == resident_before


def test_eviction_matches_engine(engine: ServedEngine, stream: EventStream) -> None:
    prompts = [engine.prompt(100 + i, PROMPT_BLOCKS) for i in range(12)]
    for i, prompt in enumerate(prompts):
        assert engine.check(stream, prompt, f"cold prompt {i}") == 0
    assert stream.index.stats["removed"] > 0, "expected evictions"
    # Most recent first, so the probes themselves do not evict what the next
    # probe expects to find: the newest prompts are still resident, the oldest
    # ones were evicted to make room for them.
    hits = [
        engine.check(stream, prompt, f"prompt {i} after eviction")
        for i, prompt in reversed(list(enumerate(prompts)))
    ]
    assert hits[0] == PROMPT_BLOCKS * BLOCK_SIZE, hits
    assert hits[-1] == 0, hits


@pytest.mark.xfail(
    strict=True,
    reason="vllm-project/vllm#63149 (publish idle resets immediately): an idle "
    "engine publishes AllBlocksCleared only with its next step, so a consumer "
    "keeps serving the cleared blocks",
)
def test_idle_reset_is_published(engine: ServedEngine, stream: EventStream) -> None:
    prompt = engine.prompt(5, PROMPT_BLOCKS)
    engine.check(stream, prompt, "cold prompt")
    assert stream.index.cached_prefix_blocks(prompt, BLOCK_SIZE) == PROMPT_BLOCKS
    cleared_before = stream.index.stats["cleared"]
    assert engine.reset_prefix_cache()
    stream.drain(timeout_s=5)
    assert stream.index.stats["cleared"] == cleared_before + 1, (
        "AllBlocksCleared was not published after an idle reset"
    )
    assert not stream.index.resident
    assert engine.check(stream, prompt, "prompt after reset") == 0


def test_reconnect_replays_missed_batches(
    engine: ServedEngine, stream: EventStream
) -> None:
    first = engine.prompt(6, PROMPT_BLOCKS)
    engine.check(stream, first, "cold prompt")
    stream.disconnect()
    missed = [engine.prompt(7, PROMPT_BLOCKS), engine.prompt(8, PROMPT_BLOCKS)]
    for prompt in missed:
        engine.complete(prompt)
    stream.connect()
    stream.resync()
    assert stream.replayed >= len(missed)
    assert stream.lost == []
    for i, prompt in enumerate(missed):
        assert engine.check(stream, prompt, f"replayed prompt {i}") == (
            PROMPT_BLOCKS * BLOCK_SIZE
        )


def test_replay_overlapping_live_batches_is_not_a_restart(
    engine: ServedEngine, stream: EventStream
) -> None:
    """A consumer that replays while live batches are queued receives those
    batches twice. The second copies are duplicates, not a restart."""
    prompts = [engine.prompt(12 + i, PROMPT_BLOCKS) for i in range(2)]
    published = stream.last_seq + len(prompts)
    with engine.subscribe() as observer:
        for prompt in prompts:
            engine.complete(prompt)
        while observer.last_seq < published:
            assert observer.drain(), "fewer batches published than requests sent"
    # The live copies of these batches are still queued on the socket.
    stream.resync()
    stream.drain()
    assert stream.restarts == 0
    assert stream.duplicates >= len(prompts)
    for i, prompt in enumerate(prompts):
        assert engine.check(stream, prompt, f"prompt {i}") == (
            PROMPT_BLOCKS * BLOCK_SIZE
        )


def test_concurrent_requests_and_preemption(
    engine: ServedEngine, stream: EventStream
) -> None:
    # Eight 12-block prompts decoding 128 tokens each need far more than the
    # 96 blocks the engine has, so some of them get preempted and recomputed.
    prompts = [engine.prompt(200 + i, 12) for i in range(8)]
    preemptions_before = engine.num_preemptions()
    errors: list[BaseException] = []

    def run(prompt: list[int]) -> None:
        try:
            engine.complete(prompt, max_tokens=128, ignore_eos=True)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(prompt,)) for prompt in prompts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=REQUEST_TIMEOUT_S)
    assert not errors, errors
    assert engine.num_preemptions() > preemptions_before, "no preemption happened"
    stream.drain()
    for i, prompt in enumerate(prompts):
        engine.check(stream, prompt, f"prompt {i} after concurrent run")


def test_replay_window_exhaustion_is_detected() -> None:
    """With a bounded replay buffer the publisher silently returns only the
    retained suffix; the consumer can only notice that the first replayed
    sequence is later than the one it asked for."""
    pub_port, replay_port = get_open_ports_list(2)
    buffer_steps = 4
    with (
        serve(DENSE, pub_port, replay_port, buffer_steps=buffer_steps) as engine,
        subscribe(pub_port, replay_port) as stream,
    ):
        engine.check(stream, engine.prompt(9, PROMPT_BLOCKS), "cold prompt")
        stream.disconnect()
        missed = [
            engine.prompt(300 + i, PROMPT_BLOCKS) for i in range(buffer_steps * 2)
        ]
        for prompt in missed:
            engine.complete(prompt)
        stream.connect()
        stream.resync()
        assert stream.lost, "the exhausted replay window went unnoticed"
        wanted, first = stream.lost[0]
        assert first > wanted
        # Blocks published in the lost batches are unknown to the consumer
        # although the engine still holds them: the gap is real.
        assert expected_cached_tokens(stream.index, missed[0], BLOCK_SIZE) == 0
        assert cached_tokens(engine.complete(missed[0])) == PROMPT_BLOCKS * BLOCK_SIZE


def test_publisher_restart_is_detected_when_the_counter_regresses() -> None:
    """An engine restarted on the same endpoints starts counting from zero
    again. While the new counter is still below the old one the consumer sees
    the regression, drops what it learned from the old process and replays
    whatever the new process published before (a subscriber reconnecting to
    a fresh publisher can miss its first batch)."""
    pub_port, replay_port = get_open_ports_list(2)
    with subscribe(pub_port, replay_port) as stream:
        with serve(DENSE, pub_port, replay_port) as engine:
            old_prompts = [engine.prompt(10 + i, PROMPT_BLOCKS) for i in range(3)]
            for i, prompt in enumerate(old_prompts):
                engine.check(stream, prompt, f"old engine prompt {i}")
            assert stream.last_seq >= 2
        with serve(DENSE, pub_port, replay_port) as engine:
            new_prompts = [engine.prompt(20 + i, PROMPT_BLOCKS) for i in range(2)]
            for i, prompt in enumerate(new_prompts):
                assert engine.check(stream, prompt, f"new engine prompt {i}") == 0
            assert stream.restarts == 1
            assert stream.lost == []
            for i, prompt in enumerate(old_prompts):
                assert (
                    engine.check(stream, prompt, f"old prompt {i} on new engine") == 0
                )
            for i, prompt in enumerate(new_prompts):
                assert engine.check(stream, prompt, f"new prompt {i} again") == (
                    PROMPT_BLOCKS * BLOCK_SIZE
                )


@pytest.mark.xfail(
    strict=True,
    reason="vllm-project/vllm#40363: a restarted publisher whose counter has "
    "already passed the old one is indistinguishable from a continuation; the "
    "event stream carries no publisher epoch",
)
def test_publisher_restart_without_regression_is_invisible() -> None:
    """The consumer saw the old engine's batches up to sequence N. While it
    reconnects, the new engine publishes batches 0..N, including the reset
    from its startup, so the next batch the consumer sees, N + 1, looks like
    the old process's next one. Replay cannot help: it is asked from N + 1.
    The consumer keeps the old engine's blocks as resident and mispredicts
    the hit."""
    pub_port, replay_port = get_open_ports_list(2)
    with subscribe(pub_port, replay_port) as stream:
        with serve(DENSE, pub_port, replay_port) as engine:
            old_prompt = engine.prompt(30, PROMPT_BLOCKS)
            engine.check(stream, old_prompt, "old engine prompt")
        old_last_seq = stream.last_seq
        stream.disconnect()
        with serve(DENSE, pub_port, replay_port) as engine:
            # Unseen by the consumer: the new counter catches up with the old.
            with engine.subscribe() as observer:
                seed = 31
                while observer.last_seq < old_last_seq:
                    engine.complete(engine.prompt(seed, 1))
                    observer.drain()
                    seed += 1
            stream.connect()
            stream.resync()
            engine.check(stream, engine.prompt(40, PROMPT_BLOCKS), "new engine prompt")
            assert stream.restarts == 0
            # The old prompt is gone from the new engine but still resident
            # in the consumer's index.
            engine.check(stream, old_prompt, "old prompt on new engine")
