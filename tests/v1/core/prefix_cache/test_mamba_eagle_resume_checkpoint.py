# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" check-points at the position a sibling resumes from.

Lookahead hashes identify the shared prefix. Mamba must materialize state at
that same position, including hash boundaries inside a physical Mamba block.

Gated by ``CacheConfig.enable_mamba_shared_prefix_checkpoint``. Every test here
drives the real ``Scheduler._mamba_block_aligned_split`` and the real
``KVCacheManager``; no chunk boundary is hard-coded.
"""

from types import SimpleNamespace

import pytest

from tests.v1.core.test_prefix_caching import (
    _make_hybrid_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler

PREFIX = list(range(1, 20_001))


def _manager(
    block_size,
    hash_block_size,
    *,
    shared_prefix_checkpoint=True,
    num_blocks=8192,
    num_hash_lookahead_tokens=1,
    num_prefill_lookahead=0,
):
    init_none_hash(sha256)
    config = _make_hybrid_kv_cache_config(block_size, num_blocks, ["full", "mamba"])
    return make_kv_cache_manager(
        kv_cache_config=config,
        max_model_len=1 << 20,
        enable_caching=True,
        hash_block_size=hash_block_size,
        num_prefill_lookahead=num_prefill_lookahead,
        enable_mamba_shared_prefix_checkpoint=shared_prefix_checkpoint,
        num_hash_lookahead_tokens=num_hash_lookahead_tokens,
    )


def _stub(manager, block_size, hash_block_size):
    """``self`` for the real ``Scheduler._mamba_block_aligned_split``.

    Derives both gates exactly as ``Scheduler.__init__`` does, so a manager that
    cannot honour the junction stop also stops the split from making one.
    """
    partial_hit = (
        hash_block_size < block_size and manager.coordinator.enable_partial_hash_hits
    )
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        max_num_scheduled_tokens=1 << 20,
        num_hash_lookahead_tokens=manager.block_pool.num_hash_lookahead_tokens,
        hash_block_size=hash_block_size,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_partial_cache_hit=partial_hit,
        mamba_shared_prefix_checkpoint=(
            partial_hit and manager.mamba_shared_prefix_checkpoint
        ),
    )


def _prefill(manager, stub, request, *, external=0) -> list[int]:
    """Schedule ``request`` to completion the way ``Scheduler.schedule()`` does.

    Returns the chunk ends. Every boundary comes from the real splitter, so no
    test in this file hard-codes one.
    """
    blocks, local, junction = manager.get_computed_blocks(request)
    request.shared_prefix_boundary = junction
    ends = []
    first = True
    while request.num_computed_tokens < request.num_tokens:
        new_local, ext = (local, external) if first else (0, 0)
        start = request.num_computed_tokens + new_local + ext
        if start >= request.num_tokens:
            break
        num_new = Scheduler._mamba_block_aligned_split(
            stub, request, request.num_tokens - start, new_local, ext
        )
        if num_new <= 0:
            break
        assert (
            manager.allocate_slots(
                request,
                num_new,
                num_new_computed_tokens=new_local,
                new_computed_blocks=blocks if first else None,
                num_external_computed_tokens=ext,
                has_scheduled_reqs=False,
            )
            is not None
        )
        request.num_computed_tokens = start + num_new
        ends.append(request.num_computed_tokens)
        _, retained = manager.take_kv_cache_block_copies()
        if retained:
            manager.block_pool.free_blocks(retained)
        manager.new_step_starts()
        first = False
    return ends


def _orphaned_full_attention_tail(manager, stub, prompt_len):
    """Produce the state a KV connector leaves behind, which creates a junction.

    NIXL reports ``num_prompt_tokens - 1`` external computed tokens for Mamba
    models (``nixl/base_scheduler.py::_get_remote_prefill_token_count``), so the
    producer starts past its own tail stop: full attention still registers a
    partial tail the Mamba group has no state for.
    """
    producer = make_request(
        "producer",
        PREFIX[:prompt_len],
        stub.hash_block_size,
        sha256,
        num_hash_lookahead_tokens=manager.block_pool.num_hash_lookahead_tokens,
    )
    _prefill(manager, stub, producer, external=prompt_len - 1)


def _sibling_hit(manager, shared, suffix, hash_block_size):
    request = make_request(
        "sibling",
        PREFIX[:shared] + suffix,
        hash_block_size,
        sha256,
        num_hash_lookahead_tokens=manager.block_pool.num_hash_lookahead_tokens,
    )
    return manager.get_computed_blocks(request)[1]


# --------------------------------------------------------------------------
# The two positions a sibling resumes at


def test_sibling_resumes_from_the_observed_junction():
    """The scheduler splits at the junction; the manager must cache there."""
    block_size, hash_block_size = 512, 32
    manager = _manager(block_size, hash_block_size)
    stub = _stub(manager, block_size, hash_block_size)
    assert stub.mamba_shared_prefix_checkpoint, "the feature must be armed"
    _orphaned_full_attention_tail(manager, stub, 2020)

    consumer = make_request(
        "consumer",
        PREFIX[:2017] + [-1] * 583,
        hash_block_size,
        sha256,
        num_hash_lookahead_tokens=manager.block_pool.num_hash_lookahead_tokens,
    )
    junction = manager.get_computed_blocks(consumer)[2]
    assert junction == 2016
    _prefill(manager, stub, consumer)

    hit = _sibling_hit(manager, 2017, [-2] * 583, hash_block_size)
    assert hit == junction, (
        f"the chunk stopped at {junction} but nothing was cached there: "
        f"sibling resumes at {hit}"
    )


def test_sibling_resumes_below_the_block_grid_when_the_prefix_ends_early():
    """An early prefix end reuses the latest materialized shared block."""
    block_size, hash_block_size = 16, 4
    manager = _manager(block_size, hash_block_size)
    stub = _stub(manager, block_size, hash_block_size)

    shared = 24
    owner = make_request(
        "owner",
        PREFIX[:shared] + [-1] * 16,
        hash_block_size,
        sha256,
        num_hash_lookahead_tokens=manager.block_pool.num_hash_lookahead_tokens,
    )
    _prefill(manager, stub, owner)

    follower = make_request(
        "follower",
        PREFIX[:shared] + [-2] * 16,
        hash_block_size,
        sha256,
        num_hash_lookahead_tokens=manager.block_pool.num_hash_lookahead_tokens,
    )
    _prefill(manager, stub, follower)

    resume = shared // block_size * block_size
    hit = _sibling_hit(manager, shared, [-3] * 16, hash_block_size)
    assert hit == resume, f"expected the resume point at {resume}, got {hit}"


def test_mamba_prefix_cache_respects_lookahead():
    block_size, hash_block_size = 1024, 64
    manager = _manager(block_size, hash_block_size)
    prompt_a = PREFIX[:1600]
    hash_ready_boundary = len(prompt_a) - hash_block_size

    _prefill(
        manager,
        _stub(manager, block_size, hash_block_size),
        make_request(
            "A", prompt_a, hash_block_size, sha256, num_hash_lookahead_tokens=1
        ),
    )

    tokens_after_a = [-1] * 500
    hit = _sibling_hit(manager, len(prompt_a), tokens_after_a, hash_block_size)
    assert hit == hash_ready_boundary


# --------------------------------------------------------------------------
# Invariants the two clauses have to keep


def _armed_configs():
    """``(block_size, hash_block_size)`` pairs the junction path can arm on.

    The prefix match unit must divide every group's block size and be strictly
    smaller, or ``enable_partial_hash_hits`` is False and the path is inert.
    """
    for block_size in (16, 128, 512):
        for hash_block_size in (4, 32):
            if hash_block_size < block_size and block_size % hash_block_size == 0:
                yield block_size, hash_block_size


def test_scheduler_never_stops_where_the_manager_refuses():
    """A junction stop must always be a position the manager caches.

    "The scheduler schedules a number of tokens it thinks can be cached, but the
    cache manager won't agree" is a property of the pair, not of one
    configuration, so sweep the armed parameter space.
    """
    evaluated, refused, skipped = 0, [], []
    for block_size, hash_block_size in _armed_configs():
        for prompt_len in (
            2 * block_size - hash_block_size,
            3 * block_size + hash_block_size + 1,
            4 * block_size - 1,
        ):
            manager = _manager(block_size, hash_block_size)
            stub = _stub(manager, block_size, hash_block_size)
            _orphaned_full_attention_tail(manager, stub, prompt_len)

            shared = prompt_len // hash_block_size * hash_block_size
            suffix = [-1] * (block_size + 3)
            consumer = make_request(
                "consumer",
                PREFIX[:shared] + suffix,
                hash_block_size,
                sha256,
                num_hash_lookahead_tokens=manager.block_pool.num_hash_lookahead_tokens,
            )
            junction = manager.get_computed_blocks(consumer)[2]
            if not junction:
                continue
            case = (block_size, hash_block_size, prompt_len, junction)
            # The stop and the check-point are both keyed on the junction, so it
            # has to be on the hash grid -- nothing else keeps them together.
            assert junction % hash_block_size == 0, case
            if junction not in _prefill(manager, stub, consumer):
                skipped.append(case)
                continue
            evaluated += 1
            hit = _sibling_hit(
                manager, shared, [-2] * (block_size + 3), hash_block_size
            )
            if hit < junction:
                refused.append(case + (hit,))

    assert not skipped, (
        f"the scheduler declined to stop at {len(skipped)}: {skipped[:3]}"
    )
    assert evaluated >= 10, f"sweep did not exercise the junction: {evaluated} cases"
    assert not refused, (
        f"{len(refused)} of {evaluated} configs split at a junction the manager then "
        f"refused (block, hash, prompt_len, junction, sibling_hit): {refused[:3]}"
    )


def test_a_junction_past_the_prompt_falls_back_and_registers_nothing():
    """A resumed request's junction can land in its output tokens.

    The manager writes nothing there -- during decode the target is the running
    state block, mutated in place, which equals what its key promises only after
    that step's forward. So the scheduler must not stop AT it, but it must still
    take the block-floored stop stock vLLM would have made: dropping the stop
    entirely loses a check-point a sibling could have resumed from.
    """
    block_size, hash_block_size = 512, 32
    manager = _manager(block_size, hash_block_size)
    stub = _stub(manager, block_size, hash_block_size)

    request = make_request(
        "r", PREFIX[:2000], hash_block_size, sha256, num_hash_lookahead_tokens=1
    )
    for _ in range(2000):
        request.append_output_token_ids(1)
    assert request.num_prompt_tokens == 2000 and request.num_tokens == 4000

    # 2208 is past the prompt: not used verbatim, but the block-floored stop
    # (2048) still applies -- not the last cacheable boundary (3072), which is
    # what dropping the stop would give.
    request.shared_prefix_boundary = 2208
    assert Scheduler._mamba_block_aligned_split(stub, request, 8192) == 2048
    # Inside the prompt it still stops, at the junction itself.
    request.shared_prefix_boundary = 992
    assert Scheduler._mamba_block_aligned_split(stub, request, 8192) == 992

    # And the manager refuses on its own at the position that matters: the
    # RUNNING state block mid-decode. Interior blocks are nulled in align mode,
    # so every other decode-side position is already refused for that reason --
    # here the block is live and holds exactly what a key would claim, and only
    # the prompt bound stops it being published a forward before it changes.
    _prefill(manager, stub, request)
    mamba = manager.coordinator.single_type_managers[1]
    running = request.num_computed_tokens
    assert not mamba.req_to_blocks["r"][running // block_size].is_null
    request.shared_prefix_boundary = running
    assert (
        mamba._cache_partial_tail_block(
            request,
            running,
            retention_interval=manager.coordinator.retention_interval,
        )
        is None
    )


@pytest.mark.parametrize("lookahead", [0, 1])
@pytest.mark.parametrize("num_prefill_lookahead", [0, 2])
def test_enabling_never_reduces_reuse(lookahead, num_prefill_lookahead):
    """Eligible junctions improve reuse; other configurations are unchanged."""
    block_size, hash_block_size = 512, 32

    def sibling_hit(shared_prefix_checkpoint):
        manager = _manager(
            block_size,
            hash_block_size,
            shared_prefix_checkpoint=shared_prefix_checkpoint,
            num_hash_lookahead_tokens=lookahead,
            num_prefill_lookahead=num_prefill_lookahead,
        )
        assert manager.mamba_shared_prefix_checkpoint == (
            shared_prefix_checkpoint and lookahead > 0 and num_prefill_lookahead <= 1
        )
        stub = _stub(manager, block_size, hash_block_size)
        _orphaned_full_attention_tail(manager, stub, 2020)
        consumer = make_request(
            "consumer",
            PREFIX[:2017] + [-1] * 583,
            hash_block_size,
            sha256,
            num_hash_lookahead_tokens=manager.block_pool.num_hash_lookahead_tokens,
        )
        _prefill(manager, stub, consumer)
        return _sibling_hit(manager, 2017, [-2] * 583, hash_block_size)

    on, off = sibling_hit(True), sibling_hit(False)
    if lookahead > 0 and num_prefill_lookahead <= 1:
        assert on > off
    else:
        assert on == off


def test_disabled_by_default():
    """Off, the junction is block-floored and no resume point is registered."""
    block_size, hash_block_size = 512, 32
    manager = _manager(block_size, hash_block_size, shared_prefix_checkpoint=False)
    stub = _stub(manager, block_size, hash_block_size)
    _orphaned_full_attention_tail(manager, stub, 2020)

    consumer = make_request(
        "consumer",
        PREFIX[:2016] + [-1] * 584,
        hash_block_size,
        sha256,
        num_hash_lookahead_tokens=manager.block_pool.num_hash_lookahead_tokens,
    )
    junction = manager.get_computed_blocks(consumer)[2]
    assert junction, "expected a junction"
    _prefill(manager, stub, consumer)

    assert junction == 1536
    assert _sibling_hit(manager, 2016, [-2] * 584, hash_block_size) == junction

    request = make_request(
        "r", PREFIX[:10000], hash_block_size, sha256, num_hash_lookahead_tokens=1
    )
    request.shared_prefix_boundary = 2000
    assert Scheduler._mamba_block_aligned_split(stub, request, 8192) == 1536
