# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" check-points at the position an EAGLE sibling resumes from.

Full attention hits where it holds a key; EAGLE prunes one hash unit off that
candidate and drops it. Mamba materializes state only on its own block grid, so
the pruned position holds nothing and the hit floors back a whole Mamba block.

Gated by ``CacheConfig.enable_mamba_fine_grained_prefix_cache``. Every test here
drives the real ``Scheduler._mamba_block_aligned_split`` and the real
``KVCacheManager``; no chunk boundary is hard-coded.
"""

from dataclasses import replace
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
    fine_grained=True,
    num_blocks=8192,
    eagle_group=None,
    num_prefill_lookahead=0,
):
    init_none_hash(sha256)
    config = _make_hybrid_kv_cache_config(
        block_size, num_blocks, ["full", "mamba_align"]
    )
    if eagle_group is not None:
        groups = list(config.kv_cache_groups)
        groups[eagle_group] = replace(groups[eagle_group], is_eagle_group=True)
        config = replace(config, kv_cache_groups=groups)
    return make_kv_cache_manager(
        kv_cache_config=config,
        max_model_len=1 << 20,
        enable_caching=True,
        hash_block_size=hash_block_size,
        use_eagle=True,
        num_prefill_lookahead=num_prefill_lookahead,
        enable_mamba_fine_grained_prefix_cache=fine_grained,
    )


def _stub(manager, block_size, hash_block_size, *, block_drop=True):
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
        use_eagle=True,
        # The EAGLE adjustments key on the block-drop bit, not plain use_eagle:
        # they exist only to compensate for the drop.
        use_eagle_block_drop=block_drop,
        hash_block_size=hash_block_size,
        mamba_has_prefill_checkpoint_blocks=False,  # forced False under eagle
        mamba_partial_cache_hit=partial_hit,
        mamba_fine_grained_prefix_cache=(
            partial_hit and manager.mamba_fine_grained_prefix_cache
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
        "producer", PREFIX[:prompt_len], stub.hash_block_size, sha256
    )
    _prefill(manager, stub, producer, external=prompt_len - 1)


def _sibling_hit(manager, shared, suffix, hash_block_size):
    request = make_request("sibling", PREFIX[:shared] + suffix, hash_block_size, sha256)
    return manager.get_computed_blocks(request)[1]


# --------------------------------------------------------------------------
# The two positions a sibling resumes at


def test_sibling_resumes_from_the_observed_junction():
    """The scheduler splits at the junction; the manager must cache there."""
    block_size, hash_block_size = 512, 32
    manager = _manager(block_size, hash_block_size)
    stub = _stub(manager, block_size, hash_block_size)
    assert stub.mamba_fine_grained_prefix_cache, "the feature must be armed"
    _orphaned_full_attention_tail(manager, stub, 2020)

    consumer = make_request(
        "consumer", PREFIX[:2016] + [-1] * 584, hash_block_size, sha256
    )
    junction = manager.get_computed_blocks(consumer)[2]
    assert junction, "expected a junction against the orphaned full-attention tail"
    _prefill(manager, stub, consumer)

    hit = _sibling_hit(manager, 2016, [-2] * 584, hash_block_size)
    assert hit == junction, (
        f"the chunk stopped at {junction} but nothing was cached there: "
        f"sibling resumes at {hit}"
    )


def test_sibling_resumes_below_the_block_grid_when_the_prefix_ends_early():
    """A system prompt followed by a per-request suffix -- the deployed shape.

    The owner's prompt tail sits over its own suffix, which no sibling shares, so
    a check-point there is unreachable. The next request's full-attention match
    stops at the last shared block boundary and EAGLE drops one hash unit below
    it, so state has to exist just under the block grid too.
    """
    block_size, hash_block_size = 16, 4
    manager = _manager(block_size, hash_block_size)
    stub = _stub(manager, block_size, hash_block_size)

    shared = 24
    owner = make_request("owner", PREFIX[:shared] + [-1] * 16, hash_block_size, sha256)
    _prefill(manager, stub, owner)

    # The first follower observes the junction (16 shared, dropped to 12) and
    # registers state there; the second one gets to resume from it.
    follower = make_request(
        "follower", PREFIX[:shared] + [-2] * 16, hash_block_size, sha256
    )
    _prefill(manager, stub, follower)

    resume = shared // block_size * block_size - hash_block_size
    hit = _sibling_hit(manager, shared, [-3] * 16, hash_block_size)
    assert hit == resume, f"expected the resume point at {resume}, got {hit}"


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
                "consumer", PREFIX[:shared] + suffix, hash_block_size, sha256
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

    request = make_request("r", PREFIX[:2000], hash_block_size, sha256)
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
    assert mamba._cache_partial_tail_block(request, running) is None


@pytest.mark.parametrize("eagle_group", [None, 0])
@pytest.mark.parametrize("num_prefill_lookahead", [0, 2, 33])
def test_enabling_never_reduces_reuse(eagle_group, num_prefill_lookahead):
    """Turning the flag on may add check-points; it must never move one away.

    Two axes decide whether the manager can honour the junction stop the
    scheduler makes for it, and both fail silently -- as lost reuse, not as an
    error. ``is_eagle_group`` annotation: the manager takes the EAGLE drop per
    group while the scheduler flag is model-wide, so on an annotated model the
    Mamba group can be outside ``eagle_group_ids``. Multi-module MTP: the
    coordinator hands the manager ``num_computed - num_reprefillable``, not the
    chunk end. In either case the junction stop displaces the block-boundary
    stop and the check-point it replaced is never written.
    """
    block_size, hash_block_size = 512, 32

    def sibling_hit(fine_grained):
        manager = _manager(
            block_size,
            hash_block_size,
            fine_grained=fine_grained,
            eagle_group=eagle_group,
            num_prefill_lookahead=num_prefill_lookahead,
        )
        stub = _stub(manager, block_size, hash_block_size)
        _orphaned_full_attention_tail(manager, stub, 2020)
        consumer = make_request(
            "consumer", PREFIX[:2016] + [-1] * 584, hash_block_size, sha256
        )
        _prefill(manager, stub, consumer)
        return _sibling_hit(manager, 2016, [-2] * 584, hash_block_size)

    on, off = sibling_hit(True), sibling_hit(False)
    assert on >= off, f"enabling the flag cut the sibling's hit from {off} to {on}"


@pytest.mark.parametrize("prompt_len", [2048, 4096])
def test_a_block_aligned_prompt_is_still_reusable(prompt_len):
    """A prompt ending exactly on a block boundary must still be reusable.

    A replay is capped at ``num_tokens - 1``, so it can never reach a
    check-point at the prompt's own end. Flooring the Mamba grid stop from
    ``num_tokens`` put the only reachable state there, and an identical replay
    got nothing at all -- not a shorter hit, zero.
    """
    block_size, hash_block_size = 512, 32
    manager = _manager(block_size, hash_block_size, fine_grained=False)
    stub = _stub(manager, block_size, hash_block_size)
    assert prompt_len % block_size == 0, "the regression needs a block-aligned prompt"

    producer = make_request("p", PREFIX[:prompt_len], hash_block_size, sha256)
    _prefill(manager, stub, producer)
    replay = make_request("r", PREFIX[:prompt_len], hash_block_size, sha256)
    assert manager.get_computed_blocks(replay)[1] >= prompt_len - 2 * block_size


def test_disabled_by_default():
    """Off, the junction is block-floored and no resume point is registered."""
    block_size, hash_block_size = 512, 32
    manager = _manager(block_size, hash_block_size, fine_grained=False)
    stub = _stub(manager, block_size, hash_block_size)
    _orphaned_full_attention_tail(manager, stub, 2020)

    consumer = make_request(
        "consumer", PREFIX[:2016] + [-1] * 584, hash_block_size, sha256
    )
    junction = manager.get_computed_blocks(consumer)[2]
    assert junction, "expected a junction"
    _prefill(manager, stub, consumer)

    assert _sibling_hit(manager, 2016, [-2] * 584, hash_block_size) < junction

    request = make_request("r", PREFIX[:10000], hash_block_size, sha256)
    request.shared_prefix_boundary = 2000
    assert Scheduler._mamba_block_aligned_split(stub, request, 8192) == 1536
