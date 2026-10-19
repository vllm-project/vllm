# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logical membership must survive physical reuse and independent task lifetimes."""

import random

import pytest

from vllm.v1.kv_hints.control import ControlError
from vllm.v1.kv_hints.native import (
    ContentSpan,
    HashTarget,
    ParentPrefix,
    RetainPayload,
    SessionTarget,
)
from vllm.v1.kv_hints.retention import RetentionDeclarations
from vllm.v1.kv_hints.state import SessionPrefixIndex


def span(number, start=0, end=16, group=0):
    return ContentSpan(f"{number:04x}", group, start, end)


@pytest.fixture
def index():
    return SessionPrefixIndex("namespace", "epoch")


def associate(index, session, blocks=(), continuation="main", **kwargs):
    return index.associate(session, continuation, blocks, expected_revision=0, **kwargs)


def test_shared_cache_hit_adds_membership_without_new_physical_copy(index):
    block = span(1)
    a = associate(index, "a", (block,))
    b = associate(index, "b", (block,))
    assert index.members(0, block.block_hash) == (("a", "main"), ("b", "main"))
    events = index.events(0, epoch="epoch")
    assert [(e.session_id, e.content) for e in events if e.kind == "associated"] == [
        ("a", block),
        ("b", block),
    ]
    assert index.resolve(SessionTarget("a")).prefixes[0].blocks == a.blocks
    assert index.resolve(SessionTarget("b")).prefixes[0].blocks == b.blocks


def test_explicit_branch_keeps_its_captured_parent_when_parent_advances(index):
    shared, left, right = span(1), span(2, 16, 32), span(3, 16, 32)
    parent = associate(index, "root", (shared,))
    boundary = ParentPrefix("root", "main", 16)
    branch = associate(
        index, "b", (left,), parent_prefix=boundary, task_parent_id="root"
    )
    index.associate("root", "main", (shared, right), expected_revision=parent.revision)
    selected = index.resolve(SessionTarget("b", "main"))
    assert selected.prefixes[0].blocks == (shared, left)
    grown = index.associate(
        "b",
        "main",
        (left, span(4, 32, 48)),
        expected_revision=branch.revision,
        parent_prefix=boundary,
    )
    assert grown.blocks == (shared, left, span(4, 32, 48))
    assert index.members(0, right.block_hash) == (("root", "main"),)


def test_session_selector_preserves_two_conflicting_branch_prefixes(index):
    a = associate(index, "s", (span(1),), continuation="left")
    b = associate(index, "s", (span(2),), continuation="right")
    all_branches = index.resolve(SessionTarget("s"))
    assert [p.continuation for p in all_branches.prefixes] == [
        ("s", "left"),
        ("s", "right"),
    ]
    assert [p.blocks for p in all_branches.prefixes] == [a.blocks, b.blocks]
    assert index.resolve(SessionTarget("s", "left")).prefixes[0].blocks == a.blocks


def test_group_specific_ranges_report_partial_blocks_without_rounding(index):
    blocks = (span(1), span(2, 16, 32), span(3, 0, 32, group=1))
    associate(index, "s", blocks)
    result = index.resolve(SessionTarget("s", "main", start_token=16, end_token=32))
    assert result.prefixes[0].blocks == (blocks[1],)
    assert result.prefixes[0].partial_blocks == (blocks[2],)
    with pytest.raises(ControlError, match="namespace"):
        index.resolve(HashTarget("old-weights", blocks))


def test_range_past_known_content_reports_the_untracked_tail(index):
    associate(index, "s", (span(1),))
    resolved = index.resolve(SessionTarget("s", "main", end_token=32)).prefixes[0]
    assert [
        (g.group_id, g.start_token, g.end_token) for g in resolved.missing_ranges
    ] == [(0, 16, 32)]
    assert resolved.blocks == (span(1),)


def test_known_partial_alias_fully_covers_a_shorter_range(index):
    alias, full = span(1, 0, 8), span(2, 0, 16)
    associate(index, "s", (alias, full))
    resolved = index.resolve(SessionTarget("s", "main", end_token=8)).prefixes[0]
    assert resolved.blocks == (alias,)
    assert resolved.partial_blocks == ()
    assert resolved.missing_ranges == ()


def test_branch_can_capture_a_short_alias_without_inheriting_the_longer_prefix(index):
    alias, full = span(1, 0, 8), span(2)
    associate(index, "parent", (alias, full))
    branch = associate(
        index,
        "child",
        (span(3, 8, 16),),
        parent_prefix=ParentPrefix("parent", "main", 8),
    )
    assert branch.blocks == (alias, span(3, 8, 16))
    assert index.members(0, full.block_hash) == (("parent", "main"),)


def test_known_empty_is_distinct_from_unknown_and_pruned(index):
    empty = associate(index, "empty")
    assert index.resolve(SessionTarget("empty")).prefixes[0].blocks == ()
    with pytest.raises(ControlError, match="unknown"):
        index.resolve(SessionTarget("absent", "main"))
    index.prune("empty", "main", expected_revision=empty.revision)
    with pytest.raises(ControlError, match="pruned"):
        index.resolve(SessionTarget("empty", "main"))


def test_stale_update_cannot_replace_newer_or_recreated_content(index):
    first = associate(index, "s", (span(1),))
    current = index.associate(
        "s", "main", (span(1), span(2, 16, 32)), expected_revision=first.revision
    )
    with pytest.raises(ControlError, match="revision"):
        index.associate("s", "main", first.blocks, expected_revision=first.revision)
    index.prune("s", "main", expected_revision=current.revision)
    recreated = associate(index, "s", (span(3),))
    assert recreated.revision > current.revision
    with pytest.raises(ControlError, match="revision"):
        index.associate("s", "main", current.blocks, expected_revision=current.revision)
    assert index.get("s", "main") == recreated


def test_task_cleanup_preserves_unrelated_shared_associations(index):
    shared = span(1)
    associate(index, "root", (shared,))
    associate(index, "child", (shared,), task_parent_id="root")
    associate(index, "grandchild", (shared,), task_parent_id="child")
    outsider = associate(index, "outside", (shared,))
    removed = index.finish_task(
        "root", expected_revision=index.session_revision("root")
    )
    assert set(removed) == {"root", "child", "grandchild"}
    assert index.members(0, shared.block_hash) == (("outside", "main"),)
    assert index.get("outside", "main") == outsider
    with pytest.raises(ControlError, match="finished"):
        index.get("grandchild", "main")


def test_child_growth_invalidates_old_task_tree_cleanup_revision(index):
    parent = associate(index, "parent")
    child = associate(index, "child", task_parent_id="parent")
    with pytest.raises(ControlError, match="revision"):
        index.finish_task("parent", expected_revision=parent.revision)
    before = index.session_revision("parent")
    index.associate("child", "main", (span(1),), expected_revision=child.revision)
    with pytest.raises(ControlError, match="revision"):
        index.finish_task("parent", expected_revision=before)
    assert set(index.task_tree("parent")) == {"parent", "child"}


@pytest.mark.parametrize(
    "bound", ["max_sessions", "max_continuations", "max_associations"]
)
def test_capacity_failure_preserves_queries_membership_and_event_position(bound):
    index = SessionPrefixIndex("namespace", "epoch", **{bound: 1})
    first = associate(index, "a", (span(1),))
    events = index.events(0, epoch="epoch")
    with pytest.raises(ControlError, match="capacity"):
        associate(index, "b", (span(2),))
    assert index.query().items == (first,)
    assert index.events(0, epoch="epoch") == events
    assert index.members(0, span(2).block_hash) == ()


def test_pruning_old_branch_reclaims_capacity_for_a_new_continuation():
    index = SessionPrefixIndex("namespace", "epoch", max_associations=2)
    old = associate(index, "a", (span(1), span(2, 16, 32)))
    index.prune("a", "main", expected_revision=old.revision)
    associate(index, "a", (span(3),), continuation="rewritten")
    associate(index, "b", (span(4),))
    assert index.members(0, span(1).block_hash) == ()
    assert index.members(0, span(2).block_hash) == ()
    assert index.members(0, span(3).block_hash) == (("a", "rewritten"),)


def test_duplicate_content_at_distinct_positions_does_not_break_membership_cleanup(
    index,
):
    blocks = (span(1), span(1, 16, 32))
    first = associate(index, "a", blocks)
    index.prune("a", "main", expected_revision=first.revision)
    assert index.members(0, blocks[0].block_hash) == ()
    other = associate(index, "b", blocks)
    index.finish_task("b", expected_revision=other.revision)
    assert index.members(0, blocks[0].block_hash) == ()


@pytest.mark.parametrize(
    "updated",
    [
        (),
        (span(1),),
        (span(3), span(2, 16, 32)),
        (span(1), span(2, 16, 32, group=1)),
        (span(1), span(3, 16, 48)),
    ],
    ids=["clear", "truncate", "rewrite", "change-group", "replace-tail"],
)
def test_rewriting_continuation_is_rejected_without_changing_state(index, updated):
    old = associate(index, "s", (span(1), span(2, 16, 32)))
    baseline = index.query()
    events = index.events(0, epoch="epoch")
    with pytest.raises(ControlError, match="new continuation ID"):
        index.associate("s", "main", updated, expected_revision=old.revision)
    assert index.query() == baseline
    assert index.events(0, epoch="epoch") == events
    assert index.members(0, span(1).block_hash) == (("s", "main"),)
    assert index.members(0, span(2).block_hash) == (("s", "main"),)
    assert index.members(0, span(3).block_hash) == ()


def test_growth_preserves_short_alias_and_accepts_new_group_observations(index):
    alias = span(1, 0, 8)
    old = associate(index, "s", (alias,))
    grown = index.associate(
        "s",
        "main",
        (alias, span(2), span(3, 16, 32), span(4, 0, 32, group=1)),
        expected_revision=old.revision,
    )
    assert grown.incarnation == old.incarnation
    assert index.resolve(SessionTarget("s", "main", end_token=8)).prefixes[
        0
    ].blocks == (alias,)
    assert (
        index.associate("s", "main", grown.blocks, expected_revision=grown.revision)
        == grown
    )


def test_paginated_queries_reject_a_mixed_revision_snapshot(index):
    a = associate(index, "a", (span(1),))
    b = associate(index, "b", (span(2),))
    first = index.query(limit=1)
    assert first.items == (a,)
    assert index.query(after=first.next_after, revision=first.revision).items == (b,)
    associate(index, "c", (span(3),))
    with pytest.raises(ControlError, match="restart pagination"):
        index.query(after=first.next_after, revision=first.revision)
    with pytest.raises(ControlError, match="requires an index revision"):
        index.query(after=first.next_after)


def test_event_gap_recovers_through_state_query_with_bounded_tombstones():
    index = SessionPrefixIndex("namespace", "epoch", max_events=2, max_tombstones=1)
    a = associate(index, "a", (span(1),))
    b = associate(index, "b", (span(1),))
    with pytest.raises(ControlError, match="history was pruned"):
        index.events(0, epoch="epoch")
    assert index.query().items == (a, b)
    index.prune("a", "main", expected_revision=a.revision)
    index.prune("b", "main", expected_revision=b.revision)
    with pytest.raises(ControlError, match="unknown"):
        index.get("a", "main")
    with pytest.raises(ControlError, match="pruned"):
        index.get("b", "main")
    with pytest.raises(ControlError, match="another epoch"):
        index.events(0, epoch="old")


def test_invalid_parent_mutations_are_atomic(index):
    root = associate(index, "root", (span(1),))
    other = associate(index, "other", (span(2),))
    baseline = index.query()
    with pytest.raises(ControlError, match="parent itself"):
        associate(index, "self", task_parent_id="self")
    with pytest.raises(ControlError, match="splits tracked content"):
        associate(index, "child", parent_prefix=ParentPrefix("root", "main", 8))
    with pytest.raises(ControlError, match="reparenting"):
        index.associate(
            "root",
            "main",
            root.blocks,
            expected_revision=root.revision,
            task_parent_id="other",
        )
    assert index.query() == baseline
    assert index.get("other", "main") == other


@pytest.fixture
def now():
    return [100.0]


@pytest.fixture
def declarations(index, now):
    return RetentionDeclarations(index, clock=lambda: now[0])


def retain(
    declarations,
    owner,
    priority,
    ttl=None,
    *,
    declaration_id="keep",
    target=None,
    follow=False,
):
    return declarations.retain(
        RetainPayload(
            target if target is not None else SessionTarget(owner, "main"),
            owner,
            declaration_id,
            priority,
            ttl,
            scope="follow_growth" if follow else "snapshot",
        )
    )


def test_high_priority_expiry_does_not_extend_or_delete_another_owner(
    index, declarations, now
):
    shared = span(1)
    associate(index, "a", (shared,))
    associate(index, "b", (shared,))
    retain(declarations, "a", 90, 5)
    low = retain(declarations, "b", 20, 20)
    assert declarations.priority(0, shared.block_hash) == 90
    now[0] += 5
    assert declarations.priority(0, shared.block_hash) == 20
    assert declarations.get("a", "keep") is None
    assert declarations.get("b", "keep") == low
    now[0] += 15
    assert declarations.priority(0, shared.block_hash) == 0


def test_cache_hits_and_following_growth_do_not_renew_ttl(index, declarations, now):
    initial = associate(index, "a", (span(1),))
    retained = retain(declarations, "a", 70, 5, follow=True)
    now[0] += 4
    associate(index, "hit", (span(1),))
    index.associate(
        "a", "main", (span(1), span(2, 16, 32)), expected_revision=initial.revision
    )
    assert declarations.get("a", "keep") == retained
    assert declarations.priority(0, span(2).block_hash) == 70
    now[0] += 1
    assert declarations.priority(0, span(1).block_hash) == 0
    assert declarations.priority(0, span(2).block_hash) == 0


def test_snapshot_and_follow_growth_have_separate_scope_and_release(
    index, declarations, now
):
    initial = associate(index, "s", (span(1),))
    snapshot = retain(declarations, "s", 80, declaration_id="snapshot")
    following = retain(
        declarations, "s", 30, 10, declaration_id="following", follow=True
    )
    index.associate(
        "s", "main", (span(1), span(2, 16, 32)), expected_revision=initial.revision
    )
    associate(index, "s", (span(3),), continuation="another-branch")
    assert declarations.coverage("s", "snapshot") == snapshot.snapshot
    assert len(declarations.coverage("s", "following").prefixes[0].blocks) == 2
    assert declarations.priority(0, span(2).block_hash) == 30
    assert declarations.priority(0, span(3).block_hash) == 0
    current = index.get("s", "main")
    with pytest.raises(ControlError, match="append-only"):
        index.associate("s", "main", (span(3),), expected_revision=current.revision)
    assert declarations.get("s", "following") == following
    assert declarations.coverage("s", "following").prefixes[0].blocks == current.blocks
    assert declarations.release("s", "snapshot")
    assert not declarations.release("s", "snapshot")
    assert declarations.priority(0, span(1).block_hash) == 30
    now[0] += 10
    assert declarations.priority(0, span(1).block_hash) == 0


def test_redeclare_keeps_new_lifetime_when_old_expiry_fires(index, declarations, now):
    associate(index, "s", (span(1),))
    retain(declarations, "s", 90, 2)
    now[0] += 1
    latest = retain(declarations, "s", 40, 10)
    now[0] += 1
    assert declarations.get("s", "keep") == latest
    assert declarations.priority(0, span(1).block_hash) == 40


def test_following_range_respects_group_boundaries(index, declarations):
    associate(index, "s", (span(1), span(2, 16, 32), span(2, 0, 32, group=1)))
    retain(
        declarations, "s", 60, target=SessionTarget("s", "main", 16, 32), follow=True
    )
    assert declarations.priority(0, span(1).block_hash) == 0
    assert declarations.priority(0, span(2).block_hash) == 60
    assert declarations.priority(1, span(2).block_hash) == 0


def test_declaration_capacity_failure_does_not_destroy_previous_bias(index, now):
    initial = associate(index, "s", (span(1),))
    declarations = RetentionDeclarations(
        index, max_declarations=1, max_snapshot_entries=1, clock=lambda: now[0]
    )
    previous = retain(declarations, "s", 50)
    index.associate(
        "s", "main", (span(1), span(2, 16, 32)), expected_revision=initial.revision
    )
    with pytest.raises(ControlError, match="capacity"):
        retain(declarations, "s", 90)
    with pytest.raises(ControlError, match="capacity"):
        retain(declarations, "s", 90, declaration_id="extra")
    assert declarations.get("s", "keep") == previous
    assert declarations.priority(0, span(1).block_hash) == 50
    assert declarations.priority(0, span(2).block_hash) == 0


def test_following_interest_prevents_pruning_until_release(index, declarations):
    initial = associate(index, "s", (span(1),))
    retain(declarations, "s", 30, follow=True)
    snapshot = retain(declarations, "s", 20, declaration_id="snapshot")
    with pytest.raises(ControlError, match="still uses"):
        declarations.prune("s", "main", expected_revision=initial.revision)
    declarations.release("s", "keep")
    declarations.prune("s", "main", expected_revision=initial.revision)
    assert declarations.coverage("s", "snapshot") == snapshot.snapshot
    assert declarations.priority(0, span(1).block_hash) == 20


def test_following_declaration_cannot_bind_to_a_recreated_continuation(
    index, declarations
):
    initial = associate(index, "s", (span(1),))
    retain(declarations, "outside", 90, target=SessionTarget("s", "main"), follow=True)
    # Simulate a lifecycle change outside the owning declaration coordinator.
    index.prune("s", "main", expected_revision=initial.revision)
    recreated = associate(index, "s", (span(2),))
    assert recreated.incarnation != initial.incarnation
    assert declarations.priority(0, span(2).block_hash) == 0
    with pytest.raises(ControlError, match="replaced"):
        declarations.coverage("outside", "keep")


def test_task_cleanup_preserves_foreign_declarations_and_freezes_foreign_growth(
    index, declarations, now
):
    shared = span(1)
    root = associate(index, "root", (shared,))
    associate(index, "child", (shared,), task_parent_id="root")
    associate(index, "outside", (shared,))
    retain(declarations, "root", 90)
    retain(declarations, "child", 80)
    retain(declarations, "outside", 20)
    foreign = retain(
        declarations,
        "outside",
        60,
        10,
        declaration_id="foreign",
        target=SessionTarget("root", "main"),
        follow=True,
    )
    now[0] += 5
    with pytest.raises(ControlError, match="changed"):
        declarations.finish_task("root", expected_revision=root.revision)
    assert declarations.get("root", "keep") is not None
    declarations.finish_task("root", expected_revision=index.session_revision("root"))
    assert declarations.get("root", "keep") is None
    assert declarations.get("child", "keep") is None
    assert declarations.priority(0, shared.block_hash) == 60
    closed = declarations.get("outside", "foreign")
    assert not closed.follows_growth
    assert closed.expires_at == foreign.expires_at
    associate(index, "root", (span(2),))
    assert declarations.priority(0, span(2).block_hash) == 0
    now[0] += 5
    assert declarations.priority(0, shared.block_hash) == 20


def test_cleanup_refuses_capacity_loss_for_another_owners_retention(index, now):
    associate(index, "root", (span(1), span(2, 16, 32)))
    declarations = RetentionDeclarations(
        index, max_snapshot_entries=1, clock=lambda: now[0]
    )
    retained = retain(
        declarations, "outside", 60, target=SessionTarget("root", "main"), follow=True
    )
    baseline = index.query()
    with pytest.raises(ControlError, match="another owner's"):
        declarations.finish_task(
            "root", expected_revision=index.session_revision("root")
        )
    assert index.query() == baseline
    assert declarations.get("outside", "keep") == retained
    assert declarations.priority(0, span(2).block_hash) == 60


@pytest.mark.parametrize("seed", [0, 41, 119])
def test_retention_interleavings_match_independent_lifetimes(
    index, declarations, now, seed
):
    rng = random.Random(seed)
    shared = span(1)
    for owner in ("a", "b", "c"):
        associate(index, owner, (shared,))
    expected = {}
    for _ in range(300):
        owner = rng.choice(("a", "b", "c"))
        action = rng.randrange(3)
        if action == 0:
            priority, ttl = rng.randrange(101), rng.randrange(1, 20)
            retain(declarations, owner, priority, ttl)
            expected[owner] = (priority, now[0] + ttl)
        elif action == 1:
            declarations.release(owner, "keep")
            expected.pop(owner, None)
        else:
            now[0] += rng.randrange(1, 4)
        want = max(
            (p for p, deadline in expected.values() if deadline > now[0]), default=0
        )
        assert declarations.priority(0, shared.block_hash) == want
