# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_hints.retain import (
    MAX_ACTIONS,
    MAX_DIRECTIVES,
    MAX_DURATION_S,
    RETAIN_ACTION_TYPE,
    RETAIN_ACTION_VERSION,
    RetainDirective,
    parse_retain_actions,
)


def _action(
    payload,
    *,
    action_type=RETAIN_ACTION_TYPE,
    version=RETAIN_ACTION_VERSION,
    action_id="a1",
):
    return KvHintAction(
        action_id=action_id,
        action_type=action_type,
        action_version=version,
        payload=payload,
    )


def _envelope(*actions):
    return KvHintsEnvelope(
        protocol_version="0.1", message_id="m1", actions=list(actions)
    )


def _retain(directives, scope="alice", **kw):
    return _action({"scope": scope, "directives": directives}, **kw)


def test_none_envelope_is_none():
    assert parse_retain_actions(None, strict=True) is None


def test_other_actions_are_ignored():
    env = _envelope(_action({"keys": [1]}, action_type="kv.fetch"))
    assert parse_retain_actions(env, strict=True) is None
    assert parse_retain_actions(env, strict=False) is None


def test_single_action_parses_in_order():
    env = _envelope(
        _retain(
            [
                {"start": 0, "end": 32, "priority": 80, "duration": 60.0},
                {"start": 32, "end": None, "priority": 40},
            ]
        )
    )
    parsed = parse_retain_actions(env, strict=True)
    assert parsed is not None
    assert parsed.scope == "alice"
    assert parsed.directives == (
        RetainDirective(priority=80, start=0, end=32, duration=60.0),
        RetainDirective(priority=40, start=32, end=None, duration=None),
    )


def test_scope_omitted_is_none():
    env = _envelope(_action({"directives": [{"start": 0, "end": 16, "priority": 50}]}))
    parsed = parse_retain_actions(env, strict=True)
    assert parsed is not None and parsed.scope is None


def test_covers_output_directive():
    env = _envelope(_retain([{"covers_output": True, "priority": 70, "duration": 5.0}]))
    parsed = parse_retain_actions(env, strict=True)
    assert parsed is not None
    (d,) = parsed.directives
    assert d.covers_output and d.priority == 70 and d.duration == 5.0


def test_multiple_actions_concatenate():
    env = _envelope(
        _retain([{"start": 0, "end": 16, "priority": 90}], action_id="a1"),
        _retain([{"start": 16, "end": 32, "priority": 50}], action_id="a2"),
    )
    parsed = parse_retain_actions(env, strict=True)
    assert parsed is not None
    assert [d.start for d in parsed.directives] == [0, 16]


def test_empty_directives_is_a_no_op_claim():
    parsed = parse_retain_actions(_envelope(_retain([])), strict=True)
    assert parsed is not None
    assert parsed.scope == "alice" and parsed.directives == ()


@pytest.mark.parametrize(
    "directives",
    [
        [
            {"start": 0, "end": 100, "priority": 30},
            {"start": 100, "end": 200, "priority": 80},
        ],
        # unsorted input is sorted by start before the check
        [
            {"start": 100, "end": 200, "priority": 80},
            {"start": 0, "end": 100, "priority": 30},
        ],
        # covers_output sorts deepest: 60 after 30 is an increase
        [
            {"start": 0, "end": 100, "priority": 30},
            {"covers_output": True, "priority": 60},
        ],
    ],
)
def test_increasing_priority_rejected_when_strict(directives):
    with pytest.raises(ValueError, match="non-increasing"):
        parse_retain_actions(_envelope(_retain(directives)), strict=True)


def test_increasing_priority_across_actions_rejected_when_strict():
    env = _envelope(
        _retain([{"start": 0, "end": 16, "priority": 30}], action_id="a1"),
        _retain([{"start": 16, "end": 32, "priority": 80}], action_id="a2"),
    )
    with pytest.raises(ValueError, match="non-increasing"):
        parse_retain_actions(env, strict=True)


def test_increasing_priority_dropped_when_lenient():
    env = _envelope(
        _retain(
            [
                {"start": 0, "end": 100, "priority": 30},
                {"start": 100, "end": 200, "priority": 80},
            ]
        )
    )
    assert parse_retain_actions(env, strict=False) is None


def test_non_increasing_priorities_valid():
    env = _envelope(
        _retain(
            [
                {"start": 0, "end": 100, "priority": 90},
                {"start": 100, "end": 200, "priority": 60},
                {"start": 200, "end": 300, "priority": 60},
            ]
        )
    )
    assert parse_retain_actions(env, strict=True) is not None


@pytest.mark.parametrize(
    "bad",
    [
        {"start": 0, "end": 16},  # no priority
        {"start": 0, "end": 16, "priority": 101},
        {"start": 0, "end": 16, "priority": -1},
        {"start": -1, "end": 16, "priority": 10},
        {"start": 16, "end": 16, "priority": 10},  # end must exceed start
        {"start": 0, "end": 16, "priority": 10, "duration": -1.0},
        {"start": 0, "end": 16, "priority": 10, "bogus": 1},
        {"covers_output": "false", "priority": 10},
        {"covers_output": True, "start": 0, "end": 16, "priority": 10},
        "not-an-object",
    ],
)
def test_invalid_directive_rejected_when_strict(bad):
    with pytest.raises(ValueError):
        parse_retain_actions(_envelope(_retain([bad])), strict=True)


def test_bool_priority_rejected():
    with pytest.raises(ValueError, match="priority"):
        parse_retain_actions(
            _envelope(_retain([{"start": 0, "end": 16, "priority": True}])), strict=True
        )


def test_invalid_action_skipped_when_lenient_but_valid_one_applies():
    env = _envelope(
        _retain([{"start": 0, "end": 16}], action_id="bad"),
        _retain([{"start": 0, "end": 16, "priority": 50}], action_id="good"),
    )
    parsed = parse_retain_actions(env, strict=False)
    assert parsed is not None
    assert parsed.directives == (RetainDirective(priority=50, start=0, end=16),)


def test_directives_not_a_list_rejected():
    with pytest.raises(ValueError, match="directives"):
        parse_retain_actions(
            _envelope(_action({"scope": "s", "directives": {}})), strict=True
        )


def test_unsupported_major_version_rejected_when_strict():
    env = _envelope(_retain([{"start": 0, "end": 16, "priority": 50}], version="2.0"))
    with pytest.raises(ValueError, match="version"):
        parse_retain_actions(env, strict=True)


def test_unsupported_major_version_skipped_when_lenient():
    env = _envelope(
        _retain(
            [{"start": 0, "end": 16, "priority": 90}], version="2.0", action_id="v2"
        ),
        _retain(
            [{"start": 0, "end": 16, "priority": 50}], version="1.3", action_id="v1"
        ),
    )
    parsed = parse_retain_actions(env, strict=False)
    assert parsed is not None
    assert parsed.directives == (RetainDirective(priority=50, start=0, end=16),)


def test_conflicting_scopes_rejected_when_strict():
    env = _envelope(
        _retain([{"start": 0, "end": 16, "priority": 50}], scope="a", action_id="a1"),
        _retain([{"start": 16, "end": 32, "priority": 50}], scope="b", action_id="a2"),
    )
    with pytest.raises(ValueError, match="scope"):
        parse_retain_actions(env, strict=True)


def test_conflicting_scope_action_skipped_when_lenient():
    env = _envelope(
        _retain([{"start": 0, "end": 16, "priority": 50}], scope="a", action_id="a1"),
        _retain([{"start": 16, "end": 32, "priority": 50}], scope="b", action_id="a2"),
    )
    parsed = parse_retain_actions(env, strict=False)
    assert parsed is not None
    assert parsed.scope == "a" and len(parsed.directives) == 1


_BAD_DURATIONS = [float("nan"), float("inf"), 10**400, MAX_DURATION_S + 1]
_BAD_DURATION_IDS = ["nan", "inf", "huge-int", "over-cap"]


@pytest.mark.parametrize("duration", _BAD_DURATIONS, ids=_BAD_DURATION_IDS)
def test_unbounded_duration_rejected_when_strict(duration):
    """A NaN expiry at the heap root would stop every other claim expiring."""
    directive = {"start": 0, "end": 16, "priority": 10, "duration": duration}
    with pytest.raises(ValueError, match="duration"):
        parse_retain_actions(_envelope(_retain([directive])), strict=True)


@pytest.mark.parametrize("duration", _BAD_DURATIONS, ids=_BAD_DURATION_IDS)
def test_unbounded_duration_skipped_when_lenient(duration):
    env = _envelope(
        _retain(
            [{"start": 0, "end": 16, "priority": 90, "duration": duration}],
            action_id="bad",
        ),
        _retain([{"start": 0, "end": 16, "priority": 50}], action_id="good"),
    )
    parsed = parse_retain_actions(env, strict=False)
    assert parsed is not None
    assert parsed.directives == (RetainDirective(priority=50, start=0, end=16),)


def _one_block_actions(count):
    return [
        _retain(
            [{"start": 16 * i, "end": 16 * (i + 1), "priority": 50}],
            action_id=f"a{i}",
        )
        for i in range(count)
    ]


def test_too_many_actions_rejected_when_strict():
    env = _envelope(*_one_block_actions(MAX_ACTIONS + 1))
    with pytest.raises(ValueError, match="actions"):
        parse_retain_actions(env, strict=True)


def test_too_many_actions_truncated_when_lenient():
    env = _envelope(*_one_block_actions(MAX_ACTIONS + 1))
    parsed = parse_retain_actions(env, strict=False)
    assert parsed is not None
    assert [d.start for d in parsed.directives] == [16 * i for i in range(MAX_ACTIONS)]


def _one_block_directives(first, count):
    return [
        {"start": 16 * i, "end": 16 * (i + 1), "priority": 50}
        for i in range(first, first + count)
    ]


def test_too_many_directives_rejected_when_strict():
    env = _envelope(
        _retain(_one_block_directives(0, MAX_DIRECTIVES - 1), action_id="a1"),
        _retain(_one_block_directives(MAX_DIRECTIVES - 1, 2), action_id="a2"),
    )
    with pytest.raises(ValueError, match="directives"):
        parse_retain_actions(env, strict=True)


def test_too_many_directives_truncated_in_order_when_lenient():
    env = _envelope(
        _retain(_one_block_directives(0, MAX_DIRECTIVES - 1), action_id="a1"),
        _retain(_one_block_directives(MAX_DIRECTIVES - 1, 2), action_id="a2"),
    )
    parsed = parse_retain_actions(env, strict=False)
    assert parsed is not None
    assert [d.start for d in parsed.directives] == [
        16 * i for i in range(MAX_DIRECTIVES)
    ]


def test_non_monotonic_action_dropped_alone_when_lenient():
    env = _envelope(
        _retain(
            [
                {"start": 0, "end": 16, "priority": 30},
                {"start": 16, "end": 32, "priority": 80},
            ],
            action_id="bad",
        ),
        _retain([{"start": 0, "end": 16, "priority": 50}], action_id="good"),
    )
    parsed = parse_retain_actions(env, strict=False)
    assert parsed is not None
    assert parsed.directives == (RetainDirective(priority=50, start=0, end=16),)
