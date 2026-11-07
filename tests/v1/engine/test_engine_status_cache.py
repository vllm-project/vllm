# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The client's FT status cache has two writers; a push must not drop either."""

from vllm.v1.engine.core_client import _update_engine_status


def test_push_replaces_engine_owned_keys():
    cache: dict[int, dict] = {0: {"id": 0, "status": "unhealthy", "fault_info": "RE"}}

    _update_engine_status(cache, 0, {"id": 0, "status": "healthy"})

    assert cache[0] == {"id": 0, "status": "healthy"}, (
        "a recovery push must clear the stale fault_info it no longer sends"
    )


def test_push_keeps_the_client_side_ft_apply_outcome():
    """handle_fault() records why an FT apply failed, for GET /status.

    The engine core knows nothing about those keys and does not resend them,
    so replacing the entry wholesale loses the operator's only record of the
    failed recovery attempt.
    """
    cache: dict[int, dict] = {
        1: {
            "id": 1,
            "status": "dead",
            "last_ft_request_id": "req-1",
            "ft_error": "status is DEAD",
        }
    }

    _update_engine_status(cache, 1, {"id": 1, "status": "unhealthy"})

    assert cache[1]["status"] == "unhealthy"
    assert cache[1]["last_ft_request_id"] == "req-1"
    assert cache[1]["ft_error"] == "status is DEAD"


def test_a_push_for_an_unseen_engine_seeds_the_entry():
    cache: dict[int, dict] = {}

    _update_engine_status(cache, 2, {"id": 2, "status": "healthy"})

    assert cache == {2: {"id": 2, "status": "healthy"}}


def test_a_push_that_carries_the_keys_itself_wins():
    """The engine core is still authoritative for anything it does send."""
    cache: dict[int, dict] = {0: {"id": 0, "ft_error": "old"}}

    _update_engine_status(cache, 0, {"id": 0, "status": "healthy", "ft_error": "new"})

    assert cache[0]["ft_error"] == "new"


def test_entries_for_other_engines_are_untouched():
    cache: dict[int, dict] = {
        0: {"id": 0, "status": "healthy"},
        1: {"id": 1, "status": "unhealthy", "ft_error": "boom"},
    }

    _update_engine_status(cache, 0, {"id": 0, "status": "unhealthy"})

    assert cache[1] == {"id": 1, "status": "unhealthy", "ft_error": "boom"}
