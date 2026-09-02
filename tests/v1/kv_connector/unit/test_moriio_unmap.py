# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MoRIIOConnectorScheduler.{map,unmap}_request_id.

vLLM's input_processor mutates a request_id after the connector has stored it
(e.g. ``cmpl-abc-0`` -> ``cmpl-abc-0-956053a4``), so an exact-match unmap can
miss. These tests cover the stable-``transfer_id`` fallback that handles that
case (and the clean exact-match and miss paths).

The map/unmap methods touch only the two id<->id dicts, so we bind them to a
lightweight stand-in rather than constructing a full scheduler.
"""

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
)

_map = MoRIIOConnectorScheduler.map_request_id
_unmap = MoRIIOConnectorScheduler.unmap_request_id


def _sched():
    return SimpleNamespace(
        transfer_id_to_request_id={},
        request_id_to_transfer_id={},
    )


def test_exact_match_unmap_clears_both_tables():
    s = _sched()
    _map(s, "rid-1", "tid-1")
    _unmap(s, "rid-1")
    assert s.request_id_to_transfer_id == {}
    assert s.transfer_id_to_request_id == {}


def test_unmap_via_transfer_id_after_request_id_mutation():
    s = _sched()
    # Connector stores the ORIGINAL request_id at map time.
    _map(s, "cmpl-abc-0", "tid-42")
    # input_processor later mutates the rid; exact match would miss, so the
    # stable transfer_id is used to find and clear the original entry.
    _unmap(s, "cmpl-abc-0-956053a4", transfer_id="tid-42")
    assert s.request_id_to_transfer_id == {}
    assert s.transfer_id_to_request_id == {}


def test_unmap_miss_is_noop_and_does_not_raise():
    s = _sched()
    _map(s, "rid-1", "tid-1")
    # Unknown rid and unknown/None transfer_id: must not raise, must not touch
    # the existing (unrelated) mapping.
    _unmap(s, "rid-unknown", transfer_id=None)
    _unmap(s, "rid-unknown", transfer_id="tid-unknown")
    assert s.request_id_to_transfer_id == {"rid-1": "tid-1"}
    assert s.transfer_id_to_request_id == {"tid-1": "rid-1"}


def test_exact_match_preferred_over_transfer_id_fallback():
    s = _sched()
    _map(s, "rid-1", "tid-1")
    # Exact rid is present, so it is removed directly even if a transfer_id is
    # also supplied; no stale entries left behind.
    _unmap(s, "rid-1", transfer_id="tid-1")
    assert s.request_id_to_transfer_id == {}
    assert s.transfer_id_to_request_id == {}
