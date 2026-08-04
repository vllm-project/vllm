# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ``StorageManager.l2_adapters()`` — the lookup the
``DELETE /l2`` / ``GET /l2/keys`` / ``GET /l2/adapters`` HTTP handlers
(and any future admin/coordinator tooling) use to reach configured L2
adapters.

Bypasses ``StorageManager.__init__`` (which requires CUDA, an
L1Manager, and a full controller stack) and instead instantiates the
class via ``__new__`` with only the two attributes the method reads:
``_l2_adapters`` and ``_adapter_descriptors``.
"""

# Standard
from dataclasses import dataclass
from typing import cast
import threading

# First Party
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.storage_controllers.store_policy import AdapterDescriptor
from lmcache.v1.distributed.storage_manager import StorageManager


@dataclass
class _StubDescriptor:
    """Replaces ``AdapterDescriptor`` — only ``type_name`` is read."""

    type_name: str


class _StubAdapter:
    """Identity-only stub. ``l2_adapters`` returns it by reference;
    none of its methods are invoked by the unit under test."""


def _make_sm(adapters: list[_StubAdapter], names: list[str]) -> StorageManager:
    sm = StorageManager.__new__(StorageManager)
    # ``_StubAdapter`` / ``_StubDescriptor`` only implement the surface
    # the method under test actually reads. Cast through the real types
    # for mypy. Adapters/descriptors are keyed by stable adapter id.
    sm._adapters_lock = threading.Lock()
    sm._l2_adapters = cast("dict[int, L2AdapterInterface]", dict(enumerate(adapters)))
    sm._adapter_descriptors = cast(
        "dict[int, AdapterDescriptor]",
        {i: _StubDescriptor(type_name=n) for i, n in enumerate(names)},
    )
    return sm


# =============================================================================
# l2_adapters
# =============================================================================


class TestL2Adapters:
    def test_returns_all_adapters_in_configuration_order(self):
        a1, a2 = _StubAdapter(), _StubAdapter()
        sm = _make_sm([a1, a2], ["s3", "fs"])

        adapters = sm.l2_adapters()

        # Pair-up matches configuration order; first element is primary.
        assert len(adapters) == 2
        assert adapters[0][1] is a1
        assert adapters[0][0].type_name == "s3"
        assert adapters[1][1] is a2
        assert adapters[1][0].type_name == "fs"

    def test_empty_when_no_adapters_configured(self):
        sm = _make_sm([], [])
        # Empty list — callers (typically HTTP handlers) decide how to
        # surface it. The SM does not raise on its own.
        assert sm.l2_adapters() == []

    def test_single_adapter_round_trip(self):
        a = _StubAdapter()
        sm = _make_sm([a], ["s3"])

        adapters = sm.l2_adapters()

        assert len(adapters) == 1
        desc, adapter = adapters[0]
        assert adapter is a
        assert desc.type_name == "s3"

    def test_each_call_re_reads_the_list(self):
        # The docstring promises the method re-reads ``_l2_adapters`` on
        # every call so a runtime reconfigure (which swaps the adapter
        # list) is picked up by the next caller. Simulate the swap by
        # mutating ``_l2_adapters`` between two calls.
        a1, a2 = _StubAdapter(), _StubAdapter()
        sm = _make_sm([a1], ["s3"])

        first = sm.l2_adapters()
        assert first[0][1] is a1
        assert first[0][0].type_name == "s3"

        # Reconfigure: swap a1 → a2 (and the descriptor with it).
        sm._l2_adapters = cast("dict[int, L2AdapterInterface]", {0: a2})
        sm._adapter_descriptors = cast(
            "dict[int, AdapterDescriptor]", {0: _StubDescriptor(type_name="fs")}
        )

        second = sm.l2_adapters()
        assert second[0][1] is a2
        assert second[0][0].type_name == "fs"

    def test_returned_list_is_independent_of_internal_state(self):
        # Callers may mutate the returned list without affecting SM
        # state — the method returns a fresh copy.
        a = _StubAdapter()
        sm = _make_sm([a], ["s3"])
        snapshot = sm.l2_adapters()
        snapshot.clear()
        assert len(sm.l2_adapters()) == 1
