# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm.v1.core.sched.kv_transfer_ledger import (
    KVTransferLedger,
    TransferDirection,
    TransferState,
)

EXPORT, IMPORT = TransferDirection.EXPORT, TransferDirection.IMPORT


def _req(rid: str):
    return SimpleNamespace(request_id=rid)


def test_export_lifecycle():
    ledger = KVTransferLedger()
    assert ledger.is_empty() and not ledger.has_exports()
    entry = ledger.open_export(_req("a"))
    assert entry.state == TransferState.OPEN
    assert ledger.has_exports() and ledger.has_retained() and ledger.needs_driving()
    assert ledger.is_referenced("a")
    # Opening twice is idempotent.
    assert ledger.open_export(_req("a")) is entry and len(ledger) == 1
    assert ledger.close("a", EXPORT) is entry
    assert ledger.close("a", EXPORT) is None
    assert ledger.is_empty() and not ledger.has_retained()


def test_import_pending_then_open_then_closed():
    ledger = KVTransferLedger()
    r = _req("b")
    entry = ledger.open_import(r)
    assert entry.state == TransferState.PENDING
    # A pending import does not retain finished blocks but needs driving
    # (lease renewals).
    assert ledger.needs_driving() and not ledger.has_retained()
    assert ledger.import_started(r) is entry
    assert entry.state == TransferState.OPEN
    assert ledger.complete("b", IMPORT) is entry
    assert entry.state == TransferState.CLOSING and entry.outcome == "ok"
    assert ledger.close("b", IMPORT) is entry
    assert ledger.is_empty()


def test_finished_request_with_open_import_is_retained():
    ledger = KVTransferLedger()
    r = _req("c")
    ledger.import_started(r)
    assert not ledger.has_retained()
    ledger.mark_finished(r)
    ledger.mark_finished(r)  # idempotent
    assert ledger.has_retained()
    ledger.close("c", IMPORT)
    assert not ledger.has_retained() and ledger.is_empty()


def test_request_referenced_until_every_entry_closes():
    ledger = KVTransferLedger()
    r = _req("d")
    ledger.import_started(r)
    ledger.open_export(r)
    ledger.close("d", IMPORT)
    assert ledger.is_referenced("d")  # the export still pins the blocks
    ledger.close("d", EXPORT)
    assert not ledger.is_referenced("d")


def test_drop_request_removes_all_and_fixes_counters():
    ledger = KVTransferLedger()
    r = _req("e")
    ledger.import_started(r)
    ledger.mark_finished(r)
    ledger.open_export(r)
    assert len(ledger.drop_request("e")) == 2
    assert ledger.is_empty() and not ledger.has_retained() and not ledger.has_exports()
    assert ledger.stats() == {
        "exports": 0, "imports_pending": 0, "imports_open": 0, "closing": 0
    }


def test_complete_unknown_is_none():
    ledger = KVTransferLedger()
    assert ledger.complete("zzz", EXPORT) is None
    assert ledger.close("zzz", IMPORT) is None


def test_received_import_does_not_need_driving():
    ledger = KVTransferLedger()
    r = _req("f")
    ledger.open_import(r)
    assert ledger.needs_driving()           # pending: renewals
    ledger.import_started(r)
    assert ledger.needs_driving()           # open: completion
    ledger.complete("f", IMPORT)
    assert not ledger.needs_driving()       # closing: only promotion is left
    assert not ledger.is_empty()
    ledger.close("f", IMPORT)
    assert ledger.is_empty() and not ledger.needs_driving()


def test_drop_exports_keeps_imports():
    ledger = KVTransferLedger()
    ledger.open_export(_req("g"))
    ledger.open_export(_req("h"))
    ledger.import_started(_req("i"))
    dropped = ledger.drop_exports()
    assert sorted(e.request_id for e in dropped) == ["g", "h"]
    assert not ledger.has_exports() and ledger.needs_driving()
    assert ledger.is_referenced("i") and not ledger.is_referenced("g")


def test_pins_blocks_only_for_open_imports_and_exports():
    ledger = KVTransferLedger()
    r = _req("j")
    ledger.open_import(r)
    assert not ledger.pins_blocks() and ledger.needs_driving()   # pending
    ledger.import_started(r)
    assert ledger.pins_blocks()                                  # open
    ledger.complete("j", IMPORT)
    assert not ledger.pins_blocks()                              # received
    ledger.open_export(_req("k"))
    assert ledger.pins_blocks()
    ledger.drop_exports()
    assert not ledger.pins_blocks()
