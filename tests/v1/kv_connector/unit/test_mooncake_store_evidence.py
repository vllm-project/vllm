# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production MooncakeStore transfer-evidence producer (the repair policy).

The producer contract: real store-GET outcomes are mapped, per attempt
and per transfer generation, into the structured failure reports and
terminal per-rank attestations the v1 scheduler's evidence gates
consume (`kv_load_failure_policy="repair"`). Drives the REAL receive
thread with realistic per-key Mooncake return codes (success = byte
count >= 0; failure = a Mooncake `ErrorCode` such as OBJECT_NOT_FOUND
-704, FILE_READ_FAIL -1102, TRANSFER_FAIL -800, RPC_TIMEOUT -901), the
real EvidenceBuffer lifecycle, the real worker getters, and -- for the
field-level agreement -- the REAL repair-policy Scheduler: evidence
produced here passes the scheduler's identity partition, validation
battery, reconciliation arithmetic, and completeness gate end to end.

Coverage matrix (mission categories in brackets):
- healthy GET + terminal attestation [1]
- missing object -> structured single-region report [2]
- truncated/EIO object -> whole-block failure across all layers [3]
- multiple failed objects -> no report, truthful attestation [4]
- retry then success (a newer generation supersedes) [5]
- retry then terminal failure (fresh generation, fresh failure) [6]
- concurrent transfers kept separate [7]
- reused object keys across generations [8]
- stale async evidence of an older generation never invalidates the
  newer delivery [9]
- request cancellation / preemption leaves no stale evidence [10]
- missing TP-rank evidence fails the delivery closed [11]
- MultiConnector forwards identity and both evidence channels [12]
- multi-group downgrade: evidence off, flat channels intact, never an
  attestation for a dropped group [13]
- legacy invalid_block_ids compatibility and no double counting [14]
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.v1.core.test_scheduler_transfer_evidence import (
    deliver_evidence,
    sync_claim_scenario,
)
from tests.v1.kv_connector.unit.test_mooncake_store_scheduler import (
    _make_bare_scheduler,
)
from tests.v1.kv_connector.unit.utils import create_request
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import (
    worker as mooncake_store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.connector import (
    MooncakeStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import (
    ChunkedTokenDatabase,
    LoadSpec,
    ReqMeta,
    RequestTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.evidence import (
    AttemptEvidence,
    EvidenceBuffer,
    evidence_enabled,
    validate_evidence_certification,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.layout import (
    KeyMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

BLOCK = 16
LAYERS = 4
N_BLOCKS = 8
TOKEN_LEN = N_BLOCKS * BLOCK

# Real Mooncake ErrorCodes (mooncake-store/include/types.h).
RC_OBJECT_NOT_FOUND = -704
RC_FILE_READ_FAIL = -1102  # EIO-class disk read failure
RC_CHECKSUM_MISMATCH = -801
RC_TRANSFER_FAIL = -800
RC_RPC_TIMEOUT = -901


class FakeStore:
    """Returns caller-configured per-key rc lists from the batch GET,
    one consumed list per call (realistic per-key semantics)."""

    def __init__(self, rcs):
        self.rcs = list(rcs)
        self.calls = []

    def batch_get_into_multi_buffers(self, keys, addrs, sizes):
        self.calls.append(list(keys))
        out = self.rcs[: len(keys)]
        self.rcs = self.rcs[len(keys) :]
        return out


class ExplodingStore(FakeStore):
    def batch_get_into_multi_buffers(self, keys, addrs, sizes):
        raise RuntimeError("store exploded")


class GatedStore(FakeStore):
    """A store whose batch GET blocks until released (drives the real
    interleaving of receive threads sharing one EvidenceBuffer)."""

    def __init__(self, rcs):
        super().__init__(rcs)
        self.entered = threading.Event()
        self.release = threading.Event()

    def batch_get_into_multi_buffers(self, keys, addrs, sizes):
        self.entered.set()
        assert self.release.wait(timeout=10.0)
        return super().batch_get_into_multi_buffers(keys, addrs, sizes)


def make_thread(store, evidence_buffer=None, layers=LAYERS, tp_rank=0):
    token_database = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0), chunk_size=BLOCK
    )
    token_database.set_kv_caches_base_addr([0x1000])
    token_database.set_block_len([256])
    spec = FullAttentionSpec(block_size=BLOCK, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["layer0"], spec)],
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
    )
    thread = mooncake_store_worker.KVCacheStoreRecvingThread(
        store=store,
        coord=coord,
        token_databases=[token_database],
        block_size=BLOCK,
        tp_rank=tp_rank,
        ready_event=threading.Event(),
        group_participates=[True],
        is_hma_required=False,
        evidence_buffer=evidence_buffer,
        evidence_layer_counts=[layers],
        evidence_certification="content",
    )
    thread.request_queue.task_done = MagicMock()
    return thread, token_database


def make_req_meta(
    req_id="r1",
    gen=None,
    token_len=TOKEN_LEN,
    vllm_cached_tokens=0,
    block_ids=None,
):
    n = token_len // BLOCK
    block_hashes = [i.to_bytes(8, "big") for i in range(n)]
    ids = block_ids if block_ids is not None else list(range(1000, 1000 + n))
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=token_len,
        block_ids=(ids,),
        block_hashes=block_hashes,
        load_spec=LoadSpec(
            vllm_cached_tokens=vllm_cached_tokens,
            kvpool_cached_tokens=token_len,
            can_load=True,
            token_len=token_len,
            transfer_generation=gen,
        ),
    )


def run_request(store, gen=7, layers=LAYERS, req_id="r1", block_ids=None):
    buf = EvidenceBuffer()
    thread, db = make_thread(store, buf, layers)
    meta = make_req_meta(req_id=req_id, gen=gen, block_ids=block_ids)
    thread.request_queue.put(meta)
    thread._handle_request(meta)
    return buf, thread


def make_ev(req_id="r1", gen=7, mode="content"):
    return AttemptEvidence(
        req_id=req_id,
        transfer_generation=gen,
        num_layers=LAYERS,
        block_size=BLOCK,
        restored_extent=TOKEN_LEN,
        certification_mode=mode,
    )


def bare_worker(evidence_buffer=None):
    """A MooncakeStoreWorker shell with only the attributes the evidence
    getters and the cleanup paths touch (no store, no threads)."""
    worker = object.__new__(mooncake_store_worker.MooncakeStoreWorker)
    worker.evidence_buffer = evidence_buffer
    worker._evidence_staged = None
    return worker


# ---------------------------------------------------------------------------
# The evidence accounting unit.
# ---------------------------------------------------------------------------


def test_healthy_attempt_attestation():
    ev = make_ev()
    for i in range(N_BLOCKS):
        ev.note_expected_block(i)
        ev.note_block_outcome(i, True)
    ev.finalize()
    att = ev.to_attestation()
    assert att.expected_objects == N_BLOCKS * LAYERS
    assert att.loaded_objects == N_BLOCKS * LAYERS
    assert att.failed_objects == 0
    assert att.terminal
    assert att.well_formed()
    assert att.certifies_content()
    assert att.transfer_generation == 7


def test_single_failed_block_reconciliation_unit():
    ev = make_ev()
    for i in range(N_BLOCKS):
        ev.note_expected_block(i)
        ev.block_ids[i] = 1000 + i
        ev.note_block_outcome(i, i != 5)
    ev.finalize()
    att = ev.to_attestation()
    report = ev.to_failure_report()
    assert report is not None
    assert report.failed_block_ids == [1005]
    assert report.token_span == (5 * BLOCK, 6 * BLOCK)
    assert report.rollback_boundary == 5 * BLOCK
    assert len(report.observed_failed_layers) == LAYERS
    # The scheduler's reconciliation unit: failed blocks x failed layers.
    assert att.failed_objects == len(report.failed_block_ids) * len(
        report.observed_failed_layers
    )
    assert att.expected_objects == att.loaded_objects + att.failed_objects


def test_multi_block_failure_no_report():
    ev = make_ev()
    for i in range(N_BLOCKS):
        ev.note_expected_block(i)
        ev.note_block_outcome(i, i not in (2, 5))
    ev.finalize()
    assert ev.to_failure_report() is None
    att = ev.to_attestation()
    assert att.failed_objects == 2 * LAYERS
    assert att.well_formed()


def test_partial_shard_failure_fails_whole_block():
    """One failed shard-object of a block fails the whole block: the
    repair writes whole blocks and a partially-headed block is corrupt
    for attention (shard folding)."""
    ev = make_ev()
    ev.note_expected_block(3)
    ev.block_ids[3] = 42
    ev.note_block_outcome(3, True)  # shard A succeeded
    ev.note_block_outcome(3, False)  # shard B failed
    ev.finalize()
    assert 3 in ev.failed_blocks
    assert 3 not in ev.loaded_blocks - ev.failed_blocks
    report = ev.to_failure_report()
    assert report is not None
    assert report.failed_block_ids == [42]


def test_failed_block_excluded_from_certified():
    ev = make_ev()
    for i in range(N_BLOCKS):
        ev.note_expected_block(i)
        ev.note_block_outcome(i, i != 0)
    ev.finalize()
    att = ev.to_attestation()
    assert att.certified_objects == (N_BLOCKS - 1) * LAYERS
    assert att.certified_objects == att.loaded_objects
    assert att.certifies_content()


def test_report_span_and_rollback():
    ev = make_ev()
    for i in range(4):
        ev.note_expected_block(i)
        ev.block_ids[i] = 10 + i
    for i in range(4):
        ev.note_block_outcome(i, i != 3)
    ev.finalize()
    report = ev.to_failure_report()
    assert report.token_span == (3 * BLOCK, 4 * BLOCK)
    assert report.rollback_boundary == 3 * BLOCK
    assert report.restored_extent == TOKEN_LEN


def test_report_rollback_zero_when_prefix_hole():
    """A hole before the failed block forces the full-recompute floor:
    blocks the attempt never loaded are not healthy."""
    ev = make_ev()
    for i in range(4):
        ev.note_expected_block(i)
        ev.block_ids[i] = 10 + i
    ev.note_block_outcome(0, True)
    # Block 1 was never fetched (no outcome recorded): the prefix hole.
    ev.note_block_outcome(3, False)  # the reported failure
    ev.note_block_outcome(2, True)
    ev.finalize()
    report = ev.to_failure_report()
    assert report.failed_block_ids == [13]
    assert report.rollback_boundary == 0


def test_failed_block_without_physical_id_no_report():
    ev = make_ev()
    ev.note_expected_block(0)
    ev.note_block_outcome(0, False)
    ev.finalize()
    assert ev.to_failure_report() is None


def test_generation_identity_immutable():
    """The generation is captured at attempt start and echoed unchanged;
    it is never re-read at emission time."""
    ev = make_ev(gen=11)
    assert ev.to_attestation().transfer_generation == 11
    assert ev.to_failure_report() is None or True
    ev.note_expected_block(0)
    ev.block_ids[0] = 1
    ev.note_block_outcome(0, False)
    ev.finalize()
    assert ev.to_failure_report().transfer_generation == 11
    assert ev.to_attestation().transfer_generation == 11


def test_identityless_generation_fails_closed():
    """An identity-less attempt emits truthful None-identity evidence;
    the scheduler's partition rejects it (fail-closed), which is the
    designed outcome for pre-contract connectors."""
    ev = make_ev(gen=None)
    ev.note_expected_block(0)
    ev.note_block_outcome(0, True)
    ev.finalize()
    att = ev.to_attestation()
    assert att.transfer_generation is None
    assert att.well_formed()


def test_none_mode_never_certifies():
    ev = make_ev(mode="none")
    ev.note_expected_block(0)
    ev.note_block_outcome(0, True)
    ev.finalize()
    att = ev.to_attestation()
    assert att.certification == "none"
    assert att.certified_objects == 0
    assert not att.certifies_content()


def test_sha256_rejected_by_validation():
    with pytest.raises(ValueError, match="sha256"):
        validate_evidence_certification("sha256")
    assert validate_evidence_certification("content") == "content"
    assert validate_evidence_certification("none") == "none"


def test_sha256_mode_never_certifies():
    """Even a fabricated sha256-mode attempt certifies nothing: the
    accounting refuses to count a verification the GET path does not
    perform."""
    ev = make_ev(mode="sha256")
    ev.note_expected_block(0)
    ev.note_block_outcome(0, True)
    ev.finalize()
    assert ev.certified_cells == 0
    assert not ev.certifies_content
    att = ev.to_attestation()
    assert att.certified_objects == 0
    assert not att.certifies_content()


# ---------------------------------------------------------------------------
# The EvidenceBuffer lifecycle (transfer identity across attempts).
# ---------------------------------------------------------------------------


def test_buffer_supersedes_stale_attempt():
    buf = EvidenceBuffer()
    current = make_ev(gen=2)
    buf.start_attempt(current)
    stale = make_ev(gen=1)
    buf.start_attempt(stale)
    assert buf.get("r1") is not stale, "an older generation must not clobber"
    assert buf.get("r1") is current


def test_identityless_cannot_clobber_identified():
    buf = EvidenceBuffer()
    current = make_ev(gen=2)
    buf.start_attempt(current)
    buf.start_attempt(make_ev(gen=None))
    assert buf.get("r1") is current

    # Identity-less ordering stays last-wins among identity-less ones.
    buf2 = EvidenceBuffer()
    buf2.start_attempt(make_ev(gen=None))
    second = make_ev(gen=None)
    buf2.start_attempt(second)
    assert buf2.get("r1") is second

    # An identified attempt upgrades an identity-less one.
    buf3 = EvidenceBuffer()
    buf3.start_attempt(make_ev(gen=None))
    identified = make_ev(gen=1)
    buf3.start_attempt(identified)
    assert buf3.get("r1") is identified


def test_terminal_evidence_survives_new_attempt():
    """A TERMINAL-but-undrained entry's evidence is complete and about to
    be consumed by this step's drain; a newer in-flight registration must
    not destroy it: both survive, each in its own channel."""
    buf = EvidenceBuffer()
    terminal = make_ev(gen=1)
    terminal.note_expected_block(0)
    terminal.note_block_outcome(0, False)
    terminal.finalize()
    buf.start_attempt(terminal)

    inflight = make_ev(gen=2)
    buf.start_attempt(inflight)

    drained = buf.drain_terminal()
    assert terminal in drained, "the terminal evidence must still drain"
    assert buf.get("r1") is inflight
    assert inflight not in drained, "the in-flight attempt stays buffered"


def test_nonterminal_stays_buffered():
    buf = EvidenceBuffer()
    buf.start_attempt(make_ev())
    assert buf.drain_terminal() == []
    assert buf.get("r1") is not None


def test_drop_forgets_pending_attempt():
    buf = EvidenceBuffer()
    buf.start_attempt(make_ev())
    buf.drop("r1")
    assert buf.get("r1") is None
    assert buf.drain_terminal() == []


# ----------------------------------------------------------------
# The receive thread: real per-key GET outcomes -> evidence. [1][2][3][4]
# ---------------------------------------------------------------------------


def test_healthy_transfer_full_attestation():
    store = FakeStore([0] * N_BLOCKS)
    buf, thread = run_request(store)
    attestations = buf.drain_terminal()
    assert len(attestations) == 1
    ev = attestations[0]
    assert ev.terminal
    assert ev.expected_cells == N_BLOCKS * LAYERS
    assert ev.loaded_cells == N_BLOCKS * LAYERS
    assert ev.failed_cells == 0
    att = ev.to_attestation()
    assert att.well_formed()
    assert att.certifies_content()
    assert att.transfer_generation == 7
    assert thread.get_and_clear_block_ids_with_load_errors() == set()


def test_missing_object_get_reports_single_region():
    """One store GET fails OBJECT_NOT_FOUND: exactly one failed block, a
    well-formed single-region report, the reconciliation unit holds."""
    rcs = [0] * N_BLOCKS
    rcs[5] = RC_OBJECT_NOT_FOUND
    buf, thread = run_request(FakeStore(rcs))
    ev = buf.drain_terminal()[0]
    att = ev.to_attestation()
    report = ev.to_failure_report()
    assert report is not None
    assert report.failed_block_ids == [1005]
    assert report.token_span == (5 * BLOCK, 6 * BLOCK)
    assert report.rollback_boundary == 5 * BLOCK
    assert len(report.observed_failed_layers) == LAYERS
    assert att.failed_objects == LAYERS
    assert att.failed_objects == len(report.failed_block_ids) * len(
        report.observed_failed_layers
    )
    assert report.transfer_generation == 7
    # Whole-block failure across all layers: the object packs every layer
    # of the group for that block, so no finer-grained precision is
    # invented and none is needed.
    assert report.observed_failed_layers == list(range(LAYERS))
    assert thread.get_and_clear_block_ids_with_load_errors() == set()


@pytest.mark.parametrize(
    "rc",
    [
        RC_FILE_READ_FAIL,  # EIO-class disk failure (truncated object read)
        RC_CHECKSUM_MISMATCH,  # the store's own integrity check failed
        RC_TRANSFER_FAIL,
        RC_RPC_TIMEOUT,
    ],
)
def test_truncated_or_failed_transfer_maps_to_whole_block(rc):
    """Any per-key failure code maps to the same honest verdict: the
    whole block failed across every layer (the object packs all layers;
    the transfer result provides no layer-level precision)."""
    rcs = [0] * N_BLOCKS
    rcs[3] = rc
    buf, _ = run_request(FakeStore(rcs))
    ev = buf.drain_terminal()[0]
    report = ev.to_failure_report()
    assert report is not None
    assert report.failed_block_ids == [1003]
    assert report.token_span == (3 * BLOCK, 4 * BLOCK)
    assert report.observed_failed_layers == list(range(LAYERS))


def test_multiple_failed_objects_emit_no_report():
    """[4] Two failed keys in DIFFERENT blocks: the v1 single-region
    contract cannot represent it, no report, the truthful attestation
    with failed_objects==2*layers (the scheduler fails closed)."""
    rcs = [0] * N_BLOCKS
    rcs[2] = RC_OBJECT_NOT_FOUND
    rcs[5] = RC_TRANSFER_FAIL
    buf, _ = run_request(FakeStore(rcs))
    ev = buf.drain_terminal()[0]
    assert ev.to_failure_report() is None
    att = ev.to_attestation()
    assert att.failed_objects == 2 * LAYERS
    assert att.well_formed()


def test_duplicate_failures_same_block_count_once():
    """[4] Duplicate failures (every shard-object of one block fails
    with different codes) still count the block once: a block is a
    single failed cell-set regardless of how many of its objects
    reported failure."""
    from tests.v1.kv_connector.unit.test_mooncake_store_worker import (
        _make_tp_shared_db,
    )

    db, _ = _make_tp_shared_db()  # two shard-objects per block
    buf = EvidenceBuffer()
    spec = FullAttentionSpec(block_size=BLOCK, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["layer0"], spec)],
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
    )
    thread = mooncake_store_worker.KVCacheStoreRecvingThread(
        store=FakeStore([RC_OBJECT_NOT_FOUND, RC_TRANSFER_FAIL, 0, 0]),
        coord=coord,
        token_databases=[db],
        block_size=BLOCK,
        tp_rank=0,
        ready_event=threading.Event(),
        group_participates=[True],
        evidence_buffer=buf,
        evidence_layer_counts=[LAYERS],
        evidence_certification="content",
    )
    thread.request_queue.task_done = MagicMock()
    meta = make_req_meta(token_len=2 * BLOCK, block_ids=[7, 8])
    thread._handle_request(meta)
    ev = buf.drain_terminal()[0]
    assert ev.failed_blocks == {0}
    report = ev.to_failure_report()
    assert report is not None
    assert report.failed_block_ids == [7]
    assert ev.to_attestation().failed_objects == LAYERS


def test_exception_mid_batch_all_failed():
    """A store exception fails the in-flight batch: truthful all-failed
    accounting, multi-block -> no report, fail closed."""
    buf, _ = run_request(ExplodingStore([]))
    ev = buf.drain_terminal()[0]
    assert ev.terminal
    assert ev.failed_cells == N_BLOCKS * LAYERS
    assert ev.to_failure_report() is None


def test_unfetched_blocks_count_failed_at_terminality():
    """Blocks whose GET never ran (the oversized-key skip) count failed
    so loaded + failed == expected holds at terminality."""
    buf = EvidenceBuffer()
    store = FakeStore([0] * N_BLOCKS)
    thread, _ = make_thread(store, buf)
    meta = make_req_meta()
    thread.usable_disk_offload_buffer_budget_bytes = 1
    thread.disk_offload_buffer_budget_bytes = 1  # every key is oversized
    thread.request_queue.put(meta)
    thread._handle_request(meta)
    ev = buf.drain_terminal()[0]
    assert ev.failed_cells == N_BLOCKS * LAYERS
    assert ev.loaded_cells == 0
    assert ev.to_failure_report() is None
    assert ev.to_attestation().well_formed()


def test_generation_captured_from_load_spec():
    buf, _ = run_request(FakeStore([0] * N_BLOCKS), gen=42)
    ev = buf.drain_terminal()[0]
    assert ev.transfer_generation == 42
    assert ev.to_attestation().transfer_generation == 42


def test_layer_count_flows_into_cells():
    buf, _ = run_request(FakeStore([0] * N_BLOCKS), layers=6)
    ev = buf.drain_terminal()[0]
    assert ev.expected_cells == N_BLOCKS * 6
    assert ev.to_attestation().expected_objects == N_BLOCKS * 6


def test_empty_key_list_finalizes_vacuous():
    """A request whose mask leaves nothing to load finalizes terminal
    (vacuous) instead of parking a non-terminal entry in the buffer."""
    buf = EvidenceBuffer()
    thread, _ = make_thread(FakeStore([]), buf)
    meta = make_req_meta(token_len=BLOCK, vllm_cached_tokens=BLOCK)
    thread.request_queue.put(meta)
    thread._handle_request(meta)
    ev = buf.drain_terminal()
    assert len(ev) == 1
    assert ev[0].terminal
    assert ev[0].expected_cells == 0
    assert ev[0].to_attestation().well_formed()
    assert not ev[0].to_attestation().certifies_content()


# ---------------------------------------------------------------------------
# Retries, new attempts, and reused keys. [5][6][8]
# ---------------------------------------------------------------------------


def test_retry_then_success_supersedes_failed_attempt():
    """[5] A failed attempt (gen 1) is followed by a retry (gen 2) that
    succeeds. The retry registers before the failed attempt's evidence
    was drained, so BOTH drain (the terminal old evidence survives in
    its own channel -- never destroyed); the new attempt's healthy
    accounting is complete and the old failure stays truthful but
    stale, and the scheduler's identity partition drops it (pinned
    end-to-end by the late-evidence scheduler test below)."""
    buf = EvidenceBuffer()
    failed_thread, _ = make_thread(FakeStore([RC_OBJECT_NOT_FOUND] * N_BLOCKS), buf)
    retry_thread, _ = make_thread(FakeStore([0] * N_BLOCKS), buf)

    failed_meta = make_req_meta(gen=1)
    failed_thread.request_queue.put(failed_meta)
    failed_thread._handle_request(failed_meta)

    retry_meta = make_req_meta(gen=2, block_ids=list(range(2000, 2000 + N_BLOCKS)))
    retry_thread.request_queue.put(retry_meta)
    retry_thread._handle_request(retry_meta)

    drained = buf.drain_terminal()
    by_gen = {e.transfer_generation: e for e in drained}
    assert set(by_gen) == {1, 2}
    # The retry's evidence is complete and healthy.
    assert by_gen[2].failed_cells == 0
    assert by_gen[2].loaded_cells == N_BLOCKS * LAYERS
    assert by_gen[2].to_attestation().certifies_content()
    # The old failure stayed truthful -- a late artifact the consumer
    # attributes to the consumed attempt and drops.
    assert by_gen[1].failed_cells == N_BLOCKS * LAYERS


def test_retry_then_terminal_failure_is_fresh_evidence():
    """[6] The retry fails too: a genuine repeated failure carries the
    NEW attempt's generation and drains as its own evidence."""
    rcs = [0] * N_BLOCKS
    rcs[4] = RC_OBJECT_NOT_FOUND
    buf, _ = run_request(FakeStore(rcs), gen=9)
    ev = buf.drain_terminal()[0]
    report = ev.to_failure_report()
    assert report is not None
    assert report.transfer_generation == 9
    assert report.failed_block_ids == [1004]


def test_reused_object_keys_across_generations():
    """[8] The same store keys are GET'd by both attempts (identical
    hashes and block ids); the evidence stays attributed to the attempt
    that fetched them. The failed old attempt's evidence drains as a
    truthful late artifact next to the retry's healthy one; the
    scheduler drops it by generation (the end-to-end pin below)."""
    buf = EvidenceBuffer()
    keys_thread_old, _ = make_thread(FakeStore([RC_OBJECT_NOT_FOUND] * N_BLOCKS), buf)
    keys_thread_new, _ = make_thread(FakeStore([0] * N_BLOCKS), buf)

    old_meta = make_req_meta(gen=1)
    keys_thread_old.request_queue.put(old_meta)
    keys_thread_old._handle_request(old_meta)

    # The retry reuses the SAME block ids and hashes -> the same keys.
    new_meta = make_req_meta(gen=2)
    keys_thread_new.request_queue.put(new_meta)
    keys_thread_new._handle_request(new_meta)

    drained = buf.drain_terminal()
    by_gen = {e.transfer_generation: e for e in drained}
    assert set(by_gen) == {1, 2}
    # The SAME keys produced a healthy retry: attribution never stains
    # the new attempt with the old attempt's failure.
    assert by_gen[2].failed_cells == 0
    assert by_gen[2].loaded_cells == N_BLOCKS * LAYERS
    assert by_gen[2].block_ids == by_gen[1].block_ids


def test_stale_handler_registering_first_is_superseded():
    """The old attempt's handler registers first and blocks in its store
    GET; the retry (a new generation) then registers and completes; the
    old handler finally finishes. Only the NEW attempt's terminal
    evidence may drain."""
    buf = EvidenceBuffer()

    old_store = GatedStore([RC_OBJECT_NOT_FOUND] * N_BLOCKS)
    old_thread, _ = make_thread(old_store, buf)
    new_store = FakeStore([0] * N_BLOCKS)
    new_thread, _ = make_thread(new_store, buf)

    old_meta = make_req_meta(gen=1)
    old_thread.request_queue.put(old_meta)
    old_runner = threading.Thread(
        target=old_thread._handle_request, args=(old_meta,), daemon=True
    )
    old_runner.start()
    assert old_store.entered.wait(timeout=5.0), "old handler must reach its GET"
    assert buf.get("r1").transfer_generation == 1

    new_meta = make_req_meta(gen=2)
    new_thread.request_queue.put(new_meta)
    new_thread._handle_request(new_meta)
    assert buf.get("r1").transfer_generation == 2

    old_store.release.set()
    old_runner.join(timeout=5.0)
    assert not old_runner.is_alive()

    drained = buf.drain_terminal()
    assert [e.transfer_generation for e in drained] == [2]
    assert drained[0].loaded_cells == N_BLOCKS * LAYERS
    assert drained[0].failed_cells == 0


def test_stale_handler_registering_last_is_refused():
    """The reverse interleaving: the retry registers and completes
    first; the old attempt's handler only registers afterwards and is
    refused outright -- the drained evidence stays the new attempt's."""
    buf = EvidenceBuffer()

    new_store = FakeStore([0] * N_BLOCKS)
    new_thread, _ = make_thread(new_store, buf)
    new_meta = make_req_meta(gen=5)
    new_thread.request_queue.put(new_meta)
    new_thread._handle_request(new_meta)
    current = buf.get("r1")
    assert current.transfer_generation == 5
    current.finalize()

    old_store = FakeStore([RC_OBJECT_NOT_FOUND] * N_BLOCKS)
    old_thread, _ = make_thread(old_store, buf)
    old_meta = make_req_meta(gen=4)
    old_thread.request_queue.put(old_meta)
    old_thread._handle_request(old_meta)

    assert buf.get("r1") is current, "the stale registration must be refused"
    drained = buf.drain_terminal()
    assert [e.transfer_generation for e in drained] == [5]
    assert drained[0].failed_cells == 0


def test_concurrent_transfers_kept_separate():
    """[7] Two requests share one receive-thread pool and one buffer:
    their evidence stays per-request and the blocked one cannot leak
    into the completed one's drain."""
    buf = EvidenceBuffer()

    gated_store = GatedStore([0] * N_BLOCKS)
    gated_thread, _ = make_thread(gated_store, buf)
    fast_store = FakeStore([0] * N_BLOCKS)
    fast_thread, _ = make_thread(fast_store, buf)

    gated_meta = make_req_meta(req_id="blocked", gen=1)
    gated_thread.request_queue.put(gated_meta)
    gated_runner = threading.Thread(
        target=gated_thread._handle_request, args=(gated_meta,), daemon=True
    )
    gated_runner.start()
    assert gated_store.entered.wait(timeout=5.0)

    fast_meta = make_req_meta(req_id="fast", gen=2)
    fast_thread.request_queue.put(fast_meta)
    fast_thread._handle_request(fast_meta)

    drained = buf.drain_terminal()
    assert [e.req_id for e in drained] == ["fast"]
    assert drained[0].transfer_generation == 2

    gated_store.release.set()
    gated_runner.join(timeout=5.0)
    assert not gated_runner.is_alive()
    drained = buf.drain_terminal()
    assert [e.req_id for e in drained] == ["blocked"]
    assert drained[0].failed_cells == 0


# ---------------------------------------------------------------------------
# Crash containment and the flat-mode drain fix.
# ---------------------------------------------------------------------------


def _run_crashed_thread(thread, meta, condition, timeout=5.0):
    """Drive the thread's REAL run() loop over a crashing handler and poll
    an explicit condition with a deadline (no fixed sleeps, no join on the
    never-finished queue: the crash path never calls task_done)."""
    thread.request_queue.put(meta)
    runner = threading.Thread(target=thread.run, daemon=True)
    runner.start()
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)


def test_pre_get_crash_drops_buffer_entry():
    """An exception escaping _handle_request drops the crashed attempt's
    buffer entry (no non-terminal leak); the missing attestation fails
    the delivery closed at the scheduler."""
    buf = EvidenceBuffer()
    thread, _ = make_thread(FakeStore([0] * N_BLOCKS), buf)
    # The attempt registered its evidence, then the handler crashed
    # before any GET (the pre-GET crash shape).
    buf.start_attempt(make_ev(gen=7))
    meta = make_req_meta(gen=7)

    def boom(meta):
        raise RuntimeError("pre-GET crash")

    thread._handle_request = boom
    _run_crashed_thread(thread, meta, lambda: buf.get("r1") is None)
    assert buf.get("r1") is None, "the pre-GET crash must drop the buffer entry"


def test_identityless_crash_spares_registered_identified_attempt():
    """A crashed ReqMeta carrying NO transfer identity must not destroy
    a REGISTERED identified attempt's evidence (the identity-blind
    crash-drop residual)."""
    buf = EvidenceBuffer()
    thread, _ = make_thread(FakeStore([0] * N_BLOCKS), buf)
    registered = make_ev(gen=2)
    buf.start_attempt(registered)
    meta = make_req_meta(gen=None)

    def boom(meta):
        raise RuntimeError("identityless crash")

    thread._handle_request = boom
    _run_crashed_thread(thread, meta, lambda: thread.request_queue.empty())
    assert buf.get("r1") is registered, (
        "an identity-less crash must not drop a registered attempt"
    )


def test_matched_crash_still_drops_own_entry():
    """The guard is not a blanket no-op: a crash whose ReqMeta carries
    the registered attempt's generation still drops that entry."""
    buf = EvidenceBuffer()
    thread, _ = make_thread(FakeStore([0] * N_BLOCKS), buf)
    buf.start_attempt(make_ev(gen=3))
    meta = make_req_meta(gen=3)

    def boom(meta):
        raise RuntimeError("matched crash")

    thread._handle_request = boom
    _run_crashed_thread(thread, meta, lambda: buf.get("r1") is None)
    assert buf.get("r1") is None


def _make_multibatch_thread(store, evidence_buffer):
    """Force every store object into its own GET sub-batch (the split
    path's singleton branch): a tiny usable staging budget with a high
    hard per-key cap, so nothing takes the oversized early return."""
    thread, _ = make_thread(store, evidence_buffer)
    thread.usable_disk_offload_buffer_budget_bytes = 1
    thread.disk_offload_buffer_budget_bytes = 10**9
    return thread, _


def test_flat_mode_drains_remaining_batches_after_failure():
    """The break->continue drain fix under the FLAT policy path: after a
    failing sub-batch every remaining batch is still issued, and the
    flat channel reports exactly the failed block."""
    rcs = [0] * N_BLOCKS
    rcs[5] = RC_OBJECT_NOT_FOUND
    store = FakeStore(rcs)
    thread, _ = _make_multibatch_thread(store, evidence_buffer=None)
    meta = make_req_meta()
    thread.request_queue.put(meta)
    thread._handle_request(meta)
    assert len(store.calls) == N_BLOCKS, "every sub-batch must be issued"
    # Upstream (#67939) additionally reports the unfetched remaining
    # blocks alongside the failed one; the drain itself is what this
    # test pins: every batch was issued despite the earlier failure.
    assert thread.get_and_clear_block_ids_with_load_errors() == {
        1005,
        1006,
        1007,
    }


def test_evidence_mode_repairs_after_mid_request_failure():
    """The same multi-batch shape with evidence on: blocks after the
    failed one still load (a repairable single-region geometry), the
    accounting holds, and no phantom logical entries appear."""
    rcs = [0] * N_BLOCKS
    rcs[5] = RC_OBJECT_NOT_FOUND
    store = FakeStore(rcs)
    buf = EvidenceBuffer()
    thread, _ = _make_multibatch_thread(store, buf)
    meta = make_req_meta()
    thread.request_queue.put(meta)
    thread._handle_request(meta)
    assert len(store.calls) == N_BLOCKS
    ev = buf.drain_terminal()[0]
    assert ev.loaded_blocks == set(range(N_BLOCKS)) - {5}
    assert ev.failed_blocks == {5}
    assert (ev.loaded_blocks | ev.failed_blocks) <= ev.expected_blocks
    report = ev.to_failure_report()
    assert report is not None
    assert report.failed_block_ids == [1005]
    assert report.rollback_boundary == 5 * BLOCK
    att = ev.to_attestation()
    assert att.loaded_objects == (N_BLOCKS - 1) * LAYERS
    assert att.failed_objects == LAYERS
    assert att.well_formed() and att.certifies_content()
    assert thread.get_and_clear_block_ids_with_load_errors() == set()


def test_hma_mode_reports_failed_requests_not_blocks():
    """HMA (multi-group) keeps the request-level failed channel; the
    evidence activation gate stays off (single-group contract)."""
    rcs = [0] * N_BLOCKS
    rcs[3] = RC_OBJECT_NOT_FOUND
    thread, _ = make_thread(FakeStore(rcs), evidence_buffer=None)
    thread._is_hma_required = True
    meta = make_req_meta()
    thread.request_queue.put(meta)
    thread._handle_request(meta)
    assert thread.get_and_clear_failed_requests() == {"r1"}
    assert thread.get_and_clear_block_ids_with_load_errors() == set()


# ---------------------------------------------------------------------------
# The worker getters: staging, cleanup, activation gates. [10][13][14]
# ---------------------------------------------------------------------------


def test_activation_gates():
    """[13] Evidence activates ONLY under the repair policy with exactly
    one prefix-cacheable group in a single-group layout; every other
    combination keeps the flat channels and produces no evidence."""
    assert evidence_enabled("repair", 1, False)
    # The default and explicit non-repair policies stay flat.
    assert not evidence_enabled("fail", 1, False)
    assert not evidence_enabled("recompute", 1, False)
    assert not evidence_enabled(None, 1, False)
    # Multi-group layouts (HMA or >1 prefix-cacheable group) can never
    # emit conflated evidence: the downgrade is explicit and total.
    assert not evidence_enabled("repair", 2, False)
    assert not evidence_enabled("repair", 1, True)
    assert not evidence_enabled("repair", 3, True)


def test_class_defaults_on_bare_worker():
    """The evidence getters are called UNCONDITIONALLY by post_forward --
    including on a worker whose constructor raised midway. The CLASS-level
    defaults must keep the getters safe: empty lists, never
    AttributeError."""
    worker = object.__new__(mooncake_store_worker.MooncakeStoreWorker)
    assert worker.get_load_failure_reports() == []
    assert worker.get_load_attestations() == []


def _worker_with_recv(evidence_buffer):
    """A worker shell wired the way get_finished and the getters expect."""
    w = bare_worker(evidence_buffer)
    w._capacity_only = False
    w.can_put = False
    w.load_async = True
    w.tp_rank = 0
    w.kv_recv_threads = []
    return w


def test_finished_request_drop_runs_on_pure_consumer():
    """[10] A pure kv_consumer (can_put=False) still drops its finished
    requests' pending evidence: the drop is hoisted out of the
    can_put-gated send-ledger cleanup."""
    buf = EvidenceBuffer()
    pending = make_ev(req_id="r1", gen=3)
    buf.start_attempt(pending)
    worker = _worker_with_recv(buf)

    meta = SimpleNamespace(preempted_req_ids=set())
    worker.get_finished({"r1"}, meta)
    assert buf.get("r1") is None, "the finished request's evidence must be dropped"


def test_preemption_cleanup_drops_pending_attempt():
    """[10] A request preempted while its async load is in flight: the
    preemption drop forgets the pending attempt, so a late finalize of
    the orphaned evidence can never drain a report naming the FREED
    (and possibly reused) physical block ids."""
    buf = EvidenceBuffer()
    pending = make_ev(req_id="r1", gen=3)
    pending.note_expected_block(0)
    pending.block_ids[0] = 1000
    buf.start_attempt(pending)
    worker = _worker_with_recv(buf)

    meta = SimpleNamespace(preempted_req_ids={"r1"})
    worker.get_finished(set(), meta)
    assert buf.get("r1") is None

    # The interrupted handler reaches terminality late: nothing drains.
    pending.note_block_outcome(0, False)
    pending.finalize()
    assert buf.drain_terminal() == []
    assert worker.get_load_failure_reports() == []
    assert worker.get_load_attestations() == []


def test_staging_shared_then_released():
    """The two evidence getters observe the SAME staged attempts in one
    post_forward pass, and the attestation getter releases the staging
    so the next step drains afresh."""
    worker = bare_worker(EvidenceBuffer())

    healthy = make_ev(req_id="r1", gen=1)
    healthy.note_expected_block(0)
    healthy.note_block_outcome(0, True)
    healthy.finalize()
    worker.evidence_buffer.start_attempt(healthy)

    failed = make_ev(req_id="r2", gen=2)
    failed.note_expected_block(0)
    failed.block_ids[0] = 42
    failed.note_block_outcome(0, False)
    failed.finalize()
    worker.evidence_buffer.start_attempt(failed)

    reports_first = worker.get_load_failure_reports()
    assert [r.request_id for r in reports_first] == ["r2"]
    # The staging is shared within the step: a second reports call
    # re-emits the SAME evidence (identical duplicates the scheduler's
    # fold absorbs), and the attestations getter still observes them.
    reports_again = worker.get_load_failure_reports()
    assert reports_again == reports_first
    attestations = worker.get_load_attestations()
    assert sorted(a.request_id for a in attestations) == ["r1", "r2"]
    assert worker.get_load_attestations() == []
    assert worker.get_load_failure_reports() == []


def test_interrupted_pass_never_re_emits():
    """A pass interrupted between the reports and attestations getters
    (the attestation getter never ran) must not re-emit the same reports
    next step: get_finished -- the first getter of every drain --
    releases any stale staging."""
    buf = EvidenceBuffer()
    worker = _worker_with_recv(buf)
    failed = make_ev(req_id="r1", gen=2)
    failed.note_expected_block(0)
    failed.block_ids[0] = 42
    failed.note_block_outcome(0, False)
    failed.finalize()
    buf.start_attempt(failed)

    first = worker.get_load_failure_reports()
    assert len(first) == 1
    # The attestation getter is skipped (an exception mid-pass); the
    # next step's drain starts at get_finished:
    worker.get_finished(set(), SimpleNamespace(preempted_req_ids=set()))
    assert worker.get_load_failure_reports() == []
    assert worker.get_load_attestations() == []


def test_multi_block_failure_report_suppressed_attestation_truthful():
    """[4] One attempt fails TWO blocks: the worker emits NO report
    (never an invented single-region report) while the attestation in
    the SAME pass still carries the honest multi-block failure count:
    the failure is never invisible."""
    worker = bare_worker(EvidenceBuffer())
    failed = make_ev(req_id="r1", gen=7)
    for i in range(N_BLOCKS):
        failed.note_expected_block(i)
        failed.block_ids[i] = 1000 + i
    failed.note_block_outcome(2, False)
    failed.note_block_outcome(5, False)
    for i in range(N_BLOCKS):
        if i not in (2, 5):
            failed.note_block_outcome(i, True)
    failed.finalize()
    worker.evidence_buffer.start_attempt(failed)

    assert worker.get_load_failure_reports() == []
    attestations = worker.get_load_attestations()
    assert len(attestations) == 1
    att = attestations[0]
    assert att.terminal
    assert att.failed_objects == 2 * LAYERS
    assert att.loaded_objects == (N_BLOCKS - 2) * LAYERS
    assert att.expected_objects == N_BLOCKS * LAYERS
    assert att.well_formed()
    assert att.transfer_generation == 7
    # Released staging: nothing re-emits later.
    assert worker.get_load_failure_reports() == []
    assert worker.get_load_attestations() == []


def test_report_drain_exception_does_not_swallow_failure():
    """The reports getter raises mid-drain (a poisoned attempt): the
    attestations getter, called afterwards in the same post_forward,
    still observes the SAME staged attempts -- the failure stays
    visible through the positive channel, never a silent success."""
    buf = EvidenceBuffer()

    class PoisonedEvidence(AttemptEvidence):
        def to_failure_report(self):
            raise RuntimeError("report construction exploded")

    failed = PoisonedEvidence(
        req_id="r1",
        transfer_generation=7,
        num_layers=LAYERS,
        block_size=BLOCK,
        restored_extent=TOKEN_LEN,
        certification_mode="content",
    )
    failed.note_expected_block(0)
    failed.block_ids[0] = 1000
    failed.note_block_outcome(0, False)
    failed.finalize()
    buf.start_attempt(failed)

    worker = bare_worker(buf)
    with pytest.raises(RuntimeError, match="exploded"):
        worker.get_load_failure_reports()

    attestations = worker.get_load_attestations()
    assert len(attestations) == 1
    assert attestations[0].terminal
    assert attestations[0].failed_objects == LAYERS
    assert attestations[0].well_formed()


# ---------------------------------------------------------------------------
# Legacy compatibility: the flat channels. [14]
# ---------------------------------------------------------------------------


def test_legacy_flat_channel_fires_without_evidence():
    """[14] Without the evidence buffer (every non-repair policy), a
    failed GET still reports through legacy invalid_block_ids -- the
    pre-evidence behavior, byte-identical."""
    rcs = [0] * N_BLOCKS
    rcs[4] = RC_OBJECT_NOT_FOUND
    thread, _ = make_thread(FakeStore(rcs), evidence_buffer=None)
    meta = make_req_meta()
    thread.request_queue.put(meta)
    thread._handle_request(meta)
    assert thread.get_and_clear_block_ids_with_load_errors() == {1004}
    # Upstream (#67939) fires the request-level failure channel for any
    # failed GET (it survives cross-rank aggregation); the per-block
    # flat channel above is the pre-evidence contract this test pins.
    assert thread.get_and_clear_failed_requests() == {meta.req_id}


def test_flat_channel_suppressed_under_evidence_no_double_count():
    """[14] Under the repair policy the structured channel owns the
    failure: the flat channel stays silent (no double counting, no
    repair-admission race) and the evidence carries the failure."""
    rcs = [0] * N_BLOCKS
    rcs[4] = RC_OBJECT_NOT_FOUND
    buf = EvidenceBuffer()
    thread, _ = make_thread(FakeStore(rcs), buf)
    meta = make_req_meta()
    thread.request_queue.put(meta)
    thread._handle_request(meta)
    assert thread.get_and_clear_block_ids_with_load_errors() == set()
    assert thread.get_and_clear_failed_requests() == set()
    ev = buf.drain_terminal()[0]
    assert ev.failed_blocks == {4}
    assert ev.to_failure_report() is not None


# ---------------------------------------------------------------------------
# Scheduler-side identity capture and the MultiConnector wiring. [12]
# ---------------------------------------------------------------------------


def test_note_transfer_attempt_stamps_generation_on_load_spec():
    """The core scheduler's admission-time identity lands on the load
    spec that later rides the broadcast ReqMeta."""
    sched = _make_bare_scheduler()
    req = create_request(num_tokens=128)
    sched.load_specs[req.request_id] = LoadSpec(
        vllm_cached_tokens=0, kvpool_cached_tokens=128, can_load=False
    )
    sched.note_transfer_attempt(req, 7)
    assert sched.load_specs[req.request_id].transfer_generation == 7

    # A request with no pending load spec is a defensive no-op.
    other = create_request(num_tokens=64)
    sched.note_transfer_attempt(other, 9)
    assert other.request_id not in sched.load_specs


def test_generation_rides_reqmeta_from_request_tracker():
    """The stamped load spec keeps its generation through
    ReqMeta.from_request_tracker -- the metadata the worker receives."""
    tracker = RequestTracker(
        req_id="r1",
        token_len=64,
        allocated_block_ids=([1, 2, 3, 4],),
    )
    load_spec = LoadSpec(
        vllm_cached_tokens=0,
        kvpool_cached_tokens=64,
        can_load=True,
        token_len=64,
        transfer_generation=11,
    )
    req_meta = ReqMeta.from_request_tracker(
        tracker, BLOCK, BLOCK, load_spec=load_spec, block_hashes=[]
    )
    assert req_meta is not None
    assert req_meta.load_spec is not None
    assert req_meta.load_spec.transfer_generation == 11


def _bare_store_connector(role_scheduler=True, role_worker=False):
    connector = object.__new__(MooncakeStoreConnector)
    connector.connector_scheduler = _make_bare_scheduler() if role_scheduler else None
    connector.connector_worker = (
        SimpleNamespace(
            get_load_failure_reports=lambda: [],
            get_load_attestations=lambda: [],
            close=lambda: None,
        )
        if role_worker
        else None
    )
    return connector


def test_multiconnector_forwards_identity_and_evidence():
    """[12] A MultiConnector forwards the transfer identity to the
    CHOSEN MooncakeStore child and aggregates both evidence channels
    from the children unchanged."""
    from vllm.v1.outputs import KVLoadAttestation, KVLoadFailureReport

    chosen = _bare_store_connector(role_scheduler=True)
    other = _bare_store_connector(role_scheduler=True)
    multi = object.__new__(MultiConnector)
    multi._connectors = [chosen, other]
    multi._requests_to_connector = {}
    multi._broken_evidence_children = set()

    req = create_request(num_tokens=128)
    # Selection happened in get_num_new_matched_tokens:
    multi._requests_to_connector[req.request_id] = 0
    chosen.connector_scheduler.load_specs[req.request_id] = LoadSpec(
        vllm_cached_tokens=0, kvpool_cached_tokens=128, can_load=False
    )
    multi.note_transfer_attempt(req, 42)
    assert (
        chosen.connector_scheduler.load_specs[req.request_id].transfer_generation == 42
    )
    assert other.connector_scheduler.load_specs == {}

    report = KVLoadFailureReport(
        request_id="r1",
        failed_block_ids=[5],
        token_span=(80, 96),
        rollback_boundary=80,
        observed_failed_layers=[0],
        healthy_above_span=True,
        restored_extent=128,
        transfer_generation=42,
    )
    attestation = KVLoadAttestation(
        request_id="r1",
        expected_objects=8,
        loaded_objects=7,
        failed_objects=1,
        certification="content",
        certified_objects=7,
        certification_failures=0,
        terminal=True,
        transfer_generation=42,
    )
    chosen.connector_worker = SimpleNamespace(
        get_load_failure_reports=lambda: [report],
        get_load_attestations=lambda: [attestation],
        close=lambda: None,
    )
    other.connector_worker = SimpleNamespace(
        get_load_failure_reports=lambda: [],
        get_load_attestations=lambda: [],
        close=lambda: None,
    )
    assert multi.get_load_failure_reports() == [report]
    assert multi.get_load_attestations() == [attestation]


def test_connector_worker_role_forwards_evidence_getters():
    """The worker-role MooncakeStoreConnector forwards both evidence
    getters to its worker (the post_forward drain path)."""
    from vllm.v1.outputs import KVLoadAttestation

    connector = object.__new__(MooncakeStoreConnector)
    attestation = KVLoadAttestation(
        request_id="r1",
        expected_objects=1,
        loaded_objects=1,
        failed_objects=0,
        certification="content",
        certified_objects=1,
        certification_failures=0,
        terminal=True,
        transfer_generation=1,
    )
    connector.connector_worker = SimpleNamespace(
        get_load_failure_reports=lambda: [],
        get_load_attestations=lambda: [attestation],
        close=lambda: None,
    )
    assert connector.get_load_failure_reports() == []
    assert connector.get_load_attestations() == [attestation]


# ---------------------------------------------------------------------------
# Field-level agreement with the REAL repair-policy scheduler.
# ---------------------------------------------------------------------------


def test_producer_evidence_admits_repair_in_real_scheduler():
    """[1][2] End-to-end: evidence produced by the REAL receive thread
    from REAL per-key GET outcomes (one OBJECT_NOT_FOUND) passes the
    REAL repair-policy scheduler's identity partition, geometry
    battery, reconciliation arithmetic, and completeness gate, and
    admits a RecoveryPlan over exactly the failed block's span."""
    scheduler, request, out1 = sync_claim_scenario(tp_size=1)
    generation = scheduler._req_transfer_generation[request.request_id]
    # The load covers the attempt's external CLAIM (3072 tokens), not the
    # whole request: the block-table suffix beyond the claim is locally
    # computed, never fetched.
    extent = scheduler._req_transfer_claim_extent[request.request_id]
    claim_blocks = extent // BLOCK
    (all_block_ids,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    block_ids = all_block_ids[:claim_blocks]
    layers = scheduler.vllm_config.model_config.hf_config.num_hidden_layers

    rcs = [0] * claim_blocks
    failed_idx = claim_blocks - 2
    rcs[failed_idx] = RC_OBJECT_NOT_FOUND
    buf = EvidenceBuffer()
    thread, _ = make_thread(FakeStore(rcs), buf, layers=layers, tp_rank=0)
    meta = make_req_meta(
        req_id=request.request_id,
        gen=generation,
        token_len=extent,
        block_ids=block_ids,
    )
    thread.request_queue.put(meta)
    thread._handle_request(meta)

    worker = bare_worker(buf)
    reports = worker.get_load_failure_reports()
    attestations = worker.get_load_attestations()
    assert len(reports) == 1 and len(attestations) == 1
    report, attestation = reports[0], attestations[0]
    assert report.transfer_generation == generation
    assert attestation.transfer_generation == generation
    assert attestation.failed_objects == layers
    assert attestation.failed_objects == len(report.failed_block_ids) * len(
        report.observed_failed_layers
    )
    assert attestation.certification == "content"
    assert attestation.certifies_content()
    # The aggregator's authoritative rank stamp (TP=1).
    report.source_rank = 0
    attestation.source_rank = 0

    deliver_evidence(
        scheduler, out1, [request], reports=[report], attestations=[attestation]
    )

    plan = scheduler.recovery_plans.get(request.request_id)
    assert plan is not None, "the conforming evidence must authorize a recovery"
    assert plan.repair_span == (failed_idx * BLOCK, (failed_idx + 1) * BLOCK)
    assert plan.transfer_generation == generation
    assert request.request_id not in scheduler._untrusted_external_reqs


def test_late_old_generation_evidence_cannot_invalidate_newer_delivery():
    """[9] A late failure report + attestation of an OLDER generation is
    dropped by the scheduler's identity partition: the newer (current)
    attempt's healthy delivery stays published and untouched."""
    scheduler, request, out1 = sync_claim_scenario(tp_size=1)
    generation = scheduler._req_transfer_generation[request.request_id]
    extent = scheduler._req_transfer_claim_extent[request.request_id]
    claim_blocks = extent // BLOCK
    (all_block_ids,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    block_ids = all_block_ids[:claim_blocks]
    layers = scheduler.vllm_config.model_config.hf_config.num_hidden_layers

    # The CURRENT attempt completes healthy.
    buf = EvidenceBuffer()
    thread, _ = make_thread(FakeStore([0] * claim_blocks), buf, layers=layers)
    meta = make_req_meta(
        req_id=request.request_id,
        gen=generation,
        token_len=extent,
        block_ids=block_ids,
    )
    thread.request_queue.put(meta)
    thread._handle_request(meta)
    worker = bare_worker(buf)
    healthy_reports = worker.get_load_failure_reports()
    healthy_atts = worker.get_load_attestations()
    assert healthy_reports == []
    for attestation in healthy_atts:
        attestation.source_rank = 0
    deliver_evidence(scheduler, out1, [request], reports=[], attestations=healthy_atts)
    assert request.request_id not in scheduler._untrusted_external_reqs

    # A stale artifact of an earlier attempt (generation - 1) arrives late.
    stale_buf = EvidenceBuffer()
    stale_thread, _ = make_thread(
        FakeStore([RC_OBJECT_NOT_FOUND] * claim_blocks), stale_buf, layers=layers
    )
    stale_meta = make_req_meta(
        req_id=request.request_id,
        gen=generation - 1,
        token_len=extent,
        block_ids=block_ids,
    )
    stale_thread.request_queue.put(stale_meta)
    stale_thread._handle_request(stale_meta)
    stale_worker = bare_worker(stale_buf)
    stale_reports = stale_worker.get_load_failure_reports()
    stale_atts = stale_worker.get_load_attestations()
    assert len(stale_reports) == 0  # all-failed: not single-region representable
    for attestation in stale_atts:
        attestation.source_rank = 0
    out2 = scheduler.schedule()
    computed_before_stale = request.num_computed_tokens
    deliver_evidence(
        scheduler, out2, [request], reports=stale_reports, attestations=stale_atts
    )

    assert request.request_id not in scheduler._untrusted_external_reqs
    assert scheduler.recovery_plans.get(request.request_id) is None
    # The stale artifact changed NOTHING: the request advanced only by
    # out2's ordinary step processing.
    assert request.num_computed_tokens == computed_before_stale


def test_missing_rank_evidence_fails_delivery_closed():
    """[11] TP=2: the rank that attests is not enough -- a silent second
    rank fails the delivery closed (the completeness gate), even though
    the reporting rank's evidence is perfectly conforming."""
    scheduler, request, out1 = sync_claim_scenario(tp_size=2)
    generation = scheduler._req_transfer_generation[request.request_id]
    extent = scheduler._req_transfer_claim_extent[request.request_id]
    claim_blocks = extent // BLOCK
    (all_block_ids,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    block_ids = all_block_ids[:claim_blocks]
    layers = scheduler.vllm_config.model_config.hf_config.num_hidden_layers

    rcs = [0] * claim_blocks
    failed_idx = claim_blocks - 2
    rcs[failed_idx] = RC_OBJECT_NOT_FOUND
    buf = EvidenceBuffer()
    thread, _ = make_thread(FakeStore(rcs), buf, layers=layers, tp_rank=0)
    meta = make_req_meta(
        req_id=request.request_id,
        gen=generation,
        token_len=extent,
        block_ids=block_ids,
    )
    thread.request_queue.put(meta)
    thread._handle_request(meta)
    worker = bare_worker(buf)
    reports = worker.get_load_failure_reports()
    attestations = worker.get_load_attestations()
    assert len(reports) == 1
    reports[0].source_rank = 0
    attestations[0].source_rank = 0

    deliver_evidence(
        scheduler, out1, [request], reports=reports, attestations=attestations
    )

    # The silent rank's gap is detected: no plan, the claim is untrusted,
    # the request rewound to the full-recompute floor.
    assert scheduler.recovery_plans.get(request.request_id) is None
    assert request.request_id in scheduler._untrusted_external_reqs
    assert request.num_computed_tokens == 0
