# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP verdict consensus under genuine multi-process execution (H5).

_ranks_repair_agree() is the repair transaction's commit gate: every
rank packs its verdict summary (outcome flags AND the work's identity:
plan_id, epoch, span) into one int64 tensor and all-gathers it; the
transaction commits only when every summary is identical and
successful. Before the hardening series this function had NO dedicated
coverage -- single-rank tests short-circuited it (ws <= 1 returned True
unconditionally) and the multi-rank behavior was only ever exercised by
the NPU end-to-end experiment.

These tests run the REAL collective over gloo in spawned processes
(world sizes 2 and 3), with:

- a scenario matrix covering agreement, every divergence class, and
  repeated sequential consensus calls;
- the cross-rank consistency requirement (every rank must derive the
  SAME boolean -- a split verdict would be its own bug class);
- a bounded-hang proof: a rank that never enters the collective makes
  the peers fail at the process-group timeout (an exception the
  execution envelope converts to a failed verdict), never an indefinite
  block. Parent-side watchdogs kill stragglers so the suite itself can
  never hang.
- single-process semantics: ws == 1 reduces to the local summary being
  a success (uniform with the multi-rank predicate).

NPU note: the production code pins the summary tensor to NPU when
torch.npu.is_available(); this CI matrix runs gloo/CPU, so the workers
force the CPU pinning path (guarded by hasattr so the suite itself has
no torch_npu dependency -- the final audit's FA-3). The NPU/HCCL
transport has its own device validation (see the audit's
device_validation/). The NPU/hccn transport keeps its own
end-to-end validation (the RFC evidence base); a dedicated NPU
regression for the consensus belongs with the device CI.
"""

import contextlib
import time
from datetime import timedelta

import pytest
import torch

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# Generous: each worker imports vllm (10-20 s) before its first
# collective; the watchdog exists so the suite can never hang, not to
# measure performance.
_WATCHDOG_S = 300.0
# The process-group timeout for the bounded-hang scenario: peers of a
# rank that never enters the collective fail after this, not never.
_PG_TIMEOUT_S = 20


def _green_args():
    return dict(
        aborted=False,
        executed_layers=4,
        pre_write_ok=True,
        auth_ownership_ok=True,
        auth_table_ok=True,
        slot_ownership=True,
        write_coverage=True,
        layer_end=4,
        backend_ok=True,
        plan_id=7,
        epoch=12,
        span=(128, 256),
    )


def _variant(**overrides):
    args = _green_args()
    args.update(overrides)
    return args


def _consensus_matrix(world_size):
    """(name, per-rank args, expected consensus) triples. The per-rank
    function must be called by EVERY rank for every scenario, in order --
    the collective sequence itself is part of the contract."""

    def rank0_only(**over):
        def pick(rank):
            return _variant(**over) if rank == 0 else _green_args()

        return pick

    def last_rank_only(**over):
        def pick(rank):
            return _variant(**over) if rank == world_size - 1 else _green_args()

        return pick

    return [
        ("all_ranks_agree_success", lambda r: _green_args(), True),
        (
            "all_ranks_agree_abort",
            lambda r: _variant(aborted=True, executed_layers=2),
            False,
        ),
        ("one_rank_aborts", rank0_only(aborted=True, executed_layers=2), False),
        ("last_rank_backend_fails", last_rank_only(backend_ok=False), False),
        (
            "odd_rank_pre_write_refused",
            lambda r: (
                _variant(pre_write_ok=False, executed_layers=0)
                if r % 2 == 1
                else _green_args()
            ),
            False,
        ),
        ("divergent_plan_id", rank0_only(plan_id=99), False),
        ("divergent_epoch", rank0_only(epoch=99), False),
        ("divergent_span", rank0_only(span=(0, 128)), False),
        (
            "divergent_executed_layers",
            rank0_only(executed_layers=3),
            False,
        ),
        # Repeated sequential consensus calls reuse the collective
        # cleanly: the matrix itself is a sequence, this pins recovery
        # to green after a divergent scenario.
        ("success_again_after_divergence", lambda r: _green_args(), True),
        ("abort_again_after_success", lambda r: _variant(aborted=True), False),
    ]


def _consensus_worker(rank, world_size, port, result_queue):
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_PG_TIMEOUT_S),
    )
    # Force the CPU summary path for the gloo matrix (see module
    # docstring) and restore it afterwards.
    _had_npu = hasattr(torch, "npu")
    if _had_npu:
        _npu_available = torch.npu.is_available
        torch.npu.is_available = lambda: False
    try:
        from vllm.v1.worker.gpu.repair_executor import (
            _ranks_repair_agree,
        )

        outcomes = []
        for name, args_for, expected in _consensus_matrix(world_size):
            got = _ranks_repair_agree(**args_for(rank))
            outcomes.append((name, got, expected))
        result_queue.put((rank, None, outcomes))
    except BaseException as exc:  # noqa: BLE001 - recorded for the parent
        result_queue.put((rank, f"{type(exc).__name__}: {exc}", None))
    finally:
        if _had_npu:
            torch.npu.is_available = _npu_available
        dist.destroy_process_group()


def _skip_worker(rank, world_size, port, result_queue):
    """The bounded-hang scenario: rank 0 never enters the first
    collective. The peers must fail at the process-group timeout with an
    exception (which the execution envelope converts to a failed
    verdict) -- never block indefinitely."""
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_PG_TIMEOUT_S),
    )
    _had_npu = hasattr(torch, "npu")
    if _had_npu:
        _npu_available = torch.npu.is_available
        torch.npu.is_available = lambda: False
    try:
        from vllm.v1.worker.gpu.repair_executor import (
            _ranks_repair_agree,
        )

        if rank == 0:
            # Never enters the collective: the peers' all_gather must
            # time out. Record the skip and finish (the process group
            # teardown also does not unblock them in gloo).
            result_queue.put((rank, None, "skipped"))
            return
        try:
            _ranks_repair_agree(**_green_args())
            result_queue.put((rank, None, "returned-without-consensus"))
        except BaseException as exc:  # noqa: BLE001
            result_queue.put((rank, None, f"raised:{type(exc).__name__}"))
    finally:
        if _had_npu:
            torch.npu.is_available = _npu_available
        with contextlib.suppress(Exception):
            dist.destroy_process_group()


def _run_distributed(worker_fn, world_size, extra=()):
    import multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    port = get_open_port()
    procs = [
        ctx.Process(
            target=worker_fn, args=(rank, world_size, port, result_queue, *extra)
        )
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    deadline = time.monotonic() + _WATCHDOG_S
    results: list[tuple[int, str | None, object]] = []
    while len(results) < world_size and time.monotonic() < deadline:
        with contextlib.suppress(Exception):
            results.append(result_queue.get(timeout=1.0))
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=10)
    hung = [i for i, p in enumerate(procs) if p.is_alive()]
    assert not hung, f"consensus workers hung (watchdog kill): ranks {hung}"
    assert len(results) == world_size, (
        f"only {len(results)}/{world_size} workers reported: {results}"
    )
    return results


@pytest.mark.parametrize("world_size", [2, 3])
def test_tp_consensus_matrix(world_size):
    """The full consensus matrix over real gloo collectives: agreement
    commits, every divergence class (plan identity, epoch, span,
    executed layers, per-rank failures) aborts on EVERY rank, sequential
    calls are independent, and all ranks derive the identical boolean."""
    results = _run_distributed(_consensus_worker, world_size)

    by_rank = {rank: (err, outcomes) for rank, err, outcomes in results}
    for rank, err, _ in results:
        assert err is None, f"rank {rank} raised inside the matrix: {err}"

    matrix = _consensus_matrix(world_size)
    per_scenario: dict[str, set[bool]] = {name: set() for name, _, _ in matrix}
    for rank in range(world_size):
        outcomes = by_rank[rank][1]
        assert outcomes is not None
        assert len(outcomes) == len(matrix)
        for (name, got, expected), (mname, _, _mexpected) in zip(outcomes, matrix):
            assert name == mname
            assert got == expected, (
                f"ws={world_size} rank={rank} scenario={name}: "
                f"consensus {got}, expected {expected}"
            )
            per_scenario[name].add(got)
    # Cross-rank consistency: every rank derived the same verdict for
    # every scenario (a split verdict would be its own bug class).
    for name, verdicts in per_scenario.items():
        assert len(verdicts) == 1, f"split verdict across ranks: {name}"


def test_tp_consensus_missing_rank_times_out():
    """The bounded-hang proof: a rank that never enters the collective
    makes its peers FAIL at the process-group timeout (an exception the
    envelope converts to a failed verdict) instead of blocking
    forever."""
    started = time.monotonic()
    results = _run_distributed(_skip_worker, 2)
    elapsed = time.monotonic() - started

    by_rank = {rank: note for rank, err, note in results}
    for rank, err, _ in results:
        assert err is None, f"rank {rank} raised unexpectedly: {err}"
    assert by_rank[0] == "skipped"
    assert by_rank[1].startswith("raised:"), (
        f"peer did not fail at the pg timeout: {by_rank[1]}"
    )
    # Bounded, never a hang: the peer fails either at the process-group
    # timeout or when the missing rank's teardown aborts the collective
    # -- both are exceptions the execution envelope converts to a failed
    # verdict. (No lower bound on the elapsed time: under load, the
    # teardown abort can precede the timeout; the safety property is
    # "raises, within the watchdog", not "waits exactly the timeout".)
    assert elapsed <= _WATCHDOG_S, f"took too long: {elapsed:.1f}s"


class _RaisingWorks(dict):
    """A works mapping whose getitem raises for one key (a rank-local
    corruption of the broadcast structure that the step-level set
    asserts cannot see)."""

    def __init__(self, base, raise_key):
        super().__init__(base)
        self._raise_key = raise_key

    def __getitem__(self, key):
        if key == self._raise_key:
            raise KeyError(key)
        return super().__getitem__(key)


def _parity_worker(rank, world_size, port, result_queue, mode):
    """Run execute_repair_only over two works under a rank-local
    corruption (mode 'lookup': the works mapping raises on one key;
    mode 'duplicate': the batch carries one request id twice) and report
    the number of consensus collectives issued plus the verdicts. The
    collective CALL COUNT must match the peers' on every path -- an
    unpaired collective leaves a peer inside all_gather until the
    process-group timeout."""
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    import torch
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_PG_TIMEOUT_S),
    )
    try:
        from types import SimpleNamespace

        from tests.v1.kv_connector.unit.utils import create_vllm_config
        from vllm.forward_context import set_forward_context
        from vllm.v1.attention.backends.cpu_attn import CPUAttentionMetadata
        from vllm.v1.core.sched.output import (
            CachedRequestData,
            RepairWork,
            SchedulerOutput,
        )
        from vllm.v1.worker.gpu import repair_executor

        calls = []
        real_agree = repair_executor._ranks_repair_agree

        def _counting(*args, **kwargs):
            calls.append(1)
            return real_agree(*args, **kwargs)

        repair_executor._ranks_repair_agree = _counting

        def _work(req_id, span, plan_id):
            # Empty authorization: the fence refuses before any layer
            # runs, so the parity being measured is the collective
            # sequence, not kernel behavior.
            return RepairWork(
                request_id=req_id,
                repair_span=span,
                layer_end=1,
                restored_extent=16,
                plan_id=plan_id,
                epoch=3,
                expected_write_block_ids=(),
            )

        works = {
            "req-a": _work("req-a", (0, 8), 1),
            "req-b": _work("req-b", (8, 16), 2),
        }
        broadcast_works = (
            _RaisingWorks(works, "req-a") if (mode == "lookup" and rank == 0) else works
        )
        batch = ["req-a", "req-b"]
        if mode == "duplicate" and rank == 0:
            batch = ["req-a", "req-a", "req-b"]
        sub = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={"req-a": 8, "req-b": 8},
            total_num_scheduled_tokens=16,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            repair_works=broadcast_works,
        )
        positions = torch.arange(16, dtype=torch.int64)
        model = SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(
                            attn=SimpleNamespace(
                                kv_cache=torch.zeros(2, 8, 8),
                                layer_name="model.layers.0",
                            )
                        )
                    )
                ],
                embed_input_ids=lambda ids: torch.zeros(ids.shape[0], 8),
            )
        )
        meta = CPUAttentionMetadata(
            num_actual_tokens=16,
            max_query_len=16,
            query_start_loc=torch.arange(3, dtype=torch.int32),
            max_seq_len=16,
            seq_lens=torch.full((2,), 16, dtype=torch.int32),
            block_table=torch.zeros(2, 1, dtype=torch.int32),
            slot_mapping=torch.arange(16, dtype=torch.int64),
            scheduler_metadata=None,
        )
        with set_forward_context(
            {"model.layers.0": meta},
            create_vllm_config(model="facebook/opt-125m"),
        ):
            results = repair_executor.execute_repair_only(
                model, sub, positions.clone(), positions, batch
            )
        verdicts = [(r.request_id, r.aborted, r.structural_check) for r in results]
        result_queue.put((rank, None, (len(calls), verdicts)))
    except BaseException as exc:  # noqa: BLE001 - recorded for the parent
        result_queue.put((rank, f"{type(exc).__name__}: {exc}", None))
    finally:
        with contextlib.suppress(Exception):
            dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3])
def test_work_lookup_failure_keeps_collective_parity(world_size):
    """A rank whose works lookup fails mid-step must still issue ONE
    consensus collective per scheduled work (the placeholder-identity
    call): the per-work iteration count is identical across ranks, so
    skipping one leaves a peer inside all_gather until the backend
    timeout instead of failing promptly."""
    results = _run_distributed(_parity_worker, world_size, extra=("lookup",))

    counts = {}
    for rank, err, payload in results:
        assert err is None, f"rank {rank} raised: {err}"
        n_calls, verdicts = payload
        counts[rank] = n_calls
        assert len(verdicts) == 2
        assert all(aborted and not sc for _rid, aborted, sc in verdicts), verdicts
    assert len(set(counts.values())) == 1, f"collective count diverged: {counts}"


@pytest.mark.parametrize("world_size", [2, 3])
def test_duplicate_batch_id_takes_step_error_path(world_size):
    """A duplicated request id in the worker-local batch must trip the
    step-level validation (the count-preserving all-failed path), not
    slip through the set-equality asserts and issue one collective per
    duplicate -- an unpaired trailing collective strands a peer until
    the backend timeout."""
    results = _run_distributed(_parity_worker, world_size, extra=("duplicate",))

    counts = {}
    for rank, err, payload in results:
        assert err is None, f"rank {rank} raised: {err}"
        n_calls, verdicts = payload
        counts[rank] = n_calls
        assert len(verdicts) == 2
        assert all(aborted and not sc for _rid, aborted, sc in verdicts), verdicts
    assert len(set(counts.values())) == 1, f"collective count diverged: {counts}"


def test_single_rank_consensus_semantics():
    """Ws == 1 reduces to the local summary being a success -- the same
    predicate the multi-rank loop derives from the gathered summaries
    (the pre-hardening code returned True unconditionally here, so a
    single-rank fence refusal or backend failure never failed
    consensus)."""
    from vllm.v1.worker.gpu.repair_executor import (
        _ranks_repair_agree,
    )

    assert _ranks_repair_agree(**_green_args())
    assert not _ranks_repair_agree(**_variant(aborted=True, executed_layers=2))
    assert not _ranks_repair_agree(**_variant(backend_ok=False))
    assert not _ranks_repair_agree(**_variant(pre_write_ok=False))
    assert not _ranks_repair_agree(**_variant(executed_layers=3))


# ---- The PRE-EXECUTION preflight consensus (A3: TP safety before native
# ---- layer execution) ----


def _preflight_setup(rank, world_size, refuse_rank=None, corrupt=None):
    """Build the repair step whose ranks diverge: `refuse_rank`'s fence
    refuses (a foreign slot mapping on that rank only), `corrupt` bends
    one rank's work identity. The layers are REAL collectives: a
    divergent-rank entry into the layer stream would deadlock the peers
    inside all_gather."""
    from types import SimpleNamespace

    from vllm.v1.core.sched.output import (
        CachedRequestData,
        RepairWork,
        SchedulerOutput,
    )

    span = (0, 8)
    auth = (5,)
    plan_id, epoch = 1, 3
    corrupt_rank = world_size - 1
    if rank == corrupt_rank:
        # ONE rank only carries the divergent identity; its geometry
        # stays self-consistent so its OWN fence passes -- only the
        # all-gathered identity comparison can catch it.
        if corrupt == "plan_id":
            plan_id = 99
        elif corrupt == "epoch":
            epoch = 99
        elif corrupt == "span":
            span = (8, 16)
        elif corrupt == "auth":
            auth = (6,)
    work = RepairWork(
        request_id="req-a",
        repair_span=span,
        layer_end=1,
        restored_extent=16,
        plan_id=plan_id,
        epoch=epoch,
        expected_write_block_ids=auth,
    )
    sub = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req-a": 8},
        total_num_scheduled_tokens=8,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        repair_works={"req-a": work},
    )
    positions = torch.arange(8, dtype=torch.int64)
    # The honest geometry: block 5, slots 40..47. A refusing rank points
    # its slot window at a foreign block (6).
    foreign = rank == refuse_rank
    slots = (
        torch.arange(6 * 8, 6 * 8 + 8, dtype=torch.int64)
        if foreign
        else torch.arange(5 * 8, 5 * 8 + 8, dtype=torch.int64)
    )
    table_row = torch.tensor([6]) if foreign else torch.tensor([5])

    import torch.distributed as dist

    class _CollectiveLayer:
        """A layer whose forward enters a REAL collective: if any rank
        enters the layer stream while a peer refuses, the peer must
        either have finished its verdict collective (paired) or the
        layer's collective blocks until the pg timeout."""

        def __init__(self):
            self.self_attn = SimpleNamespace(
                attn=SimpleNamespace(
                    kv_cache=torch.zeros(16, 8, 8),
                    layer_name="model.layers.0",
                )
            )

        def __call__(self, positions, hidden_states, residual):
            t = torch.ones(1)
            dist.all_reduce(t)
            return hidden_states + 1.0, residual

    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[_CollectiveLayer()],
            embed_input_ids=lambda ids: torch.zeros(ids.shape[0], 8),
        )
    )
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionMetadata

    meta = CPUAttentionMetadata(
        num_actual_tokens=8,
        max_query_len=8,
        query_start_loc=torch.arange(2, dtype=torch.int32),
        max_seq_len=8,
        seq_lens=torch.full((1,), 8, dtype=torch.int32),
        block_table=table_row.view(1, 1),
        slot_mapping=slots,
        scheduler_metadata=None,
    )
    return sub, positions, model, meta


def _preflight_worker(rank, world_size, port, result_queue, refuse_rank, corrupt):
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_PG_TIMEOUT_S),
    )
    try:
        from tests.v1.kv_connector.unit.utils import create_vllm_config
        from vllm.forward_context import set_forward_context
        from vllm.v1.worker.gpu import repair_executor

        sub, positions, model, meta = _preflight_setup(
            rank, world_size, refuse_rank=refuse_rank, corrupt=corrupt
        )
        layer_calls = []
        real_layer = model.model.layers[0]

        class _CountingLayer:
            """Delegates the cache views to the real layer and counts
            calls, so the preflight's geometry derivation sees a layer
            object (not a bare function)."""

            self_attn = real_layer.self_attn

            def __call__(self, positions, hidden_states, residual):
                layer_calls.append(rank)
                return real_layer(positions, hidden_states, residual)

        model.model.layers[0] = _CountingLayer()
        with set_forward_context(
            {"model.layers.0": meta},
            create_vllm_config(model="facebook/opt-125m"),
        ):
            results = repair_executor.execute_repair_only(
                model, sub, positions.clone(), positions, ["req-a"]
            )
        verdicts = [
            (
                r.request_id,
                r.aborted,
                r.structural_check,
                r.executed_layers,
                (r.commit_check or {}).get("preflight_agree"),
                (r.commit_check or {}).get("capability_ok"),
            )
            for r in results
        ]
        result_queue.put((rank, None, (len(layer_calls), verdicts)))
    except BaseException as exc:  # noqa: BLE001 - recorded for the parent
        result_queue.put((rank, f"{type(exc).__name__}: {exc}", None))
    finally:
        with contextlib.suppress(Exception):
            dist.destroy_process_group()


def _assert_preflight_abort(results, world_size, expect_calls):
    """Every rank aborted with zero executed layers, the preflight
    refused, and the layer stream ran exactly `expect_calls` times per
    rank (0 everywhere on divergence, 1 everywhere on consensus)."""
    for rank, err, payload in results:
        assert err is None, f"rank {rank} raised: {err}"
        n_calls, verdicts = payload
        assert n_calls == expect_calls, (
            f"rank {rank} entered the layer stream {n_calls} times "
            f"(expected {expect_calls}): a rank-local refusal must abort "
            "the layer execution on EVERY rank"
        )
        assert len(verdicts) == 1
        (_rid, aborted, sc, layers, pf_agree, _cap) = verdicts[0]
        assert aborted, f"rank {rank} must abort on preflight divergence"
        assert not sc
        assert layers == 0
        assert pf_agree is False, f"rank {rank} preflight_agree must be False"


@pytest.mark.parametrize("world_size", [2, 3])
@pytest.mark.parametrize("refuse_rank", [0, 1])
def test_preflight_fence_refusal_aborts_every_rank_before_layers(
    world_size, refuse_rank
):
    """THE A3 defect: a rank whose pre-write fence refuses (a foreign
    slot mapping on that rank only) must abort the step on EVERY rank
    BEFORE any rank enters native layer execution. Pre-fix, the fence
    was rank-local and the consensus post-execution: the refusing rank
    skipped the layer collectives its passing peers entered, and the
    passing peers blocked inside the layer all-reduce until the
    process-group timeout (a TP hang)."""
    if refuse_rank >= world_size:
        pytest.skip("rank beyond world size")
    results = _run_distributed(_preflight_worker, world_size, extra=(refuse_rank, None))
    _assert_preflight_abort(results, world_size, expect_calls=0)


@pytest.mark.parametrize("world_size", [2, 3])
@pytest.mark.parametrize("corrupt", ["plan_id", "epoch", "span", "auth"])
def test_preflight_identity_divergence_aborts_every_rank(world_size, corrupt):
    """A rank executing a divergent work identity (plan_id, epoch, span,
    or authorized write geometry) must abort the step on every rank
    before any native execution -- the all-gathered operation identity,
    not a boolean, is what consents."""
    results = _run_distributed(_preflight_worker, world_size, extra=(None, corrupt))
    _assert_preflight_abort(results, world_size, expect_calls=0)


@pytest.mark.parametrize("world_size", [2, 3])
def test_preflight_consensus_allows_consented_execution(world_size):
    """The positive control: with every rank's preflight green, all
    ranks enter the layer stream together (the collective layer runs
    once per rank), the verdict commits, and preflight_agree is True
    on every rank."""
    results = _run_distributed(_preflight_worker, world_size, extra=(None, None))
    for rank, err, payload in results:
        assert err is None, f"rank {rank} raised: {err}"
        n_calls, verdicts = payload
        assert n_calls == 1, f"rank {rank} layer calls: {n_calls}"
        (_rid, aborted, sc, layers, pf_agree, cap_ok) = verdicts[0]
        assert not aborted
        assert sc, f"rank {rank} verdict must commit: {verdicts}"
        assert layers == 1
        assert pf_agree is True
        assert cap_ok is True
