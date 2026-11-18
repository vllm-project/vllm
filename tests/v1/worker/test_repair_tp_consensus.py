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


def _run_distributed(worker_fn, world_size):
    import multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    port = get_open_port()
    procs = [
        ctx.Process(target=worker_fn, args=(rank, world_size, port, result_queue))
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
