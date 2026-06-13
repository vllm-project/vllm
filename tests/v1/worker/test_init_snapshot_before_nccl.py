# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for VLLM_INIT_SNAPSHOT_BEFORE_NCCL.

Verifies that when the env flag is enabled, the worker takes its
MemorySnapshot BEFORE init_worker_distributed_environment (NCCL init) runs.
This is the opposite of the default upstream ordering (which is asserted by
tests/v1/worker/test_worker_memory_snapshot.py); both tests should pass
because the flag defaults to off.

Motivation: on consumer GPUs with TP+PP, the per-rank NCCL workspace is
asymmetric (the last PP stage / heaviest rank can carry ~8.8 GiB on a 4x24G
TP=2 PP=2 setup, vs. ~1-2 GiB on rank 0). Taking the snapshot pre-NCCL keeps
gpu_memory_utilization computed against truly-free VRAM so the heaviest rank
does not OOM at init.
"""

import multiprocessing as mp
import os
import tempfile
from multiprocessing.queues import Queue
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import CacheConfig, set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment

_QUEUE: Queue | None = None


def _track(name: str, rank: int):
    if _QUEUE is not None:
        _QUEUE.put((name, rank))


def _wrap(name: str, original):
    def wrapper(*args, **kwargs):
        rank = int(os.environ.get("RANK", "-1"))
        _track(name, rank)
        return original(*args, **kwargs)

    return wrapper


def _worker_process(
    rank: int,
    world_size: int,
    distributed_init_method: str,
    queue: Queue,
    error_queue: Queue,
):
    global _QUEUE
    _QUEUE = queue

    try:
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        # Opt into the pre-NCCL snapshot path for THIS test only.
        os.environ["VLLM_INIT_SNAPSHOT_BEFORE_NCCL"] = "1"

        vllm_config = EngineArgs(
            model="facebook/opt-125m",
            tensor_parallel_size=2,
            load_format="dummy",
        ).create_engine_config()

        worker = Worker(
            vllm_config=vllm_config,
            local_rank=rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
        )

        original_init_worker = init_worker_distributed_environment
        original_memory_snapshot_init = MemorySnapshot.__init__

        init_patch = patch(
            "vllm.v1.worker.gpu_worker.init_worker_distributed_environment",
            side_effect=_wrap("init_distributed", original_init_worker),
        )
        memory_patch = patch.object(
            MemorySnapshot,
            "__init__",
            _wrap("memory_snapshot", original_memory_snapshot_init),
        )

        with (
            init_patch,
            memory_patch,
            set_current_vllm_config(vllm_config),
        ):
            worker.init_device()

        # Sanity: the worker should have stashed the pre-NCCL free-bytes
        # reading on itself when the env var is on.
        assert worker._startup_free_bytes is not None, (
            "Worker._startup_free_bytes was not populated when "
            "VLLM_INIT_SNAPSHOT_BEFORE_NCCL=1"
        )
        assert worker._startup_free_bytes > 0

        assert worker._startup_snapshot is not None, (
            "Worker._startup_snapshot was not populated when "
            "VLLM_INIT_SNAPSHOT_BEFORE_NCCL=1"
        )

        # OBJECT-IDENTITY check (stronger than value equality): prove the
        # pre-NCCL snapshot OBJECT is the one ACTUALLY USED for budgeting
        # downstream, not just taken-and-discarded with its value copied
        # into `_startup_free_bytes`.
        #
        # A regression that takes a fresh post-NCCL `MemorySnapshot` and
        # assigns it to `self.init_snapshot` while leaving
        # `_startup_free_bytes` populated from the earlier pre-NCCL read
        # would PASS a value-only check (`init_snapshot.free_memory ==
        # _startup_free_bytes`) on a quiet GPU where NCCL workspace happens
        # to be a no-op, but FAIL this `is` check. That's the actual bug
        # we're guarding against.
        assert worker.init_snapshot is worker._startup_snapshot, (
            f"Rank {rank}: worker.init_snapshot must BE "
            "(object identity) the pre-NCCL snapshot stashed on "
            "worker._startup_snapshot — proves the pre-NCCL snapshot is "
            "REUSED for budgeting, not just taken and discarded."
        )
        # Belt-and-braces value check too — sanity safeguard against
        # someone mutating MemorySnapshot.free_memory between the two reads.
        assert worker.init_snapshot.free_memory == worker._startup_free_bytes, (
            f"Rank {rank}: worker.init_snapshot.free_memory "
            f"({worker.init_snapshot.free_memory}) must equal "
            f"worker._startup_free_bytes ({worker._startup_free_bytes})."
        )

        queue.put(("success", rank))

    except Exception as e:  # pragma: no cover - surfaced via error_queue
        error_queue.put((rank, str(e), type(e).__name__))
        raise


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs for tensor parallelism",
)
def test_memory_snapshot_is_taken_before_nccl_init_when_env_enabled():
    """When VLLM_INIT_SNAPSHOT_BEFORE_NCCL=1, the first MemorySnapshot()
    call must precede init_worker_distributed_environment()."""
    world_size = 2

    with tempfile.NamedTemporaryFile(delete=False) as f:
        distributed_init_method = f"file://{f.name}"

    ctx = mp.get_context("spawn")
    operation_queue = ctx.Queue()
    error_queue = ctx.Queue()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker_process,
            args=(
                rank,
                world_size,
                distributed_init_method,
                operation_queue,
                error_queue,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=60)

    errors = []
    while not error_queue.empty():
        rank, error_msg, error_type = error_queue.get()
        errors.append(f"Rank {rank}: {error_type}: {error_msg}")
    if errors:
        pytest.fail("Worker processes failed:\n" + "\n".join(errors))

    operations = []
    while not operation_queue.empty():
        operations.append(operation_queue.get())

    for rank in range(world_size):
        rank_ops = [op for op, r in operations if r == rank]
        # First snapshot must appear before init_distributed.
        first_snapshot = rank_ops.index("memory_snapshot")
        init_distributed = rank_ops.index("init_distributed")
        assert first_snapshot < init_distributed, (
            f"Rank {rank}: with VLLM_INIT_SNAPSHOT_BEFORE_NCCL=1, the first "
            f"memory_snapshot (index {first_snapshot}) must happen BEFORE "
            f"init_distributed (index {init_distributed}). "
            f"Ops: {rank_ops}"
        )

    os.unlink(distributed_init_method.replace("file://", ""))


# ---------------------------------------------------------------------------
# No-GPU unit tests for request_memory(): exercises the OOM-guard arithmetic
# directly with a mocked MemorySnapshot. These cover the actual failure mode
# the env flag is meant to fix — asymmetric per-rank NCCL footprint causing
# the heaviest PP-stage rank to fail the `free_memory < requested_memory`
# check at startup — without needing a multi-GPU host. They run on CI.
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from vllm.v1.worker.utils import request_memory

_GiB = 1024**3


@dataclass
class _FakeSnapshot:
    """Minimal stand-in for MemorySnapshot — request_memory only reads
    free_memory, total_memory, and device_."""

    free_memory: int
    total_memory: int
    device_: str = "cuda:2"  # PP-terminal proxy on a 4-GPU TP=2 PP=2 box


@pytest.mark.parametrize(
    "pre_nccl_free,post_nccl_free,total,gmu,post_nccl_should_oom",
    [
        # Case A: PP-terminal rank on consumer 24G (e.g. 3090/4090).
        # NCCL workspace ~9 GiB lands on the heaviest rank → post-NCCL
        # free drops to ~14.9 GiB. With GMU=0.70 the ask is ~16.8 GiB,
        # which exceeds post-NCCL free → ValueError pre-fix.
        # Pre-NCCL free is ~23 GiB → request_memory passes.
        (
            int(0.96 * 24 * _GiB),
            int(0.62 * 24 * _GiB),
            24 * _GiB,
            0.70,
            True,
        ),
        # Case B: light rank (rank 0, TP=2 PP=2). NCCL ~2 GiB only,
        # post-NCCL free still ~22 GiB. Ask = 16.8 GiB → passes either
        # way. Demonstrates the rank-asymmetry: same fleet config, only
        # the PP-terminal rank tips into ValueError pre-fix.
        (
            int(0.96 * 24 * _GiB),
            int(0.92 * 24 * _GiB),
            24 * _GiB,
            0.70,
            False,
        ),
        # Case C: H100/A100 (80 GiB), GMU=0.85. Even with NCCL ~5 GiB,
        # post-NCCL has ample headroom; the issue does not manifest on
        # datacenter hardware. Confirms the fix is a no-op where it
        # should be a no-op.
        (
            int(0.98 * 80 * _GiB),
            int(0.92 * 80 * _GiB),
            80 * _GiB,
            0.85,
            False,
        ),
    ],
    ids=["consumer_24G_PP_terminal", "consumer_24G_light_rank", "datacenter_80G"],
)
def test_request_memory_pre_vs_post_nccl_snapshot(
    pre_nccl_free: int,
    post_nccl_free: int,
    total: int,
    gmu: float,
    post_nccl_should_oom: bool,
):
    """Verify request_memory()'s OOM guard against the actual failure mode
    the fix targets.

    With the fix (pre-NCCL snapshot): request_memory MUST never raise — the
    pre-NCCL free reading is large enough to cover the GMU ask on every
    parametrized case.

    Without the fix (post-NCCL snapshot): request_memory MAY raise
    ValueError on the PP-terminal rank where NCCL ate enough VRAM to push
    free_memory below the GMU ask. This is exactly the asymmetric-OOM the
    env flag exists to fix.
    """
    cache_config = MagicMock(spec=CacheConfig)
    cache_config.gpu_memory_utilization = gmu

    pre = _FakeSnapshot(free_memory=pre_nccl_free, total_memory=total)
    post = _FakeSnapshot(free_memory=post_nccl_free, total_memory=total)

    # WITH FIX: pre-NCCL snapshot reaches request_memory → never raises.
    requested = request_memory(pre, cache_config)
    assert requested > 0
    assert requested <= pre.free_memory

    # WITHOUT FIX: post-NCCL snapshot reaches request_memory → raises iff
    # the post-NCCL free dropped below the GMU ask.
    if post_nccl_should_oom:
        with pytest.raises(ValueError, match="Free memory on device"):
            request_memory(post, cache_config)
    else:
        # On the light rank / datacenter cases, post-NCCL free is still
        # comfortably above the ask — the fix is correctly a no-op here.
        request_memory(post, cache_config)


def test_request_memory_oom_message_mentions_device():
    """Sanity: the ValueError surfaces the device identifier so users on
    multi-GPU boxes can tell which rank tripped the guard."""
    cache_config = MagicMock(spec=CacheConfig)
    cache_config.gpu_memory_utilization = 0.95

    snapshot = _FakeSnapshot(
        free_memory=1 * _GiB,
        total_memory=24 * _GiB,
        device_="cuda:3",
    )
    with pytest.raises(ValueError) as exc_info:
        request_memory(snapshot, cache_config)
    assert "cuda:3" in str(exc_info.value)


# ---------------------------------------------------------------------------
# MED-3 mutation guard: the bug we're fixing is that
# `Worker.init_snapshot` was being set to a POST-NCCL MemorySnapshot even
# when VLLM_INIT_SNAPSHOT_BEFORE_NCCL=1, silently undoing the budget fix.
# The GPU-required test above asserts the object-identity invariant on a
# real worker. This no-GPU guard re-asserts the invariant at the SOURCE
# level: a regression that removes the reuse line ("self.init_snapshot
# = init_snapshot = pre_nccl_snapshot" or equivalent) inside
# `Worker.init_device` would land green on CPU-only CI without this guard,
# because CPU CI cannot run the multi-process spawn test that catches it.
#
# This is intentional belt-and-braces. It reads the file textually
# rather than via AST so the assertion failure messages point operators at
# the exact line in gpu_worker.py to inspect.
# ---------------------------------------------------------------------------
import inspect


def test_worker_init_device_source_reuses_pre_nccl_snapshot():
    """Mutation guard: confirm gpu_worker.py's `init_device` keeps the
    reuse-property that wires the pre-NCCL snapshot into `self.init_snapshot`
    when VLLM_INIT_SNAPSHOT_BEFORE_NCCL=1. A regression that removes it
    would only be caught by the GPU spawn test above; this guard catches
    it on CPU-only CI.

    Round-N+3: uses AST-based property check instead of textual regex so
    that a semantically-equivalent rewrite (e.g. splitting the chained
    assignment into two statements, or introducing a temporary) still
    passes — the property is "self.init_snapshot is assigned a value that
    references pre_nccl_snapshot in init_device", not "this exact line
    of source exists verbatim".
    """
    import ast
    import textwrap

    src = inspect.getsource(Worker.init_device)

    # The fix is: when snapshot_before_nccl is true, self.init_snapshot
    # MUST be assigned from pre_nccl_snapshot, not from a fresh
    # MemorySnapshot(device=...) call.
    assert "pre_nccl_snapshot" in src, (
        "Worker.init_device no longer references `pre_nccl_snapshot` — "
        "the VLLM_INIT_SNAPSHOT_BEFORE_NCCL fix has been removed or "
        "renamed. Update this guard or restore the fix."
    )

    tree = ast.parse(textwrap.dedent(src))

    # Property A: some Assign node whose targets include `self.init_snapshot`
    # has `pre_nccl_snapshot` referenced anywhere in its RHS expression
    # (handles chained `self.init_snapshot = init_snapshot = pre_nccl_snapshot`
    # AND a two-statement rewrite like
    #   init_snapshot = pre_nccl_snapshot
    #   self.init_snapshot = init_snapshot
    # via transitive lookup below).
    init_snapshot_rhs_sources: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and t.attr == "init_snapshot"
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    init_snapshot_rhs_sources.append(ast.unparse(node.value))

    assert init_snapshot_rhs_sources, (
        "Worker.init_device contains no assignment to `self.init_snapshot`."
    )

    # Direct property: at least one RHS references pre_nccl_snapshot.
    direct_match = any("pre_nccl_snapshot" in rhs for rhs in init_snapshot_rhs_sources)

    # Transitive property: tolerate `self.init_snapshot = <local>` where
    # <local> was earlier bound from pre_nccl_snapshot.
    transitive_match = False
    if not direct_match:
        # Find all locals bound from pre_nccl_snapshot.
        locals_from_pre: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and "pre_nccl_snapshot" in ast.unparse(
                node.value
            ):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        locals_from_pre.add(t.id)
        for rhs in init_snapshot_rhs_sources:
            if any(name in rhs for name in locals_from_pre):
                transitive_match = True
                break

    assert direct_match or transitive_match, (
        "Worker.init_device no longer reuses the pre-NCCL snapshot for "
        "`self.init_snapshot`. Expected an assignment chain that wires "
        "`self.init_snapshot` to `pre_nccl_snapshot` (directly or via a "
        "local) inside the `if snapshot_before_nccl:` branch. Without it, "
        "the fix regresses silently on multi-rank consumer-GPU TP+PP setups. "
        f"Observed RHS expressions for `self.init_snapshot`: "
        f"{init_snapshot_rhs_sources}"
    )

    # The OBJECT reference stash for the GPU identity test must also
    # remain in place — defends MED-3's `is`-check from regression.
    # Property check via AST: some Assign to self._startup_snapshot whose
    # RHS references pre_nccl_snapshot.
    startup_snapshot_rhs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and t.attr == "_startup_snapshot"
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    startup_snapshot_rhs.append(ast.unparse(node.value))

    assert any("pre_nccl_snapshot" in rhs for rhs in startup_snapshot_rhs), (
        "Worker.init_device no longer stashes `self._startup_snapshot = "
        "pre_nccl_snapshot`. The MED-3 GPU test asserts "
        "`worker.init_snapshot is worker._startup_snapshot`; without "
        "this stash, that test would fail to construct. Observed RHS "
        f"expressions for `self._startup_snapshot`: {startup_snapshot_rhs}"
    )


def test_worker_init_device_no_post_reset_of_startup_snapshot():
    """Round-N+3 guard: a regression that adds
    `self._startup_snapshot = None` (or any other reset) LATER in
    `init_device` — e.g. inside an exception handler or a cleanup block —
    would pass the reuse-guard above but break the MED-3 GPU identity
    assertion `worker.init_snapshot is worker._startup_snapshot` (since the
    field would have been reset between assignment and end-of-init_device).

    Enforce: every assignment to `self._startup_snapshot` inside
    `init_device` must (a) be from `pre_nccl_snapshot`, and (b) never
    assign None. The acceptable count is 0 (env-off path: never touched
    by init_device proper) or 1 (the pre-NCCL stash). More than 1 is a
    smell that almost certainly indicates a reset path.
    """
    import ast
    import textwrap

    src = inspect.getsource(Worker.init_device)
    tree = ast.parse(textwrap.dedent(src))

    assignments: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and t.attr == "_startup_snapshot"
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    assignments.append((node.lineno, ast.unparse(node.value)))
        elif isinstance(node, ast.AnnAssign):
            # Catch `self._startup_snapshot: MemorySnapshot | None = None`-style
            # initialisations too.
            t = node.target
            if (
                isinstance(t, ast.Attribute)
                and t.attr == "_startup_snapshot"
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
                and node.value is not None
            ):
                assignments.append((node.lineno, ast.unparse(node.value)))

    if not assignments:
        # Env-off path doesn't touch _startup_snapshot here — fine.
        return

    # The current code has a typed declaration `self._startup_snapshot:
    # MemorySnapshot | None = None` AND a real `self._startup_snapshot =
    # pre_nccl_snapshot` inside the env-on branch. We allow the typed-None
    # init only if it appears BEFORE the pre_nccl_snapshot stash (lexically
    # earlier line number), because that's the harmless "declare default"
    # idiom. We forbid any None-assignment that comes AFTER a
    # pre_nccl_snapshot-assignment, because that would be a reset.
    pre_nccl_lines = [
        ln for ln, rhs in assignments if "pre_nccl_snapshot" in rhs
    ]
    assert pre_nccl_lines, (
        "Worker.init_device has assignments to self._startup_snapshot but "
        "none of them are from pre_nccl_snapshot. The MED-3 invariant "
        f"requires the stash. Observed: {assignments}"
    )
    last_pre_nccl_line = max(pre_nccl_lines)

    for lineno, rhs in assignments:
        if lineno <= last_pre_nccl_line:
            continue
        # Any assignment AFTER the pre_nccl stash is a regression — even if
        # it re-assigns from pre_nccl_snapshot, it shouldn't happen twice.
        raise AssertionError(
            f"Worker.init_device has an assignment to self._startup_snapshot "
            f"at line {lineno} (RHS: {rhs!r}) AFTER the pre_nccl_snapshot "
            f"stash at line {last_pre_nccl_line}. This would reset the "
            f"field that the MED-3 GPU identity test "
            f"(`worker.init_snapshot is worker._startup_snapshot`) depends "
            f"on. All assignments: {assignments}"
        )

    # And: no assignment may set None as the rhs, except the typed-default
    # declaration which we allow if it appears BEFORE the pre_nccl stash.
    for lineno, rhs in assignments:
        if rhs.strip() == "None" and lineno > min(
            ln for ln, _ in assignments
        ):
            raise AssertionError(
                f"Worker.init_device assigns self._startup_snapshot = None "
                f"at line {lineno}, which would defeat the MED-3 GPU "
                f"identity invariant. All assignments: {assignments}"
            )


# ---------------------------------------------------------------------------
# Round-N+5 guard (MED finding from round-N+4 adversarial verifier):
# The existing reuse-property check above asserts that *some* assignment to
# `self.init_snapshot` references `pre_nccl_snapshot`. But it does not verify
# that EVERY assignment to `self.init_snapshot` is GATED by the
# `if snapshot_before_nccl:` block. A regression that introduces an
# unconditional `self.init_snapshot = MemorySnapshot(...)` AFTER the if/else
# block would silently override BOTH paths — the env-on path would have its
# pre-NCCL stash trampled by a fresh post-NCCL snapshot, breaking the fix —
# and slip past the existing AST checks because they never inspect nodes
# outside the if-body.
#
# This guard widens the AST inspection: it collects ALL `self.init_snapshot`
# assignments anywhere in `init_device`, classifies each by whether it lives
# inside the `if snapshot_before_nccl:` if/else block (gated) or anywhere
# else (ungated), and asserts ungated count == 0.
# ---------------------------------------------------------------------------


def _find_enclosing_snapshot_before_nccl_if(func_ast, target_node):
    """Walk the AST of `func_ast` searching for an `if snapshot_before_nccl:`
    node that contains `target_node` in its body or its else-body
    (transitively through nested if/with/try statements).

    Returns the enclosing `if snapshot_before_nccl:` `ast.If` node if
    `target_node` is inside it, else None.
    """
    import ast as _ast

    def _is_snapshot_gate_if(node) -> bool:
        if not isinstance(node, _ast.If):
            return False
        # Recognise both `if snapshot_before_nccl:` and
        # `if not snapshot_before_nccl:` (we treat `not` as the same gate
        # — its else-branch is the env-on path).
        test = node.test
        if isinstance(test, _ast.Name) and test.id == "snapshot_before_nccl":
            return True
        if (
            isinstance(test, _ast.UnaryOp)
            and isinstance(test.op, _ast.Not)
            and isinstance(test.operand, _ast.Name)
            and test.operand.id == "snapshot_before_nccl"
        ):
            return True
        return False

    def _contains(container_body, needle) -> bool:
        for stmt in container_body:
            for child in _ast.walk(stmt):
                if child is needle:
                    return True
        return False

    # Find every `if snapshot_before_nccl:` node in func_ast and check if
    # target_node lives inside its body or orelse.
    for node in _ast.walk(func_ast):
        if _is_snapshot_gate_if(node):
            if _contains(node.body, target_node) or _contains(
                node.orelse, target_node
            ):
                return node
    return None


def _classify_init_snapshot_assigns(func_ast):
    """Find every assignment to `self.init_snapshot` in `func_ast` and
    classify each by whether it's enclosed in the `if snapshot_before_nccl:`
    if/else block.

    Returns a list of (assign_node, enclosing_if_or_None) tuples.
    """
    import ast as _ast

    classified: list[tuple[_ast.AST, _ast.If | None]] = []
    for node in _ast.walk(func_ast):
        if isinstance(node, _ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, _ast.Attribute)
                    and target.attr == "init_snapshot"
                    and isinstance(target.value, _ast.Name)
                    and target.value.id == "self"
                ):
                    enclosing = _find_enclosing_snapshot_before_nccl_if(
                        func_ast, node
                    )
                    classified.append((node, enclosing))
                    break  # Don't double-count chained targets in same Assign.
        elif isinstance(node, _ast.AnnAssign):
            t = node.target
            if (
                isinstance(t, _ast.Attribute)
                and t.attr == "init_snapshot"
                and isinstance(t.value, _ast.Name)
                and t.value.id == "self"
                and node.value is not None
            ):
                enclosing = _find_enclosing_snapshot_before_nccl_if(
                    func_ast, node
                )
                classified.append((node, enclosing))
    return classified


def test_worker_init_device_no_ungated_init_snapshot_assigns():
    """Regression guard: every `self.init_snapshot = ...` MUST be inside
    the `if snapshot_before_nccl:` if/else block. An unconditional assign
    outside this gate would override both paths and break the env-on
    snapshot capture semantics — the fresh post-NCCL MemorySnapshot would
    trample the pre-NCCL stash even when VLLM_INIT_SNAPSHOT_BEFORE_NCCL=1.

    Round-N+5 adversarial finding: the prior reuse-property guard checks
    that SOME assignment references `pre_nccl_snapshot`, but does not check
    that NO LATER assignment overrides it. This guard closes that gap.
    """
    import ast
    import textwrap

    src = inspect.getsource(Worker.init_device)
    tree = ast.parse(textwrap.dedent(src))
    # The parsed tree's first body element is the function def; for
    # _classify... we want to walk that function's interior.
    fn_node = tree.body[0]
    classified = _classify_init_snapshot_assigns(fn_node)

    assert classified, (
        "Worker.init_device contains no assignment to `self.init_snapshot` "
        "at all — either the fix has been removed or this guard is searching "
        "the wrong function."
    )

    ungated = [n for n, gate in classified if gate is None]
    assert len(ungated) == 0, (
        f"Found {len(ungated)} ungated self.init_snapshot assignment(s) "
        f"in Worker.init_device — these would bypass the snapshot_before_nccl "
        f"gate and break the env-on path. Lines: "
        f"{[getattr(n, 'lineno', '?') for n in ungated]}"
    )


def test_classify_helper_catches_ungated_mutation():
    """Negative-control / self-test: confirm the AST guard above ACTUALLY
    detects an ungated assign if one were introduced. Without this, a
    silently-broken classifier would let real regressions through while
    the positive test still passes.

    Mutates the AST in-memory by appending an unconditional
    `self.init_snapshot = MemorySnapshot()` to the end of `init_device`'s
    body, runs the classifier on the mutated tree, and asserts the new
    assign shows up as ungated.
    """
    import ast
    import textwrap

    src = inspect.getsource(Worker.init_device)
    tree = ast.parse(textwrap.dedent(src))
    fn_node = tree.body[0]

    # Baseline: capture how many ungated assigns exist BEFORE injection.
    before = _classify_init_snapshot_assigns(fn_node)
    before_ungated = sum(1 for _, gate in before if gate is None)

    # Inject `self.init_snapshot = MemorySnapshot()` at the END of the
    # function body — outside any `if snapshot_before_nccl:` block.
    injected = ast.parse("self.init_snapshot = MemorySnapshot()").body[0]
    fn_node.body.append(injected)

    after = _classify_init_snapshot_assigns(fn_node)
    after_ungated = sum(1 for _, gate in after if gate is None)

    assert after_ungated >= before_ungated + 1, (
        f"Negative-control failure: injecting an ungated "
        f"`self.init_snapshot = MemorySnapshot()` at the end of "
        f"`init_device` did NOT increase the ungated-assign count "
        f"(before={before_ungated}, after={after_ungated}). The classifier "
        f"or the enclosing-if helper has a bug; real regressions could "
        f"slip past `test_worker_init_device_no_ungated_init_snapshot_assigns`."
    )


def test_worker_init_device_emits_startup_free_bytes_gauge():
    """MED-1 mutation guard: confirm the Prometheus startup_free_bytes
    gauge wiring stays in place. The actual scrape is verified at runtime
    via /metrics on a real engine; this is a source-level guard so that
    accidental removal of the gauge call is caught on CPU-only CI."""
    src = inspect.getsource(Worker.init_device)
    assert "_emit_startup_free_bytes_gauge" in src, (
        "Worker.init_device no longer calls "
        "`_emit_startup_free_bytes_gauge`. MED-1 observability for "
        "VLLM_INIT_SNAPSHOT_BEFORE_NCCL=1 has regressed."
    )


def test_worker_init_device_documents_lazy_nccl_assumption():
    """MED-4 mutation guard: confirm gpu_worker.py documents the lazy-NCCL
    allocation assumption near the snapshot ordering. Keeps future
    maintainers from re-introducing the assumption-vs-reality drift the
    adversarial verifier flagged."""
    src = inspect.getsource(Worker.init_device)
    # Accept any of these wordings — the precise prose may evolve.
    assert any(
        token in src
        for token in (
            "lazily",
            "lazy",
            "non_torch_increase",
        )
    ), (
        "Worker.init_device no longer documents the NCCL lazy-allocation "
        "assumption. Restore the comment near `pre_nccl_snapshot` so "
        "future maintainers don't re-debate whether late NCCL allocation "
        "breaks the budget."
    )


# ---------------------------------------------------------------------------
# MED-2 unit test (no-GPU): exercises the new ValueError path in
# `determine_available_memory` directly, with all GPU-touching components
# mocked. This is critical coverage — round-N+2 verifier flagged the MED-2
# diagnostic ("RAISE --gpu-memory-utilization") as entirely uncovered code.
#
# The test calls Worker.determine_available_memory as an unbound method on
# a MagicMock(spec=Worker), patching the GPU-bound helpers
# (`memory_profiling`, `torch.accelerator.memory_stats`, `current_platform`)
# so the method's arithmetic runs end-to-end on CPU.
# ---------------------------------------------------------------------------


@pytest.fixture
def _med2_mocked_worker():
    """Build a mock Worker whose `determine_available_memory` will reach the
    MED-2 budget-too-small branch when invoked. Returns (worker, params)
    where params lets tests tweak per-case values."""
    from vllm.utils.mem_utils import MemoryProfilingResult, MemorySnapshot

    worker = MagicMock(spec=Worker)
    # Critical: cache_config must be real-ish — gpu_memory_utilization is
    # interpolated into the error message and used for the "suggested" calc.
    worker.cache_config = MagicMock(spec=CacheConfig)
    worker.cache_config.gpu_memory_utilization = 0.70
    worker.cache_config.kv_cache_memory_bytes = 0  # bypass early-return path

    # init_snapshot reflects pre-NCCL free on a 24 GiB consumer GPU.
    worker.init_snapshot = _FakeSnapshot(
        free_memory=int(0.96 * 24 * _GiB),
        total_memory=24 * _GiB,
        device_="cuda:2",
    )
    # requested_memory = GMU * pre-NCCL free  ≈  0.70 * 23.04 GiB  ≈  16.13 GiB
    # We deliberately set non-KV components so their sum > requested_memory.
    worker.requested_memory = int(0.70 * 23.04 * _GiB)

    worker.device = "cuda:2"

    # model_runner.model_memory_usage is read twice (for memory_profiling and
    # in the diagnostic). Set to a value such that:
    #   model + non_torch + activations + cudagraph  >  requested_memory
    model_memory_usage = 10 * _GiB
    worker.model_runner = MagicMock()
    worker.model_runner.model_memory_usage = model_memory_usage
    worker.model_runner.profile_run = MagicMock()
    worker.model_runner.profile_cudagraph_memory = MagicMock(return_value=0)

    worker.vllm_config = MagicMock()
    # Disable cudagraph branch — keeps the test focused on the pure
    # NCCL-workspace OOM scenario. Setting cudagraph_mode to CUDAGraphMode.NONE
    # short-circuits the `cudagraph_mode != CUDAGraphMode.NONE` check.
    from vllm.config import CUDAGraphMode

    worker.vllm_config.compilation_config = MagicMock()
    worker.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    # Pre-build a profile_result with values that force MED-2.
    # NB: MemoryProfilingResult.__post_init__ OVERWRITES `before_profile`
    # and `after_profile`, so we have to assign their attributes AFTER
    # instantiation.
    before_create = MemorySnapshot.__new__(MemorySnapshot)
    before_create.device_ = "cuda:2"
    before_create.free_memory = int(0.96 * 24 * _GiB)
    before_create.total_memory = 24 * _GiB
    before_create.torch_peak = 0
    before_create.non_torch_memory = 0

    profile_result = MemoryProfilingResult(
        non_torch_increase=8 * _GiB,  # NCCL workspace inflation on PP-terminal
        weights_memory=model_memory_usage,
        before_create=before_create,
    )
    # After __post_init__ ran, override the synthesized profile snapshots.
    profile_result.before_profile.torch_peak = 0
    profile_result.before_profile.free_memory = int(0.96 * 24 * _GiB)
    profile_result.before_profile.total_memory = 24 * _GiB
    profile_result.after_profile.free_memory = int(0.5 * 24 * _GiB)
    profile_result.after_profile.total_memory = 24 * _GiB

    return worker, profile_result, model_memory_usage


def test_determine_available_memory_budget_too_small_raises(_med2_mocked_worker):
    """MED-2: when non_kv_cache_memory exceeds the GMU-derived budget,
    `determine_available_memory` must raise ValueError with an actionable
    message instructing the user to RAISE (not lower) gpu_memory_utilization.

    This exercises the MED-2 diagnostic that round-N+2 verifier reported as
    entirely uncovered. Failure mode without the diagnostic: downstream
    `_check_enough_kv_cache_memory` raises with a misleading message
    suggesting LOWERING gpu_memory_utilization, which is exactly the
    opposite of the actual remediation.
    """
    import contextlib

    worker, profile_result, _model_mem = _med2_mocked_worker

    # `torch.accelerator.memory_stats` returns the raw CUDA peak. Set so
    # `profile_torch_peak - before_profile.torch_peak` = 6 GiB activations.
    torch_peak = 6 * _GiB

    @contextlib.contextmanager
    def _fake_memory_profiling(init_snapshot, weights_memory):
        yield profile_result

    with (
        patch(
            "vllm.v1.worker.gpu_worker.memory_profiling",
            _fake_memory_profiling,
        ),
        patch(
            "vllm.v1.worker.gpu_worker.torch.accelerator.memory_stats",
            return_value={"allocated_bytes.all.peak": torch_peak},
        ),
    ):
        # Sanity: model + non_torch + activations  =  10 + 8 + 6 = 24 GiB,
        # which exceeds requested_memory (≈16.13 GiB) → MED-2 must fire.
        with pytest.raises(ValueError, match=r"RAISE --gpu-memory-utilization"):
            Worker.determine_available_memory(worker)


def test_determine_available_memory_budget_too_small_error_mentions_device(
    _med2_mocked_worker,
):
    """MED-2: ValueError should carry the device identifier so multi-GPU
    operators can tell which rank tripped the guard."""
    import contextlib

    worker, profile_result, _model_mem = _med2_mocked_worker

    @contextlib.contextmanager
    def _fake_memory_profiling(init_snapshot, weights_memory):
        yield profile_result

    with (
        patch(
            "vllm.v1.worker.gpu_worker.memory_profiling",
            _fake_memory_profiling,
        ),
        patch(
            "vllm.v1.worker.gpu_worker.torch.accelerator.memory_stats",
            return_value={"allocated_bytes.all.peak": 6 * _GiB},
        ),
    ):
        with pytest.raises(ValueError) as exc_info:
            Worker.determine_available_memory(worker)
    assert "cuda:2" in str(exc_info.value), (
        "MED-2 ValueError must include the device identifier to help "
        "operators on multi-GPU TP+PP setups identify the failing rank. "
        f"Got: {exc_info.value}"
    )


def test_determine_available_memory_budget_too_small_error_mentions_suggested_gmu(
    _med2_mocked_worker,
):
    """MED-2: ValueError should suggest a concrete GMU floor so the
    operator has an actionable next step rather than a generic 'tune it'."""
    import contextlib
    import re

    worker, profile_result, _model_mem = _med2_mocked_worker

    @contextlib.contextmanager
    def _fake_memory_profiling(init_snapshot, weights_memory):
        yield profile_result

    with (
        patch(
            "vllm.v1.worker.gpu_worker.memory_profiling",
            _fake_memory_profiling,
        ),
        patch(
            "vllm.v1.worker.gpu_worker.torch.accelerator.memory_stats",
            return_value={"allocated_bytes.all.peak": 6 * _GiB},
        ),
    ):
        with pytest.raises(ValueError) as exc_info:
            Worker.determine_available_memory(worker)
    msg = str(exc_info.value)
    # Either a "suggested: >= 0.XX" or similar — accept any decimal after
    # "suggested".
    assert re.search(r"suggested.*?\d+\.\d+", msg), (
        "MED-2 ValueError must surface a concrete suggested GMU floor "
        "(decimal). Got: " + msg
    )


def test_determine_available_memory_budget_too_small_error_warns_against_lowering_gmu(
    _med2_mocked_worker,
):
    """MED-2: the message must explicitly warn that LOWERING gpu_memory_utilization
    will make the failure WORSE — defends against the natural-but-wrong
    operator instinct of cutting GMU when seeing an OOM."""
    import contextlib

    worker, profile_result, _model_mem = _med2_mocked_worker

    @contextlib.contextmanager
    def _fake_memory_profiling(init_snapshot, weights_memory):
        yield profile_result

    with (
        patch(
            "vllm.v1.worker.gpu_worker.memory_profiling",
            _fake_memory_profiling,
        ),
        patch(
            "vllm.v1.worker.gpu_worker.torch.accelerator.memory_stats",
            return_value={"allocated_bytes.all.peak": 6 * _GiB},
        ),
    ):
        with pytest.raises(ValueError) as exc_info:
            Worker.determine_available_memory(worker)
    msg = str(exc_info.value).lower()
    # Catch both "make this worse" and "will make it worse" phrasings.
    assert "worse" in msg, (
        "MED-2 ValueError must explicitly warn that lowering "
        "gpu_memory_utilization makes the failure worse, not better. "
        f"Got: {exc_info.value}"
    )


# ---------------------------------------------------------------------------
# Round-N+4 — HIGH: gauge must emit even without PROMETHEUS_MULTIPROC_DIR
# (the standard single-API-server TP+PP deployment, which is this PR's
# target user). Round-N+2 adversarial flagged that the gauge was
# silently dead in that environment because `multiprocess_mode='mostrecent'`
# requires a multiproc dir.
# ---------------------------------------------------------------------------


def _reset_gauge_module_state():
    """Wipe the module-level gauge cache + prometheus_client default
    registry between scenarios so each scenario re-registers fresh.

    prometheus_client doesn't allow re-registering a collector with the
    same name, and mixing multiprocess and non-multiprocess gauges in the
    same registry would otherwise collide.
    """
    import vllm.v1.worker.gpu_worker as gpu_worker_mod
    from prometheus_client import REGISTRY

    gpu_worker_mod._STARTUP_FREE_BYTES_GAUGE = None
    gpu_worker_mod._STARTUP_TOTAL_BYTES_GAUGE = None

    # Drop any previously-registered vllm:startup_*_bytes collectors.
    for name in ("vllm:startup_free_bytes", "vllm:startup_total_bytes"):
        try:
            collector = REGISTRY._names_to_collectors.get(name)
        except Exception:
            collector = None
        if collector is not None:
            try:
                REGISTRY.unregister(collector)
            except Exception:
                pass


def test_startup_free_bytes_gauge_emits_value_without_multiproc_dir(
    tmp_path, monkeypatch
):
    """Emission smoke (round-N+4 / round-N+6): with
    PROMETHEUS_MULTIPROC_DIR unset (the standard single-API-server TP+PP
    deployment shape — exactly this PR's target user), the gauge wiring
    produces scrapeable values via the local registry.

    Note: this test alone does NOT differentiate the conditional-kwarg
    fix from a pre-amend always-pass-`mostrecent`-kwarg version, because
    on prometheus_client 0.25.0 the `multiprocess_mode='mostrecent'` kwarg
    is advisory and a Gauge with it will still emit fine via MutexValue
    when the multiproc dir is unset. The actual differentiator that pins
    the conditional behaviour is
    `test_startup_free_bytes_gauge_uses_mostrecent_only_when_multiproc_env_set`
    below — this test remains as defensive end-to-end emission coverage.
    """
    from prometheus_client import generate_latest

    from vllm.v1.worker.gpu_worker import _emit_startup_free_bytes_gauge

    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    _reset_gauge_module_state()

    _emit_startup_free_bytes_gauge(
        rank=0,
        free_bytes=12345,
        total_bytes=24 * _GiB,
        served_model_name="test-model",
    )

    output = generate_latest().decode("utf-8")
    assert 'vllm:startup_free_bytes{' in output, (
        "vllm:startup_free_bytes was not registered/exposed when "
        "PROMETHEUS_MULTIPROC_DIR is unset. The HIGH-priority fallback "
        f"to a non-multiprocess Gauge has regressed. Output:\n{output}"
    )
    # The label set must include rank + model_name and the value 12345.
    assert 'rank="0"' in output and 'model_name="test-model"' in output, (
        "vllm:startup_free_bytes labels missing or incorrect. "
        f"Output:\n{output}"
    )
    assert "12345" in output, (
        "vllm:startup_free_bytes value not emitted. "
        f"Output:\n{output}"
    )


def test_startup_free_bytes_gauge_emits_value_with_multiproc_dir(
    tmp_path, monkeypatch
):
    """Emission smoke (round-N+4 / round-N+6) — sibling test: with
    PROMETHEUS_MULTIPROC_DIR set (multi-API-server path), the gauge
    should still register successfully (multiprocess_mode='mostrecent'
    branch). We only assert successful registration here — multiprocess
    collection aggregates via a separate code path and is exercised by
    vLLM's existing /metrics integration tests.

    Like the sibling above, this is emission/registration smoke, not a
    pre-vs-post-amend differentiator (both pre-amend and post-amend pass
    `multiprocess_mode='mostrecent'` when the env var IS set — they only
    differ when it's UNSET). See
    `test_startup_free_bytes_gauge_uses_mostrecent_only_when_multiproc_env_set`
    for the actual differentiator."""
    from vllm.v1.worker.gpu_worker import (
        _emit_startup_free_bytes_gauge,
        _STARTUP_FREE_BYTES_GAUGE as _initial,
    )
    import vllm.v1.worker.gpu_worker as gpu_worker_mod

    multiproc_dir = tmp_path / "prom_multiproc"
    multiproc_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(multiproc_dir))
    _reset_gauge_module_state()

    _emit_startup_free_bytes_gauge(
        rank=2,
        free_bytes=99999,
        total_bytes=24 * _GiB,
        served_model_name="test-model-mp",
    )
    # The gauge must have been registered (module-level cache populated).
    assert gpu_worker_mod._STARTUP_FREE_BYTES_GAUGE is not None, (
        "vllm:startup_free_bytes failed to register when "
        "PROMETHEUS_MULTIPROC_DIR is set."
    )
    assert gpu_worker_mod._STARTUP_TOTAL_BYTES_GAUGE is not None, (
        "vllm:startup_total_bytes failed to register when "
        "PROMETHEUS_MULTIPROC_DIR is set."
    )
    # Cleanup — leave the registry tidy for any later test in the same
    # process. The temp multiproc dir is auto-cleaned by tmp_path.
    _reset_gauge_module_state()


# ---------------------------------------------------------------------------
# Round-N+6 — HIGH-priority differentiator: pin the actual conditional
# kwarg behaviour at registration time.
#
# The two emission tests above both PASS even on a hypothetical "always
# pass `multiprocess_mode='mostrecent'`" pre-amend version, because on
# prometheus_client 0.25.0 the kwarg is advisory when the multiproc dir
# is unset (the Gauge falls back to MutexValue and emits fine). So neither
# of those tests differentiate the conditional code from an unconditional
# one; this test does, by inspecting the registered Gauge's
# `_multiprocess_mode` attribute directly:
#   - env unset, post-amend (no kwarg passed)        → 'all' (default)
#   - env unset, pre-amend  (kwarg always passed)    → 'mostrecent'
# Asserting `'all'` on the env-unset path therefore CATCHES a regression
# back to "always pass the kwarg".
#
# Verified on prometheus_client 0.25.0 — see PR description for the
# probe output.
# ---------------------------------------------------------------------------


def test_startup_free_bytes_gauge_uses_mostrecent_only_when_multiproc_env_set(
    tmp_path, monkeypatch
):
    """Differentiator: the gauge registration must be conditional on
    PROMETHEUS_MULTIPROC_DIR.

    Pre-amend behaviour (always pass `multiprocess_mode='mostrecent'`):
        env unset → registered Gauge has `_multiprocess_mode == 'mostrecent'`
        env set   → registered Gauge has `_multiprocess_mode == 'mostrecent'`

    Post-amend behaviour (only pass the kwarg when env is set):
        env unset → registered Gauge has `_multiprocess_mode == 'all'`  (default)
        env set   → registered Gauge has `_multiprocess_mode == 'mostrecent'`

    The env-unset assertion (`'all'`) is what makes this a true
    differentiator — it FAILS pre-amend and PASSES post-amend.
    """
    import vllm.v1.worker.gpu_worker as gpu_worker_mod
    from vllm.v1.worker.gpu_worker import _emit_startup_free_bytes_gauge

    # --- Scenario 1: PROMETHEUS_MULTIPROC_DIR unset ---
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    _reset_gauge_module_state()

    _emit_startup_free_bytes_gauge(
        rank=0,
        free_bytes=12345,
        total_bytes=24 * _GiB,
        served_model_name="test-model",
    )

    free_gauge = gpu_worker_mod._STARTUP_FREE_BYTES_GAUGE
    total_gauge = gpu_worker_mod._STARTUP_TOTAL_BYTES_GAUGE
    assert free_gauge is not None, (
        "vllm:startup_free_bytes was not registered when "
        "PROMETHEUS_MULTIPROC_DIR is unset."
    )
    assert total_gauge is not None, (
        "vllm:startup_total_bytes was not registered when "
        "PROMETHEUS_MULTIPROC_DIR is unset."
    )
    # Default mode for a Gauge constructed with NO `multiprocess_mode`
    # kwarg on prometheus_client 0.25.0 is 'all'. A pre-amend regression
    # that always passes `multiprocess_mode='mostrecent'` would surface
    # here as 'mostrecent' instead.
    free_mode = getattr(free_gauge, "_multiprocess_mode", "MISSING")
    total_mode = getattr(total_gauge, "_multiprocess_mode", "MISSING")
    assert free_mode == "all", (
        f"vllm:startup_free_bytes was registered with "
        f"`_multiprocess_mode={free_mode!r}` when "
        f"PROMETHEUS_MULTIPROC_DIR is unset. The conditional gating in "
        f"`_emit_startup_free_bytes_gauge` has regressed — "
        f"`multiprocess_mode='mostrecent'` is being passed unconditionally. "
        f"Expected 'all' (the prometheus_client Gauge default when no "
        f"kwarg is supplied)."
    )
    assert total_mode == "all", (
        f"vllm:startup_total_bytes was registered with "
        f"`_multiprocess_mode={total_mode!r}` when "
        f"PROMETHEUS_MULTIPROC_DIR is unset. Expected 'all'."
    )

    # --- Scenario 2: PROMETHEUS_MULTIPROC_DIR set ---
    multiproc_dir = tmp_path / "prom_multiproc_diff"
    multiproc_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(multiproc_dir))
    _reset_gauge_module_state()

    _emit_startup_free_bytes_gauge(
        rank=1,
        free_bytes=54321,
        total_bytes=24 * _GiB,
        served_model_name="test-model-mp",
    )

    free_gauge = gpu_worker_mod._STARTUP_FREE_BYTES_GAUGE
    total_gauge = gpu_worker_mod._STARTUP_TOTAL_BYTES_GAUGE
    assert free_gauge is not None
    assert total_gauge is not None
    free_mode = getattr(free_gauge, "_multiprocess_mode", "MISSING")
    total_mode = getattr(total_gauge, "_multiprocess_mode", "MISSING")
    assert free_mode == "mostrecent", (
        f"vllm:startup_free_bytes was registered with "
        f"`_multiprocess_mode={free_mode!r}` when "
        f"PROMETHEUS_MULTIPROC_DIR IS set. Expected 'mostrecent'."
    )
    assert total_mode == "mostrecent", (
        f"vllm:startup_total_bytes was registered with "
        f"`_multiprocess_mode={total_mode!r}` when "
        f"PROMETHEUS_MULTIPROC_DIR IS set. Expected 'mostrecent'."
    )

    # Cleanup — leave the registry tidy for any later test in the same
    # process.
    _reset_gauge_module_state()


# ---------------------------------------------------------------------------
# Round-N+4 — MED-A: extend the AST mutation guard to verify the LAST
# write to `self.init_snapshot` inside the `if snapshot_before_nccl:`
# branch (in source order) reads `pre_nccl_snapshot`, so a regression
# that adds `self.init_snapshot = MemorySnapshot(device=self.device)`
# AFTER the pre-NCCL stash gets caught at source-level on CPU-only CI.
# ---------------------------------------------------------------------------


def test_worker_init_device_no_post_pattern_override_of_init_snapshot():
    """MED-A (round-N+4): a regression that does

        self.init_snapshot = init_snapshot = pre_nccl_snapshot
        ...
        self.init_snapshot = MemorySnapshot(device=self.device)  # OVERRIDE

    inside the `if snapshot_before_nccl:` branch would pass the existing
    "at least one assignment from pre_nccl_snapshot" guard but silently
    undo the budget fix at runtime. Catch it at the source level by
    enforcing that, in source order, the LAST assignment to
    `self.init_snapshot` reachable from the env-on branch comes from
    `pre_nccl_snapshot` (directly or transitively via a local).
    """
    import ast
    import textwrap

    src = inspect.getsource(Worker.init_device)
    tree = ast.parse(textwrap.dedent(src))

    # Locals bound from `pre_nccl_snapshot` anywhere in init_device.
    locals_from_pre: set[str] = {"pre_nccl_snapshot"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and "pre_nccl_snapshot" in ast.unparse(
            node.value
        ):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    locals_from_pre.add(t.id)

    # Walk every `If` whose test references `snapshot_before_nccl` (the
    # env-on branch that this fix lives in). Within each such body,
    # collect assignments to `self.init_snapshot` in source order.
    def _is_snapshot_before_nccl_test(test: ast.expr) -> bool:
        return "snapshot_before_nccl" in ast.unparse(test)

    init_snapshot_assigns_in_envon: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_snapshot_before_nccl_test(node.test):
            # Walk only the truthy `body` (skip `orelse` — that is the
            # default-off path which legitimately constructs a fresh
            # MemorySnapshot).
            for inner in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if isinstance(inner, ast.Assign):
                    for t in inner.targets:
                        if (
                            isinstance(t, ast.Attribute)
                            and t.attr == "init_snapshot"
                            and isinstance(t.value, ast.Name)
                            and t.value.id == "self"
                        ):
                            init_snapshot_assigns_in_envon.append(
                                (inner.lineno, ast.unparse(inner.value))
                            )

    assert init_snapshot_assigns_in_envon, (
        "Worker.init_device has no `self.init_snapshot = ...` assignment "
        "inside any `if snapshot_before_nccl:` block. The env-on path is "
        "broken — `self.init_snapshot` would never reflect the pre-NCCL "
        "snapshot."
    )

    # Sort by line number → last write in source order is the one that
    # actually wins at runtime within that branch.
    init_snapshot_assigns_in_envon.sort(key=lambda x: x[0])
    last_lineno, last_rhs = init_snapshot_assigns_in_envon[-1]

    transitively_from_pre = any(
        name in last_rhs for name in locals_from_pre
    )
    assert transitively_from_pre, (
        f"MED-A: the LAST assignment to `self.init_snapshot` inside the "
        f"`if snapshot_before_nccl:` branch (line {last_lineno}, "
        f"RHS: {last_rhs!r}) does NOT reference `pre_nccl_snapshot` "
        "(directly or via a local bound from it). A regression has "
        "added a post-pattern override that silently undoes the budget "
        "fix. All env-on-branch assignments in source order: "
        f"{init_snapshot_assigns_in_envon}. "
        f"Locals bound from pre_nccl_snapshot: {sorted(locals_from_pre)}."
    )


# ---------------------------------------------------------------------------
# Round-N+4 — MED-B: with the env flag OFF, the new `_startup_*` attrs
# must NOT be set on the Worker, so external code that uses
# `hasattr(worker, '_startup_free_bytes')` for feature detection
# continues to see the pre-PR behavior (False).
# ---------------------------------------------------------------------------


def test_worker_default_off_path_does_not_add_startup_attrs():
    """MED-B (round-N+4): on the default-off path
    (`VLLM_INIT_SNAPSHOT_BEFORE_NCCL` unset), the Worker must NOT acquire
    `_startup_free_bytes`/`_startup_total_bytes`/`_startup_snapshot` as
    instance attributes. This preserves `hasattr(worker, ...)`-style
    feature detection for any code that introspects Workers.

    Source-level check (CPU-only CI) — walks the AST and asserts that
    every assignment to those three attributes lives inside an
    `if snapshot_before_nccl:` block (or transitively below one).
    """
    import ast
    import textwrap

    src = inspect.getsource(Worker.init_device)
    tree = ast.parse(textwrap.dedent(src))

    # For each assignment / annotated assignment to self._startup_*, find
    # the nearest enclosing `if snapshot_before_nccl:` block.
    targets_to_check = (
        "_startup_free_bytes",
        "_startup_total_bytes",
        "_startup_snapshot",
    )

    # Collect: (target_attr, lineno, gated_by_snapshot_before_nccl: bool)
    findings: list[tuple[str, int, bool]] = []

    def _enclosed_by_snapshot_if(node: ast.AST, root: ast.AST) -> bool:
        """True iff `node` is lexically inside an `if snapshot_before_nccl:`
        body. We test by membership-walking the body of every such If
        we find."""
        for n in ast.walk(root):
            if isinstance(n, ast.If) and "snapshot_before_nccl" in ast.unparse(
                n.test
            ):
                for descendant in ast.walk(
                    ast.Module(body=n.body, type_ignores=[])
                ):
                    if descendant is node:
                        return True
        return False

    for node in ast.walk(tree):
        target_attr = None
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and t.attr in targets_to_check
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    target_attr = t.attr
                    break
        elif isinstance(node, ast.AnnAssign):
            t = node.target
            if (
                isinstance(t, ast.Attribute)
                and t.attr in targets_to_check
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ):
                target_attr = t.attr

        if target_attr is None:
            continue

        gated = _enclosed_by_snapshot_if(node, tree)
        findings.append((target_attr, node.lineno, gated))

    assert findings, (
        "MED-B: expected at least one assignment to one of "
        f"{targets_to_check} inside Worker.init_device. None found — has "
        "the env-on branch been removed entirely?"
    )

    ungated = [f for f in findings if not f[2]]
    assert not ungated, (
        f"MED-B: found {len(ungated)} assignment(s) to startup-tracking "
        "attribute(s) NOT gated behind `if snapshot_before_nccl:`. "
        f"Default-off path will leak these as instance attrs and break "
        f"`hasattr(...)`-based feature detection. Ungated: {ungated}. "
        f"All findings: {findings}."
    )


# ---------------------------------------------------------------------------
# Round-N+4 — MED-C: the original concurrent-GPU assertion
# (`init_snapshot.free_memory >= profile_result.after_profile.free_memory`)
# becomes trivially true on the env-on path (pre-NCCL free is much larger
# than post-profile free), so the guard silently disarms. The fix uses
# `profile_result.before_profile.free_memory` (post-NCCL) as the diagnostic
# baseline, preserving the guard on both code paths.
# ---------------------------------------------------------------------------


def test_concurrent_gpu_assertion_uses_post_nccl_baseline():
    """MED-C (round-N+4): the concurrent-GPU guard inside
    `determine_available_memory` must use a POST-NCCL baseline so it
    continues to catch real concurrent-process memory drift on the
    env-on path.

    Source-level check: assert (a) that
    `profile_result.before_profile.free_memory` is referenced AS the
    diagnostic baseline, and (b) the assertion no longer compares
    against `self.init_snapshot.free_memory` (which is pre-NCCL on the
    env-on path and trivially large).
    """
    src = inspect.getsource(Worker.determine_available_memory)

    # The old (silently-disarmed) form is the chained
    # `assert self.init_snapshot.free_memory >= free_gpu_memory`.
    # The new form must use `before_profile.free_memory`.
    assert "before_profile.free_memory" in src, (
        "MED-C: `determine_available_memory` no longer references "
        "`profile_result.before_profile.free_memory` — the post-NCCL "
        "baseline used to keep the concurrent-GPU assertion alive on "
        "the env-on path. The guard has regressed."
    )
    # And: the diagnostic assert must NOT be the literal
    # `self.init_snapshot.free_memory >= free_gpu_memory` form, which
    # is trivially true with a pre-NCCL init_snapshot.
    assert (
        "self.init_snapshot.free_memory >= free_gpu_memory" not in src
    ), (
        "MED-C: the concurrent-GPU assertion is still using "
        "`self.init_snapshot.free_memory` as the baseline. On the env-on "
        "path that's pre-NCCL → trivially true → silently disarmed. "
        "Switch to `profile_result.before_profile.free_memory`."
    )

