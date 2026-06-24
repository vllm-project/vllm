# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the pipeline-parallel cudagraph-mode consensus that
guards against the #45094 PP cudagraph-vs-eager split-brain wedge.

Root cause (vllm-project/vllm#45094): under pipeline parallelism each PP rank
runs ``CudagraphDispatcher.dispatch`` independently. Per-rank-local state
(calculate_kv_scales, cascade-attention detection, LoRA bookkeeping, a
partially-failed capture) can make one stage pick PIECEWISE (replay a graph with
baked-in inter-stage P2P send/recv shapes) while another stage picks NONE
(eager). The stages then disagree on the cross-stage P2P schedule, the
send/recv never rendezvous, and the engine wedges
(``shm_broadcast`` 60s timeout -> RPC timeout -> EngineCore crash).

``coordinate_cudagraph_mode_across_pp`` reaches consensus by taking the MIN of
the per-rank mode across the PP group (NONE=0 < PIECEWISE=1 < FULL=2), mirroring
the data-parallel consensus already in ``dp_utils._post_process_cudagraph_mode``.

These tests reproduce the divergence at the unit level (two simulated PP ranks)
and verify the consensus collapses them to a single agreed mode. The first test
fails against pre-fix code (no such function / no consensus -> ranks stay
divergent). A multi-GPU integration repro is described in the module docstring
of ``tests/v1/cudagraph/test_pp_cudagraph_consensus.py`` (this file) below.

Integration repro (requires >=2 GPUs, TP>=1 PP=2):

    vllm serve <model> -tp 1 -pp 2 \
        --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
    # then drive concurrent long-reasoning requests whose admitted batch-token
    # count crosses a captured cudagraph bucket boundary on some steps but not
    # others. Pre-fix: one PP stage replays the graph while the other runs
    # eager -> shm_broadcast "No available shared memory broadcast block found
    # in 60 seconds" -> RPC timeout -> EngineCore crash. Post-fix: both stages
    # agree (MIN) -> no wedge.
"""

from pathlib import Path
from unittest import mock

import pytest

from vllm.config import CUDAGraphMode


def _patch_pp_group(world_size: int, peer_modes: list[int] | None = None):
    """Return a mock get_pp_group() whose all_reduce(MIN) simulates peers.

    peer_modes: the cudagraph-mode ints contributed by the *other* PP ranks.
    The mocked all_reduce computes min(local, *peer_modes) in place, exactly as
    a real MIN all-reduce over the PP group would.
    """
    peer_modes = peer_modes or []

    mock_group = mock.MagicMock()
    mock_group.world_size = world_size
    mock_group.cpu_group = mock.MagicMock()

    def fake_all_reduce(tensor, op, group):
        # Emulate torch.distributed.all_reduce(MIN) over the PP cpu_group.
        local = int(tensor.item())
        tensor.fill_(min([local, *peer_modes]))

    return mock_group, fake_all_reduce


@pytest.mark.parametrize(
    "rank_a_mode, rank_b_mode, expected_consensus",
    [
        # The #45094 split-brain: one stage PIECEWISE (graph replay), the other
        # NONE (eager). MIN -> NONE so both run eager and the P2P schedule
        # stays consistent.
        (CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE, CUDAGraphMode.NONE),
        (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE),
        # FULL vs PIECEWISE -> PIECEWISE (still a divergence that would
        # mismatch graph capture across stages).
        (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE, CUDAGraphMode.PIECEWISE),
        # Already in agreement -> unchanged.
        (CUDAGraphMode.PIECEWISE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.PIECEWISE),
        (CUDAGraphMode.NONE, CUDAGraphMode.NONE, CUDAGraphMode.NONE),
    ],
)
def test_pp_consensus_collapses_divergent_modes(
    rank_a_mode, rank_b_mode, expected_consensus
):
    """Two PP ranks that dispatched divergent cudagraph modes must converge to
    the MIN after consensus — the guard against the #45094 split-brain wedge."""
    from vllm.v1.worker.pp_utils import coordinate_cudagraph_mode_across_pp

    # Rank A's view: its peer (rank B) contributed rank_b_mode.
    mock_group_a, fake_ar_a = _patch_pp_group(
        world_size=2, peer_modes=[rank_b_mode.value]
    )
    with (
        mock.patch(
            "vllm.v1.worker.pp_utils.get_pp_group", return_value=mock_group_a
        ),
        mock.patch(
            "vllm.v1.worker.pp_utils.torch.distributed.all_reduce",
            side_effect=fake_ar_a,
        ),
    ):
        result_a = coordinate_cudagraph_mode_across_pp(rank_a_mode.value)

    # Rank B's view: its peer (rank A) contributed rank_a_mode.
    mock_group_b, fake_ar_b = _patch_pp_group(
        world_size=2, peer_modes=[rank_a_mode.value]
    )
    with (
        mock.patch(
            "vllm.v1.worker.pp_utils.get_pp_group", return_value=mock_group_b
        ),
        mock.patch(
            "vllm.v1.worker.pp_utils.torch.distributed.all_reduce",
            side_effect=fake_ar_b,
        ),
    ):
        result_b = coordinate_cudagraph_mode_across_pp(rank_b_mode.value)

    # The core invariant: after consensus BOTH ranks agree (no split-brain).
    assert result_a == result_b, (
        f"PP ranks disagree post-consensus: {result_a} != {result_b}; "
        "this is the #45094 split-brain that wedges the pipeline."
    )
    assert result_a == expected_consensus.value
    assert result_b == expected_consensus.value


def test_pp_consensus_noop_on_single_stage():
    """With PP world_size == 1 there are no peers; the mode is returned
    unchanged and no collective is issued (cannot wedge a 1-stage pipeline)."""
    from vllm.v1.worker.pp_utils import coordinate_cudagraph_mode_across_pp

    mock_group, fake_ar = _patch_pp_group(world_size=1)
    with (
        mock.patch(
            "vllm.v1.worker.pp_utils.get_pp_group", return_value=mock_group
        ),
        mock.patch(
            "vllm.v1.worker.pp_utils.torch.distributed.all_reduce",
            side_effect=fake_ar,
        ) as mock_ar,
    ):
        result = coordinate_cudagraph_mode_across_pp(CUDAGraphMode.PIECEWISE.value)

    assert result == CUDAGraphMode.PIECEWISE.value
    mock_ar.assert_not_called()


def test_demonstrates_dispatcher_divergence_without_consensus():
    """Show the determinism gap the consensus closes: the dispatcher decision
    flips on per-rank-local state for an identical batch shape.

    Here we model the real #45094 trigger — one PP rank still owes a KV-scale
    calculation (forces NONE in the model runner) while its peer does not (gets
    PIECEWISE). Without consensus these two diverge for the same step; WITH
    consensus they collapse to NONE.
    """
    from vllm.v1.worker.pp_utils import coordinate_cudagraph_mode_across_pp

    # Same logical step / batch shape on both ranks would dispatch PIECEWISE...
    rank_with_graph = CUDAGraphMode.PIECEWISE.value
    # ...but the peer is forced eager by local calculate_kv_scales (model runner
    # line: `if self.calculate_kv_scales: cudagraph_mode = CUDAGraphMode.NONE`).
    rank_forced_eager = CUDAGraphMode.NONE.value

    # Pre-consensus: divergent (this is the wedge condition).
    assert rank_with_graph != rank_forced_eager

    # Post-consensus on the rank that wanted a graph: peer forced eager -> NONE.
    mock_group, fake_ar = _patch_pp_group(
        world_size=2, peer_modes=[rank_forced_eager]
    )
    with (
        mock.patch(
            "vllm.v1.worker.pp_utils.get_pp_group", return_value=mock_group
        ),
        mock.patch(
            "vllm.v1.worker.pp_utils.torch.distributed.all_reduce",
            side_effect=fake_ar,
        ),
    ):
        synced = coordinate_cudagraph_mode_across_pp(rank_with_graph)

    assert synced == CUDAGraphMode.NONE.value, (
        "PP rank that captured a graph must defer to its eager peer; "
        "otherwise it replays a graph while the peer runs eager -> #45094 wedge."
    )


# ---------------------------------------------------------------------------
# Integration test: the execute_model re-dispatch ORDERING, not just the leaf
# helper. Reproduces the real #45094 trigger end-to-end — calculate_kv_scales
# forcing eager on ONE PP rank — and proves that consensus is only correct when
# that force is folded in BEFORE the PP all-reduce (post-revision), and that the
# pre-revision ordering (force AFTER consensus) re-introduces the split-brain.
# ---------------------------------------------------------------------------


class _FakeDispatcher:
    """Faithful stand-in for CudagraphDispatcher.dispatch w.r.t. the contract
    this fix depends on:

    * ``valid_modes`` is an allow-list; ``NONE`` is the universal fallback.
    * If the agreed (synced) mode has no captured key on this rank, dispatch
      falls back to ``NONE`` — but ONLY if ``NONE`` is in ``valid_modes``;
      otherwise the real dispatcher asserts (``assert NONE in allowed_modes``).
      We raise AssertionError to mirror that crash (covers MED-2).

    ``captured_modes`` is the set of modes for which THIS rank actually holds a
    capture key for the step's batch shape.
    """

    def __init__(self, captured_modes: set[CUDAGraphMode]):
        self.captured_modes = captured_modes

    def dispatch(self, valid_modes: set[CUDAGraphMode] | None):
        allowed = valid_modes or {
            CUDAGraphMode.NONE,
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.FULL,
        }
        # Prefer the richest captured mode that is allowed (FULL > PIECEWISE).
        for mode in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE):
            if mode in allowed and mode in self.captured_modes:
                return mode
        # No captured key for an allowed graph mode -> must fall back to NONE.
        assert CUDAGraphMode.NONE in allowed, (
            "dispatcher would assert: NONE not in allowed_modes "
            f"({allowed}); this is the MED-2 re-dispatch crash."
        )
        return CUDAGraphMode.NONE


def _run_one_rank(
    *,
    dispatcher: "_FakeDispatcher",
    peer_modes: list[int],
    calculate_kv_scales: bool,
    thread_kv_scales_before_consensus: bool,
) -> CUDAGraphMode:
    """Mirror gpu_model_runner's execute_model cudagraph-mode resolution for a
    single PP rank, using the REAL coordinate_cudagraph_mode_across_pp.

    thread_kv_scales_before_consensus:
        True  -> POST-revision ordering: kv_scales force folded into the
                 dispatch BEFORE the PP all-reduce (force eager up front).
        False -> PRE-revision ordering: dispatch + PP consensus first, THEN
                 force NONE for kv_scales (the #45094 bug).
    """
    from vllm.v1.worker.pp_utils import coordinate_cudagraph_mode_across_pp

    # --- initial dispatch (optionally already forced eager, post-revision) ---
    if thread_kv_scales_before_consensus and calculate_kv_scales:
        initial_valid = {CUDAGraphMode.NONE}
    else:
        initial_valid = None
    cudagraph_mode = dispatcher.dispatch(valid_modes=initial_valid)

    # --- PP consensus all-reduce (real helper, mocked peers) ---
    mock_group, fake_ar = _patch_pp_group(world_size=2, peer_modes=peer_modes)
    with (
        mock.patch(
            "vllm.v1.worker.pp_utils.get_pp_group", return_value=mock_group
        ),
        mock.patch(
            "vllm.v1.worker.pp_utils.torch.distributed.all_reduce",
            side_effect=fake_ar,
        ),
    ):
        synced = coordinate_cudagraph_mode_across_pp(cudagraph_mode.value)
    if synced != cudagraph_mode.value:
        # Re-dispatch at the agreed mode. MED-2: keep NONE so a rank lacking
        # the agreed mode's capture key falls back instead of asserting.
        cudagraph_mode = dispatcher.dispatch(
            valid_modes={CUDAGraphMode.NONE, CUDAGraphMode(synced)}
        )

    # --- pre-revision: local kv_scales force applied AFTER consensus (BUG) ---
    if not thread_kv_scales_before_consensus and calculate_kv_scales:
        cudagraph_mode = CUDAGraphMode.NONE

    return cudagraph_mode


def test_integration_kv_scales_before_consensus_keeps_pp_consistent():
    """POST-revision: rank A owes a KV-scale calc (forces eager), rank B does
    not. Both hold a PIECEWISE key for the step. With the force threaded BEFORE
    the all-reduce, rank A contributes NONE to the consensus, so rank B is
    pulled down to NONE too -> both ranks run eager, no split-brain."""
    rank_a = _run_one_rank(
        dispatcher=_FakeDispatcher({CUDAGraphMode.PIECEWISE}),
        peer_modes=[CUDAGraphMode.PIECEWISE.value],  # B before consensus
        calculate_kv_scales=True,
        thread_kv_scales_before_consensus=True,
    )
    rank_b = _run_one_rank(
        dispatcher=_FakeDispatcher({CUDAGraphMode.PIECEWISE}),
        peer_modes=[CUDAGraphMode.NONE.value],  # A already forced eager
        calculate_kv_scales=False,
        thread_kv_scales_before_consensus=True,
    )
    assert rank_a == rank_b == CUDAGraphMode.NONE, (
        f"PP ranks diverged post-revision: A={rank_a}, B={rank_b}. The "
        "kv_scales eager-force must propagate through PP consensus."
    )


def test_integration_pre_revision_ordering_reproduces_split_brain():
    """PRE-revision: the SAME scenario but with the kv_scales force applied
    AFTER the PP consensus. Rank A contributes PIECEWISE to the all-reduce
    (its force hasn't happened yet), so neither rank is pulled to NONE; then
    rank A forces itself to NONE locally while rank B replays its PIECEWISE
    graph. Ranks diverge -> exactly the #45094 wedge. This test documents the
    bug the revision fixes; it asserts the divergence EXISTS pre-revision."""
    rank_a = _run_one_rank(
        dispatcher=_FakeDispatcher({CUDAGraphMode.PIECEWISE}),
        peer_modes=[CUDAGraphMode.PIECEWISE.value],
        calculate_kv_scales=True,
        thread_kv_scales_before_consensus=False,
    )
    rank_b = _run_one_rank(
        dispatcher=_FakeDispatcher({CUDAGraphMode.PIECEWISE}),
        peer_modes=[CUDAGraphMode.PIECEWISE.value],
        calculate_kv_scales=False,
        thread_kv_scales_before_consensus=False,
    )
    assert rank_a == CUDAGraphMode.NONE
    assert rank_b == CUDAGraphMode.PIECEWISE
    assert rank_a != rank_b, (
        "Pre-revision ordering should reproduce the #45094 split-brain; if "
        "this no longer diverges the test no longer guards the regression."
    )


def test_integration_redispatch_falls_back_when_no_capture_key():
    """MED-2: a rank is pulled to PIECEWISE by consensus but holds NO PIECEWISE
    capture key (e.g. FULL_DECODE_ONLY build). The re-dispatch must fall back to
    NONE rather than assert-crash. With valid_modes={NONE, PIECEWISE} it falls
    back; with {PIECEWISE} alone it would raise (the pre-fix MED-2 crash)."""
    # Peer agreed PIECEWISE (lower than this rank's FULL), forcing re-dispatch.
    rank = _run_one_rank(
        dispatcher=_FakeDispatcher({CUDAGraphMode.FULL}),  # no PIECEWISE key
        peer_modes=[CUDAGraphMode.PIECEWISE.value],
        calculate_kv_scales=False,
        thread_kv_scales_before_consensus=True,
    )
    assert rank == CUDAGraphMode.NONE, (
        "re-dispatch must fall back to NONE when the agreed mode has no "
        "capture key on this rank (MED-2)."
    )

    # And prove the bug it guards: pinning to {synced} alone asserts.
    bad = _FakeDispatcher({CUDAGraphMode.FULL})
    with pytest.raises(AssertionError):
        bad.dispatch(valid_modes={CUDAGraphMode.PIECEWISE})


# ---------------------------------------------------------------------------
# Source-level guard: tie the regression test to the ACTUAL gpu_model_runner
# code so a future revert of the threading fails CI here even though the unit
# helpers still pass. Asserts (a) execute_model threads force_eager_kv_scales
# into _determine_batch_execution_and_padding, and (b) there is no
# `cudagraph_mode = CUDAGraphMode.NONE` reassignment positioned AFTER the PP
# consensus block (which was the #45094 ordering bug).
# ---------------------------------------------------------------------------


def _gpu_model_runner_source() -> str:
    # Read the file directly (not via import) so this guard does not drag in the
    # full model_runner import chain (multimodal, lora, prometheus, ...). Locate
    # it relative to this test file: <repo>/vllm/v1/worker/gpu_model_runner.py.
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / "vllm" / "v1" / "worker" / "gpu_model_runner.py").read_text()


def _gpu_worker_source() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / "vllm" / "v1" / "worker" / "gpu_worker.py").read_text()


def test_source_threads_kv_scales_force_into_dispatch():
    src = _gpu_model_runner_source()
    assert "force_eager_kv_scales=self.calculate_kv_scales" in src, (
        "execute_model must thread calculate_kv_scales into "
        "_determine_batch_execution_and_padding so the eager force happens "
        "BEFORE the PP/DP consensus all-reduce (#45094)."
    )
    assert "force_eager = force_eager or force_eager_kv_scales" in src, (
        "_determine_batch_execution_and_padding must fold the kv_scales force "
        "into the dispatch decision ahead of the consensus reductions."
    )


def test_source_has_no_kv_scales_force_after_pp_consensus():
    """The pre-revision bug was `cudagraph_mode = CUDAGraphMode.NONE` for
    calculate_kv_scales executed AFTER the PP consensus. The consensus all-reduce
    now lives in Worker.execute_model (hoisted above the inter-stage recv,
    #45610), and the runner consumes the agreed value via
    `pp_synced_cudagraph_mode`. Ensure no calculate_kv_scales-driven NONE force
    reappears downstream of that consumption point in the runner."""
    src = _gpu_model_runner_source()
    anchor = "pp_synced_cudagraph_mode is not None"
    idx = src.find(anchor)
    assert idx != -1, (
        "pp_synced_cudagraph_mode consumption site not found in "
        "gpu_model_runner; the runner must consume the worker-agreed PP mode."
    )
    after = src[idx:]
    # The kv_scales fold must happen BEFORE this point (via force_eager_kv_scales,
    # folded inside _determine_batch_execution_and_padding). The old buggy pair
    # (a bare calculate_kv_scales force re-deciding the mode) must not reappear
    # after consensus consumption.
    assert not (
        "if self.calculate_kv_scales:" in after
        and "cudagraph_mode = CUDAGraphMode.NONE" in after
    ), (
        "Found a calculate_kv_scales-driven `cudagraph_mode = NONE` AFTER the "
        "PP consensus consumption — this is the #45094 split-brain ordering. "
        "Thread the force into _determine_batch_execution_and_padding instead."
    )


# ---------------------------------------------------------------------------
# Ordering guard (#45610): the PP consensus all-reduce MUST run BEFORE the
# inter-stage recv.
#
# Root cause (root-caused live via py-spy): the group-wide PP all-reduce
# `coordinate_cudagraph_mode_across_pp` used to be issued INSIDE the model
# runner's execute_model -> _determine_batch_execution_and_padding, which
# Worker.execute_model (vllm/v1/worker/gpu_worker.py) invokes AFTER the stage-1
# `irecv_tensor_dict`. Stage 0's activation SEND happens only after its runner
# returns. That forms a structural deadlock cycle on the very first real decode
# step:
#   stage0.all_reduce  waits-for  stage1.all_reduce
#   stage1.recv        waits-for  stage0.activation_send (after stage0.all_reduce)
# so PP0 parks in coordinate_cudagraph_mode_across_pp -> all_reduce while PP1
# parks in irecv_tensor_dict gloo waitRecv, forever. Triton-JIT only affects
# timing; the inversion is unconditional.
#
# The fix hoists the consensus into Worker.execute_model, ahead of the recv, and
# threads the agreed mode down to the runner (which no longer self-issues the
# all-reduce). These tests pin that ordering: a behavioral fake of
# Worker.execute_model on a stage-1 (not is_first_rank) PP rank records the order
# of the all-reduce vs. the recv, and an AST guard pins the source placement.
# Both FAIL against the pre-fix tree (where the all-reduce trailed the recv).
# ---------------------------------------------------------------------------


def test_pp_all_reduce_precedes_irecv_in_worker_execute_model():
    """Layer A (ordering): drive a real Worker.execute_model on a stage-1
    (not is_first_rank) PP rank, world_size=2, with all_reduce + irecv recorded
    as fakes, and assert the PP all-reduce is issued BEFORE irecv_tensor_dict.

    Pre-fix the all-reduce lived inside the runner (after the worker's recv) so
    the recorded order was [irecv, ...] with the all-reduce never reached on a
    real pipeline (deadlock); here the runner is a stub, so pre-fix would record
    irecv first (or no all-reduce at all from the worker). Post-fix the worker
    issues the all-reduce first."""
    import vllm.v1.worker.gpu_worker as gw

    events: list[str] = []

    # --- fake PP group: world_size 2, this rank is NOT the first stage (so it
    #     performs the inter-stage recv), and IS not the last (so it would send).
    pp_group = mock.MagicMock()
    pp_group.world_size = 2
    pp_group.is_first_rank = False
    pp_group.is_last_rank = False
    pp_group.cpu_group = mock.MagicMock()

    def fake_irecv(*args, **kwargs):
        events.append("irecv_tensor_dict")
        # (tensor_dict, comm_handles, comm_postprocess)
        return {"hidden_states": object()}, None, None

    pp_group.irecv_tensor_dict.side_effect = fake_irecv

    def fake_all_reduce(tensor, op, group):
        events.append("pp_all_reduce")
        # min over a single simulated peer that agrees (no-op on value)
        tensor.fill_(int(tensor.item()))

    # --- minimal worker stand-in: only the attributes Worker.execute_model
    #     touches up to and including the runner call. The runner is a stub that
    #     returns a ModelRunnerOutput-like sentinel so the worker returns early.
    worker = mock.MagicMock()
    worker._pp_send_work = []
    worker.use_v2_model_runner = False
    worker.vllm_config.compilation_config.pass_config.enable_sp = False
    worker.vllm_config.parallel_config.pipeline_parallel_size = 2

    # predispatch returns this rank's local mode; execute_model records nothing
    # about ordering but must be a real ModelRunnerOutput-typed object so the
    # worker returns it. We use NoneType (a valid early-return type).
    worker.model_runner.predispatch_cudagraph_mode.return_value = (
        CUDAGraphMode.PIECEWISE.value
    )
    worker.model_runner.execute_model.return_value = None  # NoneType -> early return

    scheduler_output = mock.MagicMock()
    scheduler_output.total_num_scheduled_tokens = 8

    with (
        mock.patch.object(gw, "get_pp_group", return_value=pp_group),
        mock.patch.object(gw, "get_tp_group", return_value=mock.MagicMock()),
        mock.patch.object(
            gw,
            "coordinate_cudagraph_mode_across_pp",
            wraps=lambda v: v,
        ) as wrapped_consensus,
    ):
        # Record the consensus call as the all-reduce event (it wraps the real
        # all-reduce; recording at the helper boundary is equivalent for ordering
        # and avoids needing a live gloo group).
        def record_consensus(v):
            events.append("pp_all_reduce")
            return v

        wrapped_consensus.side_effect = record_consensus

        gw.Worker.execute_model(worker, scheduler_output)

    assert "pp_all_reduce" in events, (
        "Worker.execute_model must issue the PP cudagraph-mode consensus "
        "all-reduce; it was never called."
    )
    assert "irecv_tensor_dict" in events, (
        "test setup error: the stage-1 recv was not exercised."
    )
    assert events.index("pp_all_reduce") < events.index("irecv_tensor_dict"), (
        f"PP all-reduce must precede the inter-stage recv; got order {events}. "
        "If the all-reduce trails the recv the pipeline deadlocks on the first "
        "real decode step (#45094/#45610)."
    )


def _call_lineno_in_function(
    src: str, func_name: str, callee_name: str
) -> int | None:
    """Return the source line of the FIRST *call* to ``callee_name`` inside the
    function ``func_name`` (matching either ``callee_name(...)`` or
    ``obj.callee_name(...)``), or ``None`` if there is no such call.

    Unlike a bare ``str.find(callee_name)`` this ignores import lines, comments,
    and docstrings — so it pins the real CALL-SITE ORDERING, not the position of
    a module-level ``import`` (which always sorts to the top of the file and
    makes a naive find-based ordering check pass even if the call is misplaced).
    """
    import ast

    tree = ast.parse(src)
    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == func_name
        ),
        None,
    )
    if fn is None:
        return None
    best: int | None = None
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        if name == callee_name:
            ln = node.lineno
            if best is None or ln < best:
                best = ln
    return best


def test_source_worker_issues_pp_consensus_before_irecv():
    """Layer B (AST/source guard): in Worker.execute_model the PP consensus CALL
    (coordinate_cudagraph_mode_across_pp) must execute BEFORE the
    irecv_tensor_dict CALL, and the runner's
    _determine_batch_execution_and_padding must NO LONGER self-issue the
    all-reduce.

    HARDENED (was previously a naive ``wsrc.find(...)`` over the whole file,
    which matched the module-level ``from ... import
    coordinate_cudagraph_mode_across_pp`` at the TOP of the file — so it passed
    even if the actual CALL were moved BELOW the recv, the exact #45610 deadlock
    this guards). We now resolve the line of each *call* inside the
    Worker.execute_model function body via the AST and compare CALL-SITE lines."""
    wsrc = _gpu_worker_source()
    consensus_call_ln = _call_lineno_in_function(
        wsrc, "execute_model", "coordinate_cudagraph_mode_across_pp"
    )
    irecv_call_ln = _call_lineno_in_function(
        wsrc, "execute_model", "irecv_tensor_dict"
    )
    assert consensus_call_ln is not None, (
        "Worker.execute_model must CALL coordinate_cudagraph_mode_across_pp "
        "(not merely import it); the PP consensus was hoisted into the worker "
        "(#45610). No call found in the function body."
    )
    assert irecv_call_ln is not None, (
        "Worker.execute_model must call irecv_tensor_dict (the inter-stage recv)."
    )
    assert consensus_call_ln < irecv_call_ln, (
        "The PP consensus all-reduce CALL "
        f"(line {consensus_call_ln}) must execute BEFORE the irecv_tensor_dict "
        f"CALL (line {irecv_call_ln}) in Worker.execute_model; otherwise the "
        "collective and the recv deadlock (#45094/#45610). NOTE: this asserts "
        "the *call* order, not the import position."
    )

    # Belt-and-braces: the import must NOT be the only occurrence (i.e. a real
    # call exists), and the textual import precedes every call — proving the old
    # naive find() would have matched the import and could not have detected a
    # misplaced call.
    import_idx = wsrc.find(
        "from vllm.v1.worker.pp_utils import coordinate_cudagraph_mode_across_pp"
    )
    assert import_idx != -1, "expected the helper to be imported."

    # The runner must consume the agreed value, NOT issue the all-reduce itself.
    rsrc = _gpu_model_runner_source()
    assert _call_lineno_in_function(
        rsrc, "_determine_batch_execution_and_padding",
        "coordinate_cudagraph_mode_across_pp",
    ) is None, (
        "gpu_model_runner._determine_batch_execution_and_padding must not CALL "
        "coordinate_cudagraph_mode_across_pp anymore (the all-reduce moved to "
        "the worker, ahead of the recv)."
    )
    assert "pp_synced_cudagraph_mode" in rsrc, (
        "_determine_batch_execution_and_padding must accept and consume the "
        "worker-agreed pp_synced_cudagraph_mode."
    )


def test_predispatch_cudagraph_mode_issues_no_collective():
    """predispatch_cudagraph_mode (called pre-recv in the worker) must compute a
    LOCAL mode without any collective — the all-reduce is the worker's job,
    separately, right after. AST: predispatch's body must not reference
    coordinate_cudagraph_mode_across_pp or torch.distributed.all_reduce."""
    import ast

    src = _gpu_model_runner_source()
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "predispatch_cudagraph_mode"
    )
    # Exclude the docstring (which legitimately *mentions* the helper) and check
    # only the executable statements.
    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    body_src = "\n".join(ast.get_source_segment(src, stmt) or "" for stmt in body)
    assert "coordinate_cudagraph_mode_across_pp" not in body_src, (
        "predispatch_cudagraph_mode must NOT issue the PP consensus; it only "
        "computes this rank's local mode. The worker issues the all-reduce."
    )
    assert "all_reduce" not in body_src, (
        "predispatch_cudagraph_mode must issue no collective (no all_reduce)."
    )


def test_dummy_run_path_does_not_call_predispatch_or_consensus():
    """The dummy/capture path (_dummy_run) must never trigger the PP consensus.
    It calls _determine_batch_execution_and_padding with pp_synced_cudagraph_mode
    left as the None default (so no reconciliation), and never calls
    predispatch_cudagraph_mode (only the worker pre-recv path does)."""
    import ast

    src = _gpu_model_runner_source()
    tree = ast.parse(src)
    dummy = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_dummy_run"
    )
    for node in ast.walk(dummy):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        assert name != "predispatch_cudagraph_mode", (
            "_dummy_run must not call predispatch_cudagraph_mode; the PP "
            "consensus must never be tied to the per-stage-asynchronous capture "
            "path or PP>1 deadlocks at startup (#45610)."
        )
        if name == "_determine_batch_execution_and_padding":
            for kw in node.keywords:
                assert kw.arg != "pp_synced_cudagraph_mode", (
                    "_dummy_run must leave pp_synced_cudagraph_mode at its None "
                    "default (no PP reconciliation on the capture path)."
                )


def test_worker_threads_pp_synced_mode_into_runner_execute_model():
    """Positive call-site test (work-item 2d): on the real execute_model path the
    worker not only RUNS the consensus before the recv, it THREADS the agreed
    value into model_runner.execute_model via the pp_synced_cudagraph_mode kwarg.

    Behavioral: drive Worker.execute_model on a stage-1 (not is_first_rank) PP
    rank; the consensus stub returns a sentinel agreed mode (NONE); assert the
    runner's execute_model received exactly that value as
    pp_synced_cudagraph_mode. Pre-fix the worker did not compute or pass it at
    all -> kwarg absent / None."""
    import vllm.v1.worker.gpu_worker as gw

    pp_group = mock.MagicMock()
    pp_group.world_size = 2
    pp_group.is_first_rank = False
    pp_group.is_last_rank = False
    pp_group.cpu_group = mock.MagicMock()
    pp_group.irecv_tensor_dict.return_value = ({"hidden_states": object()}, None, None)

    worker = mock.MagicMock()
    worker._pp_send_work = []
    worker.use_v2_model_runner = False
    worker.vllm_config.compilation_config.pass_config.enable_sp = False
    worker.vllm_config.parallel_config.pipeline_parallel_size = 2
    # This rank locally wanted PIECEWISE...
    worker.model_runner.predispatch_cudagraph_mode.return_value = (
        CUDAGraphMode.PIECEWISE.value
    )
    worker.model_runner.execute_model.return_value = None  # early return

    scheduler_output = mock.MagicMock()
    scheduler_output.total_num_scheduled_tokens = 8

    with (
        mock.patch.object(gw, "get_pp_group", return_value=pp_group),
        mock.patch.object(gw, "get_tp_group", return_value=mock.MagicMock()),
        # ...but the PP consensus agreed NONE (a peer chose eager).
        mock.patch.object(
            gw,
            "coordinate_cudagraph_mode_across_pp",
            return_value=CUDAGraphMode.NONE.value,
        ),
    ):
        gw.Worker.execute_model(worker, scheduler_output)

    assert worker.model_runner.execute_model.called, "runner execute_model not called"
    _, kwargs = worker.model_runner.execute_model.call_args
    assert "pp_synced_cudagraph_mode" in kwargs, (
        "Worker.execute_model must thread the agreed mode into the runner via "
        "the pp_synced_cudagraph_mode kwarg; it was not passed."
    )
    assert kwargs["pp_synced_cudagraph_mode"] == CUDAGraphMode.NONE.value, (
        "The worker must pass the CONSENSUS value "
        f"({CUDAGraphMode.NONE.value}) to the runner, not this rank's local "
        f"pre-consensus mode; got {kwargs['pp_synced_cudagraph_mode']}."
    )


def test_predispatch_only_kv_scale_inputs_issues_no_collective():
    """Work-item 2a (behavioral, V1): predispatch_cudagraph_mode is the
    dispatch-only prefix — given a step whose only eager driver is KV-scale
    calibration (calculate_kv_scales=True), it must resolve a LOCAL mode (NONE,
    via the force) and issue NO collective. We stub the dispatcher and assert it
    is consulted with the eager-forcing valid_modes while NO all_reduce / PP
    consensus is touched.

    This complements the AST guard (test_predispatch_cudagraph_mode_issues_no_collective)
    with a runtime check that the function body actually runs collective-free."""
    import vllm.v1.worker.gpu_model_runner as gmr

    runner = mock.MagicMock()
    runner.calculate_kv_scales = True  # the canonical #45094 eager trigger
    runner.model_config.is_encoder_decoder = False
    runner.input_batch.lora_id_to_lora_request = {}
    runner.uniform_decode_query_len = 1
    runner._is_uniform_decode.return_value = True
    runner._pad_for_sequence_parallelism.side_effect = lambda n: n
    # Dispatcher returns whatever valid_modes pins; record the call.
    dispatch_calls: list = []

    def fake_dispatch(**kwargs):
        dispatch_calls.append(kwargs)
        return CUDAGraphMode.NONE, mock.MagicMock()

    runner.cudagraph_dispatcher.dispatch.side_effect = fake_dispatch

    scheduler_output = mock.MagicMock()
    scheduler_output.num_scheduled_tokens = {0: 1, 1: 1}
    scheduler_output.total_num_scheduled_tokens = 2
    scheduler_output.scheduled_encoder_inputs = {}

    # NOTE: the AST guard test_predispatch_cudagraph_mode_issues_no_collective is
    # the AUTHORITATIVE proof that this function body issues no collective (it
    # inspects the source so it cannot be fooled by a mock that misses the real
    # call site). This runtime check is complementary. predispatch_cudagraph_mode
    # never imports/calls coordinate_cudagraph_mode_across_pp, so the only
    # all_reduce that could fire is gmr's own; we patch that here, and also patch
    # the pp_utils all_reduce (where the real PP consensus collective lives) so a
    # future regression that wired the consensus into predispatch would be caught
    # at runtime too, not only by the AST guard.
    import vllm.v1.worker.pp_utils as pp_utils

    with (
        mock.patch.object(gmr.torch.distributed, "all_reduce") as mock_ar,
        mock.patch.object(pp_utils.torch.distributed, "all_reduce") as mock_pp_ar,
    ):
        result = gmr.GPUModelRunner.predispatch_cudagraph_mode(
            runner, scheduler_output
        )

    mock_ar.assert_not_called()
    mock_pp_ar.assert_not_called()
    assert result == CUDAGraphMode.NONE.value
    assert dispatch_calls, "dispatcher must be consulted"
    # KV-scale force must be folded as an eager-only allow-list before consensus.
    assert dispatch_calls[0].get("valid_modes") == {CUDAGraphMode.NONE}, (
        "predispatch must fold calculate_kv_scales into valid_modes={NONE} so "
        "the value contributed to the PP MIN already reflects the eager force "
        f"(got valid_modes={dispatch_calls[0].get('valid_modes')})."
    )
