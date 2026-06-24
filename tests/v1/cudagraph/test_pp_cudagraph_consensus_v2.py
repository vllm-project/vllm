# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2-model-runner regression tests for the pipeline-parallel cudagraph-mode
consensus that guards against the #45094 PP cudagraph-vs-eager split-brain wedge.

This is the V2 analog of ``tests/v1/cudagraph/test_pp_cudagraph_consensus.py``.
The maintainer (njhill, vllm-project/vllm#45610) noted that vLLM is migrating to
the V2 model runner and recommends it for pipeline parallelism, and asked whether
the #45094 split-brain also applies there. It does:

* The V2 dispatch+sync path is ``dispatch_cg_and_sync_dp``
  (``vllm/v1/worker/gpu/dp_utils.py``). It takes a MIN of ``cg_mode`` across the
  *DP* group (``sync_cudagraph_and_dp_padding``) but had **no PP-rank consensus**.
* ``CudaGraphManager.dispatch`` runs independently per PP stage, so the same
  per-rank-local divergence the V1 fix addresses (calculate_kv_scales force-eager,
  encoder-decoder skip_compiled, cascade-attn detection, a partially-failed
  capture, LoRA bookkeeping) produces a PIECEWISE/FULL-vs-NONE split-brain across
  PP stages on V2 too — the stages disagree on the inter-stage P2P send/recv
  schedule, the rendezvous never completes, and the engine wedges
  (``shm_broadcast`` 60s timeout -> RPC timeout -> EngineCore crash).

The fix mirrors V1: a MIN all-reduce of the cudagraph mode across the PP CPU
(gloo) group, reusing the shared V1 helper ``coordinate_cudagraph_mode_across_pp``
(``vllm/v1/worker/pp_utils.py``). It is gated to the real (non-dummy, non-profile)
execute path so the collective is never issued during the per-stage-asynchronous
capture/profile path (that gating was the #45610 startup-deadlock follow-up).

These are unit tests against ``dp_utils._coordinate_cudagraph_mode_across_pp`` and
``dispatch_cg_and_sync_dp`` with a mocked PP group, mirroring how the V1 tests
mock ``get_pp_group``. They FAIL against pre-fix code (the function does not exist
/ no PP consensus -> the two simulated stages stay divergent).

Integration repro (requires >=2 GPUs, PP=2, V2 runner):

    VLLM_USE_V1_MODEL_RUNNER_V2=1 vllm serve <model> -tp 1 -pp 2 \\
        --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
    # then drive concurrent long-reasoning requests whose admitted batch-token
    # count crosses a captured cudagraph bucket boundary on some steps but not
    # others. Pre-fix: one PP stage replays the graph while the other runs eager
    # -> shm_broadcast "No available shared memory broadcast block found in 60
    # seconds" -> RPC timeout -> EngineCore crash. Post-fix: both stages agree
    # (MIN) -> no wedge.
"""

from pathlib import Path
from unittest import mock

import pytest

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


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
        local = int(tensor.item())
        tensor.fill_(min([local, *peer_modes]))

    return mock_group, fake_all_reduce


class _FakeCudaGraphManager:
    """Minimal stand-in for CudaGraphManager.dispatch.

    ``captured_mode`` is the richest mode this rank holds a capture key for at
    the step's batch shape. ``dispatch`` returns a descriptor at that mode
    (FULL/PIECEWISE) or NONE if it holds no graph — matching the real
    dispatcher's "best matching captured descriptor, else NONE" contract.
    """

    def __init__(self, captured_mode: CUDAGraphMode):
        self.captured_mode = captured_mode

    def dispatch(self, num_reqs, num_tokens, uniform_token_count):
        return BatchExecutionDescriptor(
            cg_mode=self.captured_mode,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
        )


def _run_pp_consensus_one_rank(
    *,
    local_mode: CUDAGraphMode,
    peer_modes: list[int],
) -> CUDAGraphMode:
    """Drive the V2 reconciliation for one PP rank: compute the agreed MIN via
    the REAL leaf helper ``coordinate_cudagraph_mode_across_pp`` (with mocked PP
    peers) — exactly as ``Worker.execute_model`` now does *before* the recv —
    then feed it to ``dp_utils._reconcile_cudagraph_mode_across_pp``. Returns the
    post-reconciliation cg_mode."""
    from vllm.v1.worker.gpu import dp_utils
    from vllm.v1.worker.pp_utils import coordinate_cudagraph_mode_across_pp

    batch_desc = BatchExecutionDescriptor(
        cg_mode=local_mode, num_tokens=128, num_reqs=4
    )

    mock_group, fake_ar = _patch_pp_group(world_size=2, peer_modes=peer_modes)
    with (
        mock.patch("vllm.v1.worker.gpu.dp_utils.get_pp_group", return_value=mock_group),
        mock.patch("vllm.v1.worker.pp_utils.get_pp_group", return_value=mock_group),
        mock.patch(
            "vllm.v1.worker.pp_utils.torch.distributed.all_reduce",
            side_effect=fake_ar,
        ),
    ):
        # The worker computes the agreed MIN ahead of the recv...
        pp_synced = coordinate_cudagraph_mode_across_pp(local_mode.value)
        # ...and the runner consumes it (no collective issued here).
        synced = dp_utils._reconcile_cudagraph_mode_across_pp(
            batch_desc, 4, 128, pp_synced
        )
    return synced.cg_mode


@pytest.mark.parametrize(
    "rank_a_mode, rank_b_mode, expected_a, expected_b",
    [
        # The #45094 split-brain on V2: one stage on a graph, the other NONE
        # (eager). Both must end eager so the inter-stage P2P schedule agrees.
        (
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.NONE,
            CUDAGraphMode.NONE,
            CUDAGraphMode.NONE,
        ),
        (
            CUDAGraphMode.NONE,
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.NONE,
            CUDAGraphMode.NONE,
        ),
        (
            CUDAGraphMode.FULL,
            CUDAGraphMode.NONE,
            CUDAGraphMode.NONE,
            CUDAGraphMode.NONE,
        ),
        # FULL vs PIECEWISE: an UNREACHABLE same-shape graph-vs-graph divergence
        # (see the NOTE on the test below), pinned only to document the
        # mechanical behavior. The PP MIN is PIECEWISE; the PIECEWISE rank is
        # already at the MIN and keeps it, while the FULL rank cannot be
        # re-dispatched to PIECEWISE (V2 dispatch is shape-deterministic, no
        # valid_modes allow-list) so it drops to NONE. That is a *split*
        # (NONE vs PIECEWISE), NOT a convergence — it is safe ONLY because this
        # input cannot occur (config-global capture keys => all graph-capable
        # stages dispatch the same graph mode at a given shape; the only
        # reachable PP divergence is graph-vs-NONE, covered by the rows above,
        # which DO converge on NONE).
        (
            CUDAGraphMode.FULL,
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.NONE,
            CUDAGraphMode.PIECEWISE,
        ),
        # Already in agreement -> unchanged (no divergence, no drop).
        (
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.PIECEWISE,
        ),
        (
            CUDAGraphMode.FULL,
            CUDAGraphMode.FULL,
            CUDAGraphMode.FULL,
            CUDAGraphMode.FULL,
        ),
        (
            CUDAGraphMode.NONE,
            CUDAGraphMode.NONE,
            CUDAGraphMode.NONE,
            CUDAGraphMode.NONE,
        ),
    ],
)
def test_v2_pp_consensus_reconciles_divergent_modes(
    rank_a_mode, rank_b_mode, expected_a, expected_b
):
    """Each PP rank that dispatched a mode != the PP MIN drops to eager on the
    V2 path; a rank already at the MIN keeps it. The guard against #45094.

    NOTE on the FULL-vs-PIECEWISE row: it is an UNREACHABLE input, pinned only to
    document the mechanical behavior — and that behavior is a *split*
    (a=NONE, b=PIECEWISE), NOT a convergence. It is safe precisely because the
    input cannot occur: cudagraph capture sizes are config-global, so all
    graph-capable PP stages hold identical capture keys and dispatch the SAME
    graph mode at a given batch shape. The only PP divergence that can actually
    arise is graph-vs-NONE (one stage forced eager by per-step-local state:
    calculate_kv_scales, cascade-attn detection, LoRA bookkeeping, a partially
    failed capture), and for that the MIN is NONE so BOTH stages drop to NONE and
    converge (the graph-vs-NONE rows above, and
    test_v2_pp_consensus_graph_vs_eager_always_converges). We assert the split
    here for the FULL/PIECEWISE row ONLY to lock the mechanical contract; if a
    future change ever makes same-shape FULL-vs-PIECEWISE reachable, this split
    becomes a real wedge and the reconcile must be taught to re-dispatch toward
    the agreed mode (as the V1 path does via valid_modes={NONE, pp_synced})."""
    result_a = _run_pp_consensus_one_rank(
        local_mode=rank_a_mode, peer_modes=[rank_b_mode.value]
    )
    result_b = _run_pp_consensus_one_rank(
        local_mode=rank_b_mode, peer_modes=[rank_a_mode.value]
    )

    assert result_a == expected_a
    assert result_b == expected_b

    # Honesty guard: for every REACHABLE row both stages must end on the SAME
    # mode (no split-brain). The single FULL-vs-PIECEWISE row is the documented
    # UNREACHABLE exception (see the docstring) and is the only place the modes
    # are allowed to differ.
    graph_vs_graph_unreachable = {rank_a_mode, rank_b_mode} == {
        CUDAGraphMode.FULL,
        CUDAGraphMode.PIECEWISE,
    }
    if not graph_vs_graph_unreachable:
        assert result_a == result_b, (
            f"reachable PP divergence ({rank_a_mode} vs {rank_b_mode}) must "
            f"converge, got a={result_a}, b={result_b} — that is the #45094 "
            "split-brain."
        )


def test_v2_pp_consensus_graph_vs_eager_always_converges():
    """The load-bearing invariant for the realistic divergence (graph-vs-NONE):
    BOTH stages end at the SAME mode (NONE), eliminating the split-brain."""
    for graph_mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
        a = _run_pp_consensus_one_rank(
            local_mode=graph_mode, peer_modes=[CUDAGraphMode.NONE.value]
        )
        b = _run_pp_consensus_one_rank(
            local_mode=CUDAGraphMode.NONE, peer_modes=[graph_mode.value]
        )
        assert a == b == CUDAGraphMode.NONE, (
            f"graph({graph_mode}) vs eager PP stages diverged: a={a}, b={b}; "
            "this is the #45094 split-brain that wedges the pipeline."
        )


def test_v2_pp_consensus_noop_on_single_stage():
    """With PP world_size == 1 there are no peers; the descriptor is returned
    unchanged (the reconciliation short-circuits)."""
    from vllm.v1.worker.gpu import dp_utils

    batch_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=128, num_reqs=4
    )
    mock_group, _ = _patch_pp_group(world_size=1)
    with mock.patch(
        "vllm.v1.worker.gpu.dp_utils.get_pp_group", return_value=mock_group
    ):
        # pp_synced is irrelevant on a 1-stage pipeline; pass this rank's own
        # value to show it is returned unchanged.
        result = dp_utils._reconcile_cudagraph_mode_across_pp(
            batch_desc, 4, 128, CUDAGraphMode.PIECEWISE.value
        )

    assert result is batch_desc  # unchanged, no drop


def test_v2_dispatch_cg_and_sync_dp_runs_pp_consensus_when_dp1():
    """End-to-end through the public entry point: dp_size==1 (pure PP) with the
    PP-agreed mode supplied (as Worker.execute_model now does, pre-recv), the two
    stages still reconcile (DP MIN never runs in pure PP, so the PP
    reconciliation is the only guard)."""
    from vllm.v1.worker.gpu import dp_utils
    from vllm.v1.worker.pp_utils import coordinate_cudagraph_mode_across_pp

    def run_rank(local_captured, peer_mode):
        mgr = _FakeCudaGraphManager(local_captured)
        mock_group, fake_ar = _patch_pp_group(
            world_size=2, peer_modes=[peer_mode.value]
        )
        with (
            mock.patch(
                "vllm.v1.worker.gpu.dp_utils.get_pp_group", return_value=mock_group
            ),
            mock.patch("vllm.v1.worker.pp_utils.get_pp_group", return_value=mock_group),
            mock.patch(
                "vllm.v1.worker.pp_utils.torch.distributed.all_reduce",
                side_effect=fake_ar,
            ),
        ):
            # Worker computes the agreed MIN before the recv from this rank's
            # local captured mode...
            pp_synced = coordinate_cudagraph_mode_across_pp(local_captured.value)
            # ...and the runner consumes it.
            desc, across_dp = dp_utils.dispatch_cg_and_sync_dp(
                mgr,
                num_reqs=4,
                num_tokens=128,
                uniform_token_count=None,
                dp_size=1,
                dp_rank=0,
                need_eager=False,
                pp_synced_cudagraph_mode=pp_synced,
            )
        assert across_dp is None  # pure PP: no DP padding tensor
        return desc.cg_mode

    # Stage A dispatched PIECEWISE; stage B was forced eager (NONE).
    a = run_rank(CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE)
    b = run_rank(CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE)
    assert a == b == CUDAGraphMode.NONE, (
        f"pure-PP stages diverged through dispatch_cg_and_sync_dp: A={a}, B={b}"
    )


def test_v2_dispatch_cg_and_sync_dp_pp_consensus_off_by_default():
    """Default (dummy/profile/capture path): pp_synced_cudagraph_mode is None, so
    NO PP reconciliation happens — the guard that keeps the
    per-stage-asynchronous capture path from deadlocking at startup (#45094/#45610
    follow-up). The descriptor is returned unreconciled."""
    from vllm.v1.worker.gpu import dp_utils

    mgr = _FakeCudaGraphManager(CUDAGraphMode.PIECEWISE)
    mock_group, _ = _patch_pp_group(
        world_size=2, peer_modes=[CUDAGraphMode.NONE.value]
    )
    with mock.patch(
        "vllm.v1.worker.gpu.dp_utils.get_pp_group", return_value=mock_group
    ):
        desc, _ = dp_utils.dispatch_cg_and_sync_dp(
            mgr,
            num_reqs=4,
            num_tokens=128,
            uniform_token_count=None,
            dp_size=1,
            dp_rank=0,
            need_eager=False,
            # default: pp_synced_cudagraph_mode=None -> no reconciliation
        )

    assert desc.cg_mode == CUDAGraphMode.PIECEWISE  # unreconciled, as on capture


# ---------------------------------------------------------------------------
# Source-level guards: tie the V2 regression to the ACTUAL call-site gating so
# a future revert fails CI here even if the unit helpers still pass. Mirrors the
# V1 test's source guards. Asserts (a) dispatch_cg_and_sync_dp accepts the gate
# flag defaulting False, and (b) the model_runner execute_model call site only
# requests PP coordination on the real (non-dummy, non-profile) path.
# ---------------------------------------------------------------------------


def _dp_utils_source() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root / "vllm" / "v1" / "worker" / "gpu" / "dp_utils.py"
    ).read_text()


def _v2_model_runner_source() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"
    ).read_text()


def _gpu_worker_source() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / "vllm" / "v1" / "worker" / "gpu_worker.py").read_text()


def test_v2_source_dispatch_consumes_presynced_mode():
    src = _dp_utils_source()
    assert "pp_synced_cudagraph_mode: int | None = None" in src, (
        "dispatch_cg_and_sync_dp must accept pp_synced_cudagraph_mode (a "
        "pre-agreed mode value computed by the worker before the recv), "
        "defaulting to None (no reconciliation on the dummy/capture path)."
    )
    # dp_utils must NOT issue the all-reduce itself anymore — the worker does it
    # ahead of the recv (#45610). Only doc-comment references are allowed.
    code_without_docstrings = "\n".join(
        line for line in src.splitlines() if "``" not in line
    )
    assert "coordinate_cudagraph_mode_across_pp(" not in code_without_docstrings, (
        "dp_utils must NOT call coordinate_cudagraph_mode_across_pp; the PP "
        "all-reduce moved to Worker.execute_model, ahead of the inter-stage "
        "recv (#45610)."
    )


def test_v2_source_execute_model_consumes_presynced_mode():
    src = _v2_model_runner_source()
    assert "pp_synced_cudagraph_mode=pp_synced_cudagraph_mode" in src, (
        "V2 execute_model must thread the worker-agreed pp_synced_cudagraph_mode "
        "into dispatch_cg_and_sync_dp (the consensus all-reduce is issued by "
        "Worker.execute_model before the recv; the runner only consumes it)."
    )
    assert "def predispatch_cudagraph_mode(" in src, (
        "V2 model runner must expose predispatch_cudagraph_mode so the worker "
        "can compute this rank's local mode before the recv (#45610)."
    )


def test_v2_worker_hoists_pp_consensus_before_recv():
    """The shared Worker.execute_model (drives both V1 and V2 runners) must issue
    the PP consensus all-reduce before the inter-stage recv, and the Ray
    compiled-graph path (which bypasses Worker.execute_model) must also reach
    consensus before consuming intermediate tensors (#45610).

    HARDENED: both ordering checks resolve the actual *call* line via the AST
    (inside the relevant function), not a whole-file ``str.find`` that would
    match the module-level import at the top and pass even if the call were
    misplaced below the recv."""
    import ast

    def _call_lineno(src: str, func_name: str, callee_name: str) -> int | None:
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
        best = None
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", None))
                if name == callee_name and (best is None or node.lineno < best):
                    best = node.lineno
        return best

    wsrc = _gpu_worker_source()
    c_ln = _call_lineno(wsrc, "execute_model", "coordinate_cudagraph_mode_across_pp")
    r_ln = _call_lineno(wsrc, "execute_model", "irecv_tensor_dict")
    assert c_ln is not None and r_ln is not None and c_ln < r_ln, (
        "Worker.execute_model must CALL coordinate_cudagraph_mode_across_pp "
        f"(line {c_ln}) BEFORE irecv_tensor_dict (line {r_ln}) — drives the V2 "
        "runner too; #45610. (call-site order, not import position)."
    )

    repo_root = Path(__file__).resolve().parents[3]
    ray_src = (
        repo_root / "vllm" / "v1" / "executor" / "ray_utils.py"
    ).read_text()
    # The Ray path must (a) actually CALL the consensus inside execute_model_ray,
    # and (b) call it BEFORE it consumes intermediate_tensors via the runner's
    # execute_model. A bare ``in ray_src`` substring would be satisfied by the
    # import alone — harden to call-site ordering.
    ray_consensus_ln = _call_lineno(
        ray_src, "execute_model_ray", "coordinate_cudagraph_mode_across_pp"
    )
    assert ray_consensus_ln is not None, (
        "The Ray compiled-graph executor (execute_model_ray) bypasses "
        "Worker.execute_model, so it must CALL coordinate_cudagraph_mode_across_pp "
        "itself before running the model, or PP+cudagraph wedges under Ray "
        "(#45610). No call found in the function body (an import alone is "
        "insufficient)."
    )
    # The runner execute_model call on the Ray path must come AFTER the consensus.
    ray_run_ln = _call_lineno(ray_src, "execute_model_ray", "execute_model")
    # (execute_model matches the runner call; consensus must precede it.)
    assert ray_run_ln is not None and ray_consensus_ln < ray_run_ln, (
        "On the Ray path the PP consensus "
        f"(line {ray_consensus_ln}) must run before the model is executed "
        f"(line {ray_run_ln}); otherwise the PP stages can split-brain on the "
        "DAG's P2P edge (#45610)."
    )
    # And the Ray path must thread the agreed value into the runner.
    assert "pp_synced_cudagraph_mode=pp_synced_cudagraph_mode" in ray_src, (
        "execute_model_ray must thread the agreed pp_synced_cudagraph_mode into "
        "the runner's execute_model."
    )


def test_v2_dp_and_pp_reconcile_composes_via_dp_min_fold():
    """Work-item 2b + MED-2 regression (V2): DP>1 AND PP>1.

    Models the (DP replica x PP stage) grid where the PP-agreed mode is folded
    INTO the DP all-reduce (the MED-2 fix). Pre-fix the PP drop ran *after* the
    DP collective over an orthogonal MIN slice, so two replicas of the same PP
    stage could end on different modes (one graph, one NONE) — a DP-lockstep
    split that re-wedges the pipeline. Post-fix the DP MIN subsumes the PP MIN
    globally, so every replica of a stage ends identical.

    We drive ``sync_cudagraph_and_dp_padding`` for each DP rank of stage s with a
    DP all-reduce simulated over the *other* DP rank's CONTRIBUTED value, and
    feed each rank its replica's pp_synced. We assert: with the fold, both DP
    replicas of the stage agree; and the FAIL-PRE check shows that without the
    fold (contributing the bare local mode) they would diverge."""
    from vllm.v1.worker.gpu import dp_utils

    # Grid (2 replicas x 2 stages); we examine stage s0 across both replicas.
    #   r0 local @ s0 = PIECEWISE ; r0 pp_synced (MIN over r0's stages) = PIECEWISE
    #   r1 local @ s0 = PIECEWISE ; r1 pp_synced (MIN over r1's stages) = NONE
    # Pre-fix DP MIN over {PIECEWISE, PIECEWISE} = PIECEWISE; r1 then drops to
    # NONE post-collective -> stage-0 split (PIECEWISE vs NONE).
    # Post-fix contributions are min(local, pp_synced): r0->PIECEWISE, r1->NONE;
    # DP MIN = NONE -> both replicas agree on NONE.
    def run_stage_rank(local_mode, pp_synced, peer_contributed, dp_rank, *, fold):
        mgr = _FakeCudaGraphManager(local_mode)
        # Simulate the DP all-reduce: each rank contributes either the folded
        # value (post-fix, done inside sync_cudagraph_and_dp_padding) or the bare
        # local value (pre-fix). We emulate the SUM all_reduce of the 3xN tensor
        # by injecting the peer's contribution at its dp_rank slot.
        peer_rank = 1 - dp_rank

        def fake_dp_all_reduce(tensor, group=None):
            # tensor rows: [num_tokens, cg_mode, uniform]; fill the peer slot.
            tensor[0][peer_rank] = 128
            tensor[1][peer_rank] = peer_contributed
            tensor[2][peer_rank] = 0

        mock_pp, _ = _patch_pp_group(world_size=2)
        with (
            mock.patch(
                "vllm.v1.worker.gpu.dp_utils.get_dp_group",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "vllm.v1.worker.gpu.dp_utils.get_pp_group", return_value=mock_pp
            ),
            mock.patch(
                "vllm.v1.worker.gpu.dp_utils.dist.all_reduce",
                side_effect=fake_dp_all_reduce,
            ),
        ):
            desc, _ = dp_utils.dispatch_cg_and_sync_dp(
                mgr,
                num_reqs=4,
                num_tokens=128,
                uniform_token_count=None,
                dp_size=2,
                dp_rank=dp_rank,
                need_eager=False,
                pp_synced_cudagraph_mode=(pp_synced if fold else None),
            )
        return desc.cg_mode

    # --- POST-FIX (fold ON): contributions are min(local, pp_synced). ---
    # r0 contributes min(PIECEWISE, PIECEWISE)=PIECEWISE; r1 min(PIECEWISE,NONE)=NONE.
    r0 = run_stage_rank(
        local_mode=CUDAGraphMode.PIECEWISE,
        pp_synced=CUDAGraphMode.PIECEWISE.value,
        peer_contributed=CUDAGraphMode.NONE.value,  # r1's folded contribution
        dp_rank=0,
        fold=True,
    )
    r1 = run_stage_rank(
        local_mode=CUDAGraphMode.PIECEWISE,
        pp_synced=CUDAGraphMode.NONE.value,
        peer_contributed=CUDAGraphMode.PIECEWISE.value,  # r0's folded contribution
        dp_rank=1,
        fold=True,
    )
    assert r0 == r1 == CUDAGraphMode.NONE, (
        f"DP>1 & PP>1 must compose: both replicas of stage 0 must agree post-"
        f"fold, got r0={r0}, r1={r1}. The PP MIN (NONE in r1) must be folded "
        "into the DP MIN so neither replica is left graph-mode (#45094 MED-2)."
    )


def test_v2_med2_pre_fix_ordering_would_split_dp_replicas():
    """FAIL-PRE companion to the test above: demonstrate that taking the DP MIN
    over the BARE local modes and applying the PP drop only AFTER the collective
    (the pre-fix ordering) diverges the two DP replicas of a stage. This pins WHY
    the fold is required; it asserts the divergence EXISTS in the pre-fix model
    (computed here directly, not by calling the fixed code)."""
    # Pre-fix: DP MIN over bare local modes {PIECEWISE(r0,s0), PIECEWISE(r1,s0)}.
    dp_min_s0 = min(CUDAGraphMode.PIECEWISE.value, CUDAGraphMode.PIECEWISE.value)
    # Then each replica applies its own PP drop AFTER the collective:
    pp_synced_r0 = CUDAGraphMode.PIECEWISE.value
    pp_synced_r1 = CUDAGraphMode.NONE.value

    def post_collective_drop(dp_synced, pp_synced):
        # _reconcile_cudagraph_mode_across_pp: drop to NONE iff pp_synced != mode.
        return (
            CUDAGraphMode.NONE.value if pp_synced != dp_synced else dp_synced
        )

    r0_final = post_collective_drop(dp_min_s0, pp_synced_r0)  # stays PIECEWISE
    r1_final = post_collective_drop(dp_min_s0, pp_synced_r1)  # drops to NONE
    assert r0_final != r1_final, (
        "Pre-fix ordering (DP MIN over bare locals, PP drop after) must split "
        "the DP replicas of stage 0; if it no longer does, this test no longer "
        "guards the #45094 MED-2 regression."
    )
    assert r0_final == CUDAGraphMode.PIECEWISE.value
    assert r1_final == CUDAGraphMode.NONE.value


def test_v2_source_sync_dp_folds_pp_synced_into_reduce():
    """Source guard (MED-2 fix): sync_cudagraph_and_dp_padding must FOLD the
    PP-agreed mode into the value it contributes to the DP all-reduce, i.e.
    accept ``pp_synced_cudagraph_mode`` and apply ``min(local, pp_synced)``
    BEFORE ``dist.all_reduce``. A future revert that drops the fold (and relies
    only on the post-reduce reconcile) re-introduces the DP-lockstep split."""
    import ast

    src = _dp_utils_source()
    assert "pp_synced_cudagraph_mode: int | None = None" in src
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "sync_cudagraph_and_dp_padding"
    )
    # Find the line of the min(...) fold and the line of dist.all_reduce; the
    # fold must precede the collective.
    min_ln = None
    ar_ln = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", None))
            if name == "min" and min_ln is None:
                min_ln = node.lineno
            if name == "all_reduce" and ar_ln is None:
                ar_ln = node.lineno
    assert min_ln is not None, (
        "sync_cudagraph_and_dp_padding must fold the PP-agreed mode via "
        "min(local, pp_synced) before contributing to the DP all-reduce (MED-2)."
    )
    assert ar_ln is not None, "expected a dist.all_reduce call."
    assert min_ln < ar_ln, (
        f"the min(local, pp_synced) fold (line {min_ln}) must execute BEFORE "
        f"the DP all_reduce (line {ar_ln}) so the DP MIN subsumes the PP MIN "
        "(MED-2)."
    )


# ---------------------------------------------------------------------------
# Layer A (ordering) for the V2 path. The V2 runner is driven by the SAME
# Worker.execute_model as V1 (gpu_worker.py dispatches to the V2 runner when
# self.use_v2_model_runner is True), so the all-reduce-before-recv ordering is
# enforced in one place. This test pins it with use_v2_model_runner=True so a
# V2-only regression of the worker ordering is caught here too.
# ---------------------------------------------------------------------------


def test_v2_pp_all_reduce_precedes_irecv_in_worker_execute_model():
    """Drive Worker.execute_model with a V2 runner stub on a stage-1 (not
    is_first_rank) PP rank, world_size=2, and assert the PP consensus all-reduce
    is issued BEFORE irecv_tensor_dict. FAILS pre-fix (the worker did not issue
    the consensus at all; the V2 runner did, after the recv -> deadlock)."""
    from unittest import mock

    import vllm.v1.worker.gpu_worker as gw

    events: list[str] = []

    pp_group = mock.MagicMock()
    pp_group.world_size = 2
    pp_group.is_first_rank = False
    pp_group.is_last_rank = False
    pp_group.cpu_group = mock.MagicMock()

    def fake_irecv(*args, **kwargs):
        events.append("irecv_tensor_dict")
        return {"hidden_states": object()}, None, None

    pp_group.irecv_tensor_dict.side_effect = fake_irecv

    worker = mock.MagicMock()
    worker._pp_send_work = []
    worker.use_v2_model_runner = True  # V2 path
    worker.model_runner.is_pooling_model = False  # so None returns early
    worker.vllm_config.compilation_config.pass_config.enable_sp = False
    worker.vllm_config.parallel_config.pipeline_parallel_size = 2
    worker.model_runner.predispatch_cudagraph_mode.return_value = (
        CUDAGraphMode.PIECEWISE.value
    )
    worker.model_runner.execute_model.return_value = None  # NoneType -> early return

    scheduler_output = mock.MagicMock()
    scheduler_output.total_num_scheduled_tokens = 8

    with (
        mock.patch.object(gw, "get_pp_group", return_value=pp_group),
        mock.patch.object(gw, "get_tp_group", return_value=mock.MagicMock()),
        mock.patch.object(gw, "coordinate_cudagraph_mode_across_pp") as consensus,
    ):
        def record_consensus(v):
            events.append("pp_all_reduce")
            return v

        consensus.side_effect = record_consensus
        gw.Worker.execute_model(worker, scheduler_output)

    assert "pp_all_reduce" in events and "irecv_tensor_dict" in events, (
        f"both the consensus and the recv must run; got {events}"
    )
    assert events.index("pp_all_reduce") < events.index("irecv_tensor_dict"), (
        f"PP all-reduce must precede the inter-stage recv on the V2 path; got "
        f"{events}. Otherwise the pipeline deadlocks on the first decode step "
        "(#45094/#45610)."
    )
