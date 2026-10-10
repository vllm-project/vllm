# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLB-backed L2 token routing.

Run on a single device without a process group: ``static`` and ``dynamic``
replica policies make a local decision, so no EP collective is issued.  The
``lplb`` policy does issue one and is covered by the multi-rank CUDA-graph
probe instead.

The important property under test is the accuracy-preserving invariant: MLB
may only change *which physical replica* serves a token, never which logical
expert the router chose.
"""

import pytest
import torch

pytest.importorskip("moe_load_balancer")

from vllm.distributed.eplb.eplb_state import (  # noqa: E402
    EplbLayerState,
    compute_logical_maps,
)
from vllm.distributed.eplb.mlb_runtime import (  # noqa: E402
    MlbRoutingRuntime,
    reset_mlb_routing,
)

NUM_LAYERS = 4
NUM_LOGICAL = 32
NUM_PHYSICAL = 40  # 8 redundant replicas
EP_SIZE = 4
NUM_TOKENS = 64
TOPK = 4


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_mlb_routing()


def _placement() -> torch.Tensor:
    """A placement where the first 8 logical experts get a second replica."""
    rows = []
    for _ in range(NUM_LAYERS):
        row = list(range(NUM_LOGICAL)) + list(range(NUM_PHYSICAL - NUM_LOGICAL))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int64)


def _runtime(algorithm: str) -> MlbRoutingRuntime:
    phy2log = _placement()
    log2phy, replica_count = compute_logical_maps(phy2log, NUM_LOGICAL)
    rt = MlbRoutingRuntime(
        algorithm,
        ep_size=EP_SIZE,
        ep_rank=0,
        num_logical_experts=NUM_LOGICAL,
        num_physical_experts=NUM_PHYSICAL,
        physical_to_logical_map=phy2log,
    )
    rt.register_logical_maps(log2phy, replica_count)
    return rt


def _layer_state(rt: MlbRoutingRuntime, layer_id: int = 0) -> EplbLayerState:
    state = EplbLayerState()
    state.set_layer_state(
        layer_id,
        torch.zeros(NUM_LAYERS, NUM_PHYSICAL, dtype=torch.int64),
        rt._logical_to_physical_map,
        rt._logical_replica_count,
    )
    state.should_record_tensor = torch.tensor(1, dtype=torch.int32)
    return state


def test_every_replica_policy_resolves_ids_rather_than_going_silent():
    """A policy that returned nothing here was not falling back gracefully --
    it never ran at all, and the caller silently kept its built-in choice."""
    rt = _runtime("dynamic")
    state = _layer_state(rt)
    torch.manual_seed(0)
    logical = torch.randint(0, NUM_LOGICAL, (NUM_TOKENS, TOPK), dtype=torch.int64)
    shares, ids = rt.resolve_routing(logical, torch.rand(NUM_TOKENS, TOPK), state, None)
    # Every replica policy resolves its own ids now; none hands back a share
    # table for this method to apply.
    assert shares is None
    assert ids is not None and ids.shape == logical.shape


def test_ultraep_fast_refresh_degrades_gracefully_without_a_mover():
    """The refresh makes memory match the placement it solves by moving expert
    weight with ``EplbState``'s own staging buffer and P2P communicator. A
    caller that does not supply them has given the policy nothing to move
    weight with, so the fast-refresh path must stay off and leave placement and
    routing working on the slow rearrange cadence alone -- not fail, and not
    half-enable itself."""
    num_local_physical = NUM_PHYSICAL // EP_SIZE
    expert_weights = [
        [torch.zeros(num_local_physical, 8, 8), torch.zeros(num_local_physical, 8, 8)]
        for _ in range(NUM_LAYERS)
    ]

    phy2log = _placement()
    rt = MlbRoutingRuntime(
        "ultraep",
        ep_size=EP_SIZE,
        ep_rank=0,
        num_logical_experts=NUM_LOGICAL,
        num_physical_experts=NUM_PHYSICAL,
        physical_to_logical_map=phy2log,
        expert_weights=expert_weights,
        # expert_buffer / communicator deliberately omitted.
    )
    assert rt._ultraep_expert_weights is None
    assert rt._ultraep_communicator is None
    assert rt._ultraep_refresh_gate is None

    # Routing must still work normally -- the fast-refresh path is purely
    # additive, so its absence must be silent, not a degraded dispatch.
    log2phy, replica_count = compute_logical_maps(phy2log, NUM_LOGICAL)
    rt.register_logical_maps(log2phy, replica_count)
    state = _layer_state(rt)
    torch.manual_seed(0)
    logical = torch.randint(0, NUM_LOGICAL, (NUM_TOKENS, TOPK), dtype=torch.int64)
    shares, ids = rt.resolve_routing(logical, torch.rand(NUM_TOKENS, TOPK), state, None)
    assert shares is None
    assert ids is not None and ids.shape == logical.shape


def test_ultraep_fast_refresh_skips_decode_batches(monkeypatch):
    """UltraEP is a prefill-time algorithm (the paper's own scoping; SGLang's
    reference should_refresh gates the same way, rejecting only its provable
    "decode" stage). A decode batch must return before touching the refresh
    gate's counter or moving any weight."""
    import vllm.distributed.eplb.mlb_runtime as mlb_runtime

    num_local_physical = NUM_PHYSICAL // EP_SIZE
    expert_weights = [
        [torch.zeros(num_local_physical, 8, 8), torch.zeros(num_local_physical, 8, 8)]
        for _ in range(NUM_LAYERS)
    ]
    phy2log = _placement()
    rt = MlbRoutingRuntime(
        "ultraep",
        ep_size=EP_SIZE,
        ep_rank=0,
        num_logical_experts=NUM_LOGICAL,
        num_physical_experts=NUM_PHYSICAL,
        physical_to_logical_map=phy2log,
        expert_weights=expert_weights,
    )

    # Force the gate open -- pre-existing quota buffers, so a non-decode call
    # would proceed into real collective and weight-move work. A decode-stage
    # call must never reach it: the stub raises if it does.
    def _explode(*args, **kwargs):
        raise AssertionError("must not move weight for a decode-stage batch")

    monkeypatch.setattr(rt, "_commit_placement_weights", _explode)
    rt._ultraep_rank_quota_prefix = torch.zeros(NUM_LAYERS, NUM_LOGICAL, EP_SIZE)
    monkeypatch.setattr(mlb_runtime, "_current_stage", lambda: "decode")

    topk_ids = torch.randint(0, NUM_LOGICAL, (NUM_TOKENS, TOPK), dtype=torch.int64)
    rt._ultraep_fast_refresh(0, topk_ids, num_unpadded_tokens=None)


def test_current_stage_under_dp_ignores_this_ranks_own_decode_shape(monkeypatch):
    """Everything the refresh does past its stage check issues a collective, so
    the check has to reach the same verdict on every EP rank. Under DP it does
    not: a rank with no requests runs `_dummy_run(uniform_decode=True)` and
    looks like a textbook decode to itself while its peers run real prefill.
    Answering "decode" off this rank's own descriptor is what splits the group,
    so the DP-reduced flag wins and a locally-decode-shaped batch still reports
    "mixed" when its peers are not decoding."""
    import vllm.distributed.eplb.mlb_runtime as mlb_runtime
    from vllm.forward_context import BatchDescriptor

    class _Ctx:
        # Shaped exactly like a decode to this rank, and vLLM's cross-rank
        # reduction says not every rank agrees.
        batch_descriptor = BatchDescriptor(num_tokens=4, num_reqs=4, uniform=True)
        uniform_decode_across_dp = False
        dp_metadata = object()

    monkeypatch.setattr(
        mlb_runtime, "is_forward_context_available", lambda: True, raising=False
    )
    monkeypatch.setattr(
        "vllm.forward_context.is_forward_context_available", lambda: True
    )
    monkeypatch.setattr("vllm.forward_context.get_forward_context", lambda: _Ctx())
    assert mlb_runtime._current_stage() == "mixed"

    # And when every rank really is decoding, all of them say so together.
    _Ctx.uniform_decode_across_dp = True
    assert mlb_runtime._current_stage() == "decode"


def test_dp_token_counts_reads_the_tensor_that_actually_exists():
    """The cross-rank token count is the input that makes every rank agree on
    whether a batch is worth refreshing for, so reading it has to actually
    return it.

    It lives on `DPMetadata.num_tokens_across_dp_cpu`. `set_forward_context`
    accepts an argument called `num_tokens_across_dp`, but does not keep a field
    by that name -- it is consumed into DPMetadata. Asking the context for it
    directly returned None on every rank, which silently routed this check to
    its single-rank fallback everywhere, i.e. produced precisely the per-rank
    disagreement it exists to prevent, while looking like a working fix. The
    assertion that matters here is simply that it is not None."""
    from vllm.distributed.eplb.mlb_runtime import _dp_token_counts
    from vllm.forward_context import DPMetadata

    counts = torch.tensor([4096, 0, 0, 0], dtype=torch.int32)

    class _Ctx:
        dp_metadata = DPMetadata(num_tokens_across_dp_cpu=counts)

    import vllm.forward_context as fc

    saved_avail, saved_get = fc.is_forward_context_available, fc.get_forward_context
    try:
        fc.is_forward_context_available = lambda: True
        fc.get_forward_context = lambda: _Ctx()
        got = _dp_token_counts()
        assert got is not None, (
            "_dp_token_counts returned None with DP metadata present -- the "
            "cross-rank check has silently degraded to a per-rank one"
        )
        assert torch.equal(got, counts)
        assert int(got.sum()) == 4096

        # No DP: nothing to read, and the caller's local fallback is correct
        # there because no peer exists to disagree with it.
        class _NoDp:
            dp_metadata = None

        fc.get_forward_context = lambda: _NoDp()
        assert _dp_token_counts() is None
    finally:
        fc.is_forward_context_available = saved_avail
        fc.get_forward_context = saved_get


def test_ultraep_fast_refresh_arms_for_composed_expressions():
    """`VLLM_MLB_L2_ALGORITHM` carries a whole pipeline expression, not a bare
    policy name. UltraEP owns placement in every expression it appears as the
    placement policy of -- `ultraep` and `ultraep+waterfill` alike -- and the
    refresh has to arm for all of them.

    It used to be selected by `algorithm == "ultraep"`, so any composition
    silently disarmed it, and the failure was not graceful: `_snapshot` still
    saw an allocated quota tensor and routed through placement tables no
    refresh had ever written."""
    from vllm.distributed.eplb.mlb_runtime import _algorithm_places_with_ultraep

    assert _algorithm_places_with_ultraep("ultraep")
    assert _algorithm_places_with_ultraep("ultraep+waterfill")
    assert not _algorithm_places_with_ultraep("lplb")
    assert not _algorithm_places_with_ultraep("static")
    assert not _algorithm_places_with_ultraep("waterfill")
    # An unparsable expression is not UltraEP's to claim, and must not raise here.
    assert not _algorithm_places_with_ultraep("not-a-policy")


def test_snapshot_ignores_uncommitted_ultraep_tables():
    """Allocating the quota tensor and filling a layer's slice are different
    events: the bootstrap solve does the first for the whole model, a refresh
    does the second one layer at a time. Between them the MLB-side tables
    describe a layout the framework never applied, so routing through them
    sends tokens to slots holding other experts. Until a layer is committed,
    the snapshot must hand back the framework's own candidates instead."""
    num_local_physical = NUM_PHYSICAL // EP_SIZE
    expert_weights = [
        [torch.zeros(num_local_physical, 8, 8), torch.zeros(num_local_physical, 8, 8)]
        for _ in range(NUM_LAYERS)
    ]
    phy2log = _placement()
    rt = MlbRoutingRuntime(
        "ultraep",
        ep_size=EP_SIZE,
        ep_rank=0,
        num_logical_experts=NUM_LOGICAL,
        num_physical_experts=NUM_PHYSICAL,
        physical_to_logical_map=phy2log,
        expert_weights=expert_weights,
    )
    log2phy, replica_count = compute_logical_maps(phy2log, NUM_LOGICAL)
    rt.register_logical_maps(log2phy, replica_count)
    state = _layer_state(rt)

    # Quota allocated whole, as the bootstrap solve leaves it, and deliberately
    # inconsistent with the committed placement -- exactly the state that made
    # the real failure route 90% of tokens to the wrong expert.
    rt._ultraep_rank_quota_prefix = torch.zeros(NUM_LAYERS, NUM_LOGICAL, EP_SIZE)
    rt._ultraep_logical_to_physical = torch.zeros(
        NUM_LAYERS, NUM_LOGICAL, EP_SIZE, dtype=torch.int64
    )
    rt._ultraep_replica_counts = torch.ones(NUM_LAYERS, NUM_LOGICAL, dtype=torch.int64)
    assert rt._ultraep_committed_layers == set()

    snap = rt._snapshot(state, 0)
    # Uncommitted: the framework's candidates, which always agree with
    # physical_to_logical_map.
    assert torch.equal(
        snap.logical_to_physical_candidates,
        state.logical_to_physical_map[:, : rt._max_replicas],
    )

    # Committed: MLB's own tables win, which is the whole point of the branch.
    rt._ultraep_committed_layers.add(0)
    snap = rt._snapshot(state, 0)
    assert torch.equal(
        snap.logical_to_physical_candidates, rt._ultraep_logical_to_physical[0]
    )


def test_synthesized_default_prefers_local_replicas():
    """VLLM keeps no per-rank dispatch table, so the adapter builds one.
    It must prefer a replica this rank owns -- that is the locality SGLang
    gets for free by collapsing its candidate list."""
    from moe_load_balancer.adapters.vllm import nearest_replica_table

    phy2log = _placement()
    log2phy, counts = compute_logical_maps(phy2log, NUM_LOGICAL)
    slots = NUM_PHYSICAL // EP_SIZE

    for ep_rank in range(EP_SIZE):
        table = nearest_replica_table(
            log2phy[0], counts[0], ep_rank=ep_rank, num_local_physical_experts=slots
        )
        assert table.shape == (NUM_LOGICAL,)
        # Chosen slot must really hold that logical expert.
        assert torch.equal(phy2log[0][table], torch.arange(NUM_LOGICAL))
        low, high = ep_rank * slots, (ep_rank + 1) * slots
        for logical in range(NUM_LOGICAL):
            valid = log2phy[0, logical, : counts[0, logical]]
            local = valid[(valid >= low) & (valid < high)]
            if local.numel() > 0:
                assert int(table[logical]) in local.tolist(), (
                    f"expert {logical} has a replica on rank {ep_rank} "
                    "but the default table points elsewhere"
                )


def test_waterfill_is_rejected_unless_the_shared_expert_is_dispatched(monkeypatch):
    """A shared-expert decision needs somewhere to be written back, and on this
    runtime that place exists only when shared-expert fusion is on: without it
    the shared expert is a per-rank replicated MLP that never enters dispatch,
    so there is no destination rank to choose.

    The guard used to live in the id converter, which raised on any decision
    carrying a shared_expert_rank. It now lives at configuration time instead,
    which is the better place -- a pipeline that cannot work is refused before
    the server starts rather than part-way through a forward -- but the
    property being guarded is the same one: never accept the decision and then
    drop it.
    """
    from vllm.distributed.eplb.mlb_runtime import l2_inapplicable_reason

    monkeypatch.setenv("VLLM_FUSE_SHARED_EXPERTS", "0")
    import importlib

    import vllm.envs

    importlib.reload(vllm.envs)
    for expr in ("waterfill", "ultraep+waterfill"):
        reason = l2_inapplicable_reason(expr, num_redundant_experts=16)
        assert reason is not None, f"{expr} must be refused without fusion"
        assert "shared expert" in reason

    # ultraep alone routes nothing shared and is unaffected either way.
    assert l2_inapplicable_reason("ultraep", num_redundant_experts=16) is None

    monkeypatch.setenv("VLLM_FUSE_SHARED_EXPERTS", "1")
    importlib.reload(vllm.envs)
    for expr in ("waterfill", "ultraep+waterfill"):
        assert l2_inapplicable_reason(expr, num_redundant_experts=16) is None


def test_l2_applicability_is_decided_on_the_config_the_run_uses(monkeypatch):
    """Only ParallelConfig may disable L2, because only it is certainly used.

    EPLBConfig defaults l2_algorithm from the environment, so the instance
    EngineArgs builds from the field default carries an algorithm while its
    redundancy is still zero. Deciding applicability there announced a disable
    that never applied to the run -- once per engine core, in the same log that
    the workers were filling with "L2 routing enabled". The decision belongs on
    the object that ends up in the running config.
    """
    from vllm.config.parallel import EPLBConfig, ParallelConfig

    monkeypatch.setenv("VLLM_MLB_L2_ALGORITHM", "ultraep")

    # The default instance takes the algorithm from the environment and leaves
    # it alone: its zero redundancy is a default, not a deployment.
    assert EPLBConfig().l2_algorithm == "ultraep"

    # The assembled config of a run that enables EPLB decides. Redundancy
    # stays zero here, so ultraep -- which only has replicas to choose between
    # when there are redundant experts -- is switched off.
    assembled = ParallelConfig(
        data_parallel_size=2,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config=EPLBConfig(l2_algorithm="ultraep"),
    )
    assert assembled.eplb_config.l2_algorithm == ""

    # A config that does not enable EPLB has no routing boundary for L2 to sit
    # in and is left alone, silently: this is also what a throwaway instance
    # built from the field defaults looks like, and it must not announce a
    # disable that never applied to any run.
    idle = ParallelConfig(eplb_config=EPLBConfig(l2_algorithm="ultraep"))
    assert idle.eplb_config.l2_algorithm == "ultraep"


def test_placement_commit_refreshes_policy_state():
    """A committed rearrangement must be observable without re-registration:
    vLLM updates its maps in place, and the runtime holds those tensors."""
    rt = _runtime("dynamic")
    rt.on_placement_committed()  # must not raise
    # Simulate vLLM committing a new placement in place.
    rt.physical_to_logical_map[0, NUM_LOGICAL:] = torch.arange(
        NUM_PHYSICAL - NUM_LOGICAL, dtype=torch.int64
    ).flip(0)
    new_log2phy, new_counts = compute_logical_maps(
        rt.physical_to_logical_map, NUM_LOGICAL
    )
    rt._logical_to_physical_map.copy_(new_log2phy)
    rt._logical_replica_count.copy_(new_counts)
    rt.on_placement_committed()

    state = _layer_state(rt)
    torch.manual_seed(0)
    logical = torch.randint(0, NUM_LOGICAL, (NUM_TOKENS, TOPK), dtype=torch.int64)
    # The snapshot handed to MLB must reflect the committed maps, not the ones
    # captured at registration.
    snapshot = rt._snapshot(state, 0)
    assert torch.equal(
        snapshot.logical_to_physical_candidates, rt._logical_to_physical_map[0]
    )
    shares, ids = rt.resolve_routing(logical, torch.rand(NUM_TOKENS, TOPK), state, None)
    assert (shares is None) != (ids is None)


def test_capabilities_gate_the_work_the_framework_does():
    """MLB declares what the framework must prepare; Glue must honour it
    instead of doing every lifecycle step for every policy."""
    dyn = _runtime("dynamic")
    # dynamic keeps no placement-derived state and needs no dispatch table.
    assert dyn.requires_post_topk_routing
    assert not dyn.requires_placement_state
    assert not dyn.requires_rank_dispatch_map
    assert dyn._default_replicas is None, (
        "the rank dispatch table was built for a policy that never reads it"
    )

    st = _runtime("static")
    assert st.requires_rank_dispatch_map
    assert st._default_replicas is not None


def test_replica_routing_sees_every_replica_not_just_the_local_one():
    """A replica-routing policy must be shown the whole candidate list.

    Narrowing each expert to its local replica is what a rank-dispatch policy
    asks for; handing the same view to LPLB decides in advance the very thing
    LPLB is there to decide, and on a rank that holds a copy of everything it
    sees it removes the redundancy entirely, so the LP is skipped. It also
    keeps the returned shares indexable by vLLM's own map, since
    ``replica_shares`` passes on the probabilities without the candidates.
    """
    rt = _runtime("lplb")
    assert not rt.requires_rank_dispatch_map
    snapshot = rt._snapshot(_layer_state(rt), 0)
    assert torch.equal(
        snapshot.logical_to_physical_candidates, rt._logical_to_physical_map[0]
    ), "LPLB was handed a narrowed candidate list"
    assert torch.equal(snapshot.logical_to_physical_count, rt._logical_replica_count[0])
    # The first 8 logical experts are replicated by _placement(); the choice
    # between their copies is exactly what the policy has to solve.
    assert int(snapshot.logical_to_physical_count[0]) == 2

    # `static` is the policy the collapse exists for, and it still gets it.
    st = _runtime("static")
    assert st.requires_rank_dispatch_map
    st_snapshot = st._snapshot(_layer_state(st), 0)
    assert int(st_snapshot.logical_to_physical_count[0]) == 1


def test_ultraep_quota_and_candidates_reach_the_snapshot_only_when_set():
    """The per-layer quota table and candidate table UltraEP's L1 solve
    produces have to reach L2 routing through the snapshot *together*, or the
    quota's replica columns get read against vLLM's own candidates -- rebuilt
    independently from physical_to_logical_map, not guaranteed to share MLB's
    width or column ordering -- rather than the table MLB solved it against.
    """
    rt = _runtime("ultraep")
    state = _layer_state(rt)

    # Before any L1 solve has run, there is nothing to thread through, and
    # the snapshot falls back to vLLM's own candidates untouched.
    baseline = rt._snapshot(state, 0)
    assert baseline.metadata.get("rank_quota_prefix") is None
    assert torch.equal(
        baseline.logical_to_physical_candidates, rt._logical_to_physical_map[0]
    )

    # A refresh stashes the whole per-layer tensors and commits the layer it
    # wrote; _snapshot() must slice each by the layer actually being routed,
    # and must prefer MLB's own candidate table over vLLM's for those layers.
    #
    # Committing is a separate step from stashing on purpose: the bootstrap
    # solve allocates these tensors for the whole model before any refresh has
    # written a layer, and what it leaves in them describes a placement the
    # framework never applied. Routing through that sends tokens to slots
    # holding other experts, so an uncommitted layer falls back instead --
    # covered by test_snapshot_ignores_uncommitted_ultraep_tables.
    quota = torch.arange(NUM_LAYERS * 3 * 5, dtype=torch.int32).reshape(
        NUM_LAYERS, 3, 5
    )
    candidates = (
        torch.arange(NUM_LAYERS * 3 * 5, dtype=torch.int64).reshape(NUM_LAYERS, 3, 5)
        + 1000
    )
    counts = torch.full((NUM_LAYERS, 3), 2, dtype=torch.int64)
    rt._ultraep_rank_quota_prefix = quota
    rt._ultraep_logical_to_physical = candidates
    rt._ultraep_replica_counts = counts
    rt._ultraep_committed_layers = set(range(NUM_LAYERS))
    for layer_id in range(NUM_LAYERS):
        snapshot = rt._snapshot(state, layer_id)
        assert torch.equal(snapshot.metadata["rank_quota_prefix"], quota[layer_id])
        assert torch.equal(
            snapshot.logical_to_physical_candidates, candidates[layer_id]
        )
        assert torch.equal(snapshot.logical_to_physical_count, counts[layer_id])


def test_commit_is_skipped_for_policies_without_placement_state(monkeypatch):
    rt = _runtime("dynamic")
    calls = []
    monkeypatch.setattr(rt, "_commit_layers", lambda ids: calls.append(list(ids)))
    rt.on_placement_committed([0, 1])
    rt.announce_initial_placement()
    assert calls == [], "dynamic does not keep placement state; nothing to refresh"


def test_only_changed_layers_are_refreshed(monkeypatch):
    """Refreshing an untouched layer is pure stall: a shape-changing rebuild
    costs a JIT build plus warmup per layer."""
    rt = _runtime("static")  # any policy; we intercept the MLB call
    seen = []
    monkeypatch.setattr(rt, "requires_placement_state", True)
    monkeypatch.setattr(rt, "_commit_layers", lambda ids: seen.append(list(ids)))

    rt.on_placement_committed([1, 3])
    assert seen == [[1, 3]]

    seen.clear()
    rt.on_placement_committed([])
    assert seen == [], "an empty change set must not touch any layer"

    seen.clear()
    rt.on_placement_committed(None)  # unknown -> conservative full refresh
    assert seen == [list(range(NUM_LAYERS))]


def test_graphs_plus_rearranging_placement_state_is_refused(monkeypatch):
    """LPLB replaces its solver tensors when a committed placement changes
    their layout, and a captured CUDA graph cannot follow that -- replay reads
    freed memory, observed as Xid 43 on two devices. It is intermittent (it
    needs a rearrangement that actually changes the layout), so it has to be
    refused up front rather than left to surface in a long serving run."""
    import sys
    import types

    from vllm.distributed.eplb import mlb_runtime

    class _Mode:
        name = "FULL_AND_PIECEWISE"

    class _Compilation:
        cudagraph_mode = _Mode()

    class _Cfg:
        compilation_config = _Compilation()

    stub = types.ModuleType("vllm.config")
    stub.get_current_vllm_config = lambda: _Cfg()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.config", stub)

    with pytest.raises(ValueError, match="freed memory|Xid 43"):
        mlb_runtime._reject_graphs_with_rearranging_placement_state(rearranges=True)

    # Pinned placement is the supported combination and must stay allowed --
    # that is how the reported LPLB numbers were measured.
    mlb_runtime._reject_graphs_with_rearranging_placement_state(rearranges=False)

    _Mode.name = "NONE"  # eager is the other escape
    mlb_runtime._reject_graphs_with_rearranging_placement_state(rearranges=True)


def test_initial_placement_is_announced():
    """The contract is 'after initial weight loading AND after each commit';
    skipping the first half leaves the policy to build state lazily inside the
    first forward, from whatever placement that forward happens to see."""
    rt = _runtime("static")
    seen = []
    rt.requires_placement_state = True
    rt._commit_layers = lambda ids: seen.append(list(ids))
    rt.announce_initial_placement()
    assert seen == [list(range(NUM_LAYERS))]


def _runtime_with_redundancy(num_redundant: int) -> MlbRoutingRuntime:
    """A runtime whose only interesting property is whether experts repeat."""
    num_physical = NUM_LOGICAL + num_redundant
    phy2log = torch.stack(
        [torch.arange(num_physical) % NUM_LOGICAL for _ in range(NUM_LAYERS)]
    )
    return MlbRoutingRuntime(
        "lplb",
        ep_size=EP_SIZE,
        ep_rank=0,
        num_logical_experts=NUM_LOGICAL,
        num_physical_experts=num_physical,
        physical_to_logical_map=phy2log,
    )


@pytest.mark.parametrize("num_redundant, allocated", [(0, False), (8, True)])
def test_count_buffers_exist_only_where_a_policy_consumes_them(
    num_redundant, allocated, monkeypatch
):
    """An LP with nothing to split must not cost a collective every step.

    Without redundant experts every logical expert holds exactly one replica,
    so LPLB returns identity dispatch and never reads the count. These buffers
    are what make vLLM run a per-layer count kernel and one EP all-reduce per
    step to produce it -- work whose result is then discarded.
    """
    monkeypatch.delenv("MLB_KEEP_ZERO_REDUNDANCY_COUNTS", raising=False)
    rt = _runtime_with_redundancy(num_redundant)
    assert (rt._logical_count_local is not None) is allocated
    assert (rt._logical_count_global is not None) is allocated


def test_the_zero_redundancy_override_restores_the_buffers(monkeypatch):
    """The override exists so the skipped work has a measured size rather than
    only an argued one; a benchmark that cannot restore the old path cannot
    report what removing it bought."""
    monkeypatch.setenv("MLB_KEEP_ZERO_REDUNDANCY_COUNTS", "1")
    rt = _runtime_with_redundancy(0)
    assert rt._logical_count_local is not None
    assert rt._logical_count_global is not None


def test_finalize_step_counts_is_inert_without_buffers(monkeypatch):
    """It must not reach for the EP group when there is nothing to reduce.

    These tests run with no distributed init, so an unguarded get_ep_group()
    would raise -- which is exactly the assertion.
    """
    monkeypatch.delenv("MLB_KEEP_ZERO_REDUNDANCY_COUNTS", raising=False)
    _runtime_with_redundancy(0).finalize_step_counts()


def _sized_runtime(algorithm: str, num_redundant: int = 8) -> MlbRoutingRuntime:
    num_physical = NUM_LOGICAL + num_redundant
    phy2log = torch.stack(
        [torch.arange(num_physical) % NUM_LOGICAL for _ in range(NUM_LAYERS)]
    )
    return MlbRoutingRuntime(
        algorithm,
        ep_size=EP_SIZE,
        ep_rank=0,
        num_logical_experts=NUM_LOGICAL,
        num_physical_experts=num_physical,
        physical_to_logical_map=phy2log,
    )


@pytest.mark.parametrize(
    "algorithm, consumes",
    [("lplb", True), ("static", False), ("dynamic", False), ("ultraep", False)],
)
def test_only_policies_that_read_the_load_pay_for_gathering_it(algorithm, consumes):
    """The count machinery follows what the policy declares, not "has a
    post-TopK policy".

    A replica policy can route from placement alone. Keying the count on
    requires_post_topk_routing charged every such policy a per-layer kernel and
    one collective per step for a number it never reads.
    """
    rt = _sized_runtime(algorithm)
    assert rt.consumes_global_logical_count is consumes
    assert (rt._logical_count_local is not None) is consumes


def test_the_runtime_reports_what_the_framework_must_respect():
    """Concurrency and graph stability are declared, not inferred from LPLB.

    Both guards used to key off properties LPLB happens to have, which barred
    combinations that were never at risk of the fault being guarded against.
    """
    st = _sized_runtime("static")
    assert st.supports_concurrent_microbatches is True
    assert st.graph_stability == "stable"

    lp = _sized_runtime("lplb")
    assert lp.supports_concurrent_microbatches is False
    assert lp.graph_stability == "realloc_on_placement_change"

    # ultraep caches a per-layer state dict on every route_tokens() call, and
    # the quota tensor it caches is a fresh allocation each L1 replan rather
    # than an in-place update -- the same hazard LPLB's solver state has.
    ultraep_rt = _sized_runtime("ultraep")
    assert ultraep_rt.supports_concurrent_microbatches is False
    assert ultraep_rt.graph_stability == "realloc_on_placement_change"


def test_placement_and_routing_share_one_balancer():
    """An engine gets one balancer serving both layers.

    Two instances cannot see each other, which rules out any placement decision
    that depends on what routing observed.
    """
    from vllm.distributed.eplb.mlb_runtime import (
        get_mlb_integration,
        reset_mlb_routing,
    )

    reset_mlb_routing()
    try:
        integration = get_mlb_integration()
        integration.algorithm = "lplb"
        num_physical = NUM_LOGICAL + 8
        phy2log = torch.stack(
            [torch.arange(num_physical) % NUM_LOGICAL for _ in range(NUM_LAYERS)]
        )
        routing = integration.bind_routing(
            ep_size=EP_SIZE,
            ep_rank=0,
            num_logical_experts=NUM_LOGICAL,
            num_physical_experts=num_physical,
            physical_to_logical_map=phy2log,
        )
        assert integration.balancer() is routing._mlb
        assert get_mlb_integration() is integration
    finally:
        reset_mlb_routing()


def test_async_placement_commits_are_delivered_on_the_main_thread():
    """The async worker commits a layer's new map without telling the policy.

    Left undelivered, a policy holding placement-derived state keeps solving
    against the layout the layer used to have. The dispatch kernel clamps to the
    live replica count and reads the committed map, so the token still reaches a
    valid replica of the right expert -- what degrades silently is the split,
    which was computed to balance a placement that no longer exists.

    Delivery is queued rather than immediate because rebuilding that state does
    GPU work and the commit runs on a worker thread.
    """
    from vllm.distributed.eplb import eplb_state as st

    st._ASYNC_COMMITTED_LAYERS.clear()
    st._ASYNC_COMMITTED_LAYERS.update({2, 0})

    rt = _runtime("lplb")
    seen: list[list[int]] = []
    rt.on_placement_committed = lambda ids: seen.append(list(ids))
    rt.finalize_step_counts = lambda: None

    # Mirror what prepare_forward does with the queue.
    if st._ASYNC_COMMITTED_LAYERS and rt.requires_placement_state:
        pending = sorted(st._ASYNC_COMMITTED_LAYERS)
        st._ASYNC_COMMITTED_LAYERS.clear()
        rt.on_placement_committed(pending)

    assert seen == [[0, 2]]
    assert not st._ASYNC_COMMITTED_LAYERS


def test_a_policy_without_placement_state_needs_no_such_delivery():
    """Static routes from the committed map directly, so there is nothing
    cached that a placement change could invalidate."""
    rt = _runtime("static")
    assert rt.requires_placement_state is False


def test_the_nearest_replica_table_is_refreshed_on_every_commit():
    """This side's derived table has to follow the placement, whatever the
    policy keeps.

    static reads this table and holds nothing else, so gating the refresh on
    requires_placement_state left it routing by the placement the run started
    with. Once experts move, most defaults are no longer in their expert's
    candidate list, the share table falls back to column 0 for them, and the
    traffic those replicas exist to spread piles onto one.
    """
    for algorithm, keeps_state in (("static", False), ("lplb", True)):
        rt = _runtime(algorithm)
        assert rt.requires_placement_state is keeps_state
        calls: list[int] = []
        rt._rebuild_default_replicas = lambda c=calls: c.append(1)
        rt._commit_layers = lambda ids: None
        rt.on_placement_committed([0])
        assert calls, (
            f"{algorithm}: the nearest-replica table was not refreshed, so it "
            "still describes the placement before this commit"
        )


def test_bootstrap_solve_is_routed_against_not_just_stored(monkeypatch):
    """A committed placement's own tables must be usable by the very next forward.

    `_snapshot` hands the router MLB's candidate table, replica counts and quota
    only for layers in `_ultraep_committed_layers`. That set used to be filled in
    one place -- inside `_ultraep_fast_refresh` -- while the commit path filled
    the three tables for *every* layer and marked none of them. The router
    therefore ignored the solve it had just been handed and fell back to vLLM's
    own candidates until a fast refresh rewrote each layer individually, which at
    interval=64 is most of a run: measured balance sat at the no-balancer
    baseline for the first ~50 sampling windows of those cells.

    The tables come from the same plan whose physical_to_logical_map the caller
    commits, so they are consistent for every layer at that moment; there is
    nothing to wait for.
    """
    from vllm.distributed.eplb import mlb_runtime

    num_layers, num_logical, max_replicas = 4, NUM_LOGICAL, 3

    class _Plan:
        physical_to_logical_map = torch.zeros(
            (num_layers, NUM_PHYSICAL), dtype=torch.int32
        )
        logical_to_all_physical_map = torch.zeros(
            (num_layers, num_logical, max_replicas), dtype=torch.int32
        )
        logical_to_physical_count = torch.ones(
            (num_layers, num_logical), dtype=torch.int32
        )
        metadata = {
            "rank_quota_prefix": torch.ones(
                (num_layers, num_logical, max_replicas), dtype=torch.int32
            )
        }

    class _Routing:
        _ultraep_rank_quota_prefix = None
        _ultraep_logical_to_physical = None
        _ultraep_replica_counts = None
        _ultraep_committed_layers: set[int] = set()

    class _Balancer:
        def plan_placement(self, request):
            return _Plan()

    routing = _Routing()

    class _Integration:
        def __init__(self):
            self.routing = routing

        def balancer(self):
            return _Balancer()

    integration = _Integration()
    monkeypatch.setattr(mlb_runtime, "get_mlb_integration", lambda: integration)
    # plan_placement imports this inside the function body, so the name has to
    # be replaced where it is looked up rather than on mlb_runtime itself.
    monkeypatch.setattr(
        "moe_load_balancer.adapters.vllm.to_vllm_physical_to_logical",
        lambda plan: plan.physical_to_logical_map,
    )

    mlb_runtime.plan_placement(object())

    assert routing._ultraep_rank_quota_prefix is not None, (
        "the commit path must stash the quota"
    )
    assert routing._ultraep_committed_layers == set(range(num_layers)), (
        "every layer the commit path wrote tables for must be marked usable; "
        "leaving the set empty makes the router discard this solve until a fast "
        "refresh rewrites each layer, and how long that takes scales with the "
        "refresh interval"
    )
