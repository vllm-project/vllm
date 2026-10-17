# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-owned glue for MLB's L2 token-routing layer.

vLLM's built-in replica choice is a Knuth hash of the token index, fused into
the same Triton kernel that records expert load
(``fused_moe/router/base_router.py``).  When an MLB routing algorithm is
selected the kernel is kept: MLB supplies the per-expert replica shares it
solved for, and the kernel samples from them instead of hashing.  Recording,
the record gate and the padding mask therefore stay where vLLM put them, with
nothing reimplemented alongside.

Enable with ``VLLM_MLB_L2_ALGORITHM``, e.g. ``lplb``, ``dynamic``, ``static``.
Unset (the default) leaves vLLM's fused kernel in charge, so this module costs
nothing when it is not used.

Two things are deliberately *not* supported and fail loudly rather than
silently degrading:

* **Waterfill / shared-expert routing.**  On vLLM's CUDA path the shared expert
  is a per-rank replicated MLP that never enters EP dispatch, so there is no
  destination-rank decision to write back.
* **DBO (dual-batch overlap).**  MLB's ``RoutingRequest`` carries a single
  token count and its policies keep one solver state per layer, so two
  concurrently-executing micro-batches would clobber each other.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.eplb.eplb_state import EplbLayerState

logger = init_logger(__name__)

ALGORITHM_ENV = "VLLM_MLB_L2_ALGORITHM"

_runtime: MlbRoutingRuntime | None = None


def mlb_l2_algorithm() -> str:
    """Configured MLB routing expression, or "" when MLB routing is off."""
    return envs.VLLM_MLB_L2_ALGORITHM


class VllmRoutingCollectives:
    """Expose vLLM's EP process group through MLB's generic transport API.

    Validated to be CUDA-graph capturable: vLLM routes EP collectives through
    pynccl, and capture happens inside vLLM's own ``graph_capture()`` context.
    """

    @property
    def rank(self) -> int:
        from vllm.distributed.parallel_state import get_ep_group

        return get_ep_group().rank_in_group

    @property
    def world_size(self) -> int:
        from vllm.distributed.parallel_state import get_ep_group

        return get_ep_group().world_size

    def all_reduce_sum(self, payload: torch.Tensor) -> torch.Tensor:
        from vllm.distributed.parallel_state import get_ep_group

        return get_ep_group().all_reduce(payload)


def _algorithm_places_with_ultraep(algorithm: str) -> bool:
    """Whether this L2 expression puts UltraEP in charge of placement.

    ``RoutingPipeline`` reports that as ``placement_policy``, which is exactly
    the question the fast-refresh path needs answered: it owns the placement
    solve and the weight movement that follows it, and it must run for every
    expression UltraEP places for -- ``ultraep`` and ``ultraep+waterfill``
    alike. An unparsable expression is not UltraEP's to claim; the caller
    reports that separately.
    """
    from moe_load_balancer.core.routing_pipeline import RoutingPipeline

    try:
        return RoutingPipeline.from_value(algorithm).placement_policy == "ultraep"
    except Exception:
        return False


def _current_stage() -> str | None:
    """Map vLLM's current batch onto MLB's routing stage.

    SGLang hands MLB a discrete ForwardMode (EXTEND / DECODE / IDLE / ...).
    vLLM has no equivalent: with continuous batching and chunked prefill a
    single batch routinely mixes prefill and decode tokens, so a clean
    "prefill" stage does not exist.  Only decode is provable -- every request
    contributing exactly one token -- and everything else is reported as mixed
    rather than guessed at, so a stage-gated policy never runs on a stage it
    did not ask for.
    """
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return None
    ctx = get_forward_context()
    descriptor = getattr(ctx, "batch_descriptor", None)
    if descriptor is None:
        return None
    # `uniform_decode_across_dp` is vLLM's own cross-rank reduction of exactly
    # the predicate below (see `coordinate_batch_across_dp`): it is True only
    # when *every* DP rank is decoding this step. The descriptor describes this
    # rank alone, and under DP that is not the same question -- a rank with no
    # requests runs a dummy decode batch while its peers prefill, so reading the
    # descriptor makes a rank answer "decode" on a step its peers are treating
    # as prefill. Any caller gating a collective on the result would then split
    # the EP group. Prefer the reduced answer whenever the framework supplies
    # one; fall back to the local predicate only where there is no DP peer that
    # could disagree.
    if getattr(ctx, "uniform_decode_across_dp", False):
        return "decode"
    if getattr(ctx, "dp_metadata", None) is not None:
        # Under DP the reduced flag above is the only admissible answer. It is
        # False here, so at least one rank is not decoding and every rank has
        # to say so -- including this one, even if its own batch is a textbook
        # decode. Falling through to the descriptor would be the split this
        # function exists to avoid. Read off the forward context rather than
        # the config so the "is this DP" test itself comes from the same
        # per-step object every rank sees.
        return "mixed"
    if (
        descriptor.uniform
        and descriptor.num_reqs is not None
        and descriptor.num_reqs > 0
        and descriptor.num_tokens == descriptor.num_reqs
    ):
        return "decode"
    return "mixed"


def _dp_token_counts() -> torch.Tensor | None:
    """Cross-rank-synchronized per-DP-rank token count for this step, if any.

    vLLM computes this once per step -- real batch or dummy -- and publishes
    it to every rank before any MoE layer's forward runs
    (``set_forward_context(num_tokens_across_dp=...)``, both in
    ``execute_model`` and in ``_dummy_run``). Every rank sees the identical
    array regardless of whether it personally has real requests this step,
    which is exactly the property ``_ultraep_fast_refresh`` needs from its
    own "is this worth a refresh" check: reading this instead of a per-rank
    local token count means every rank reaches the same verdict even when
    some ranks are running a dummy batch and never see real ``topk_ids`` at
    all. ``None`` outside DP (single rank, or TP/PP-only) -- the caller falls
    back to a local check there, where no dummy-batch peer exists to
    disagree with.
    """
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    # On `dp_metadata`, not on the context itself. `set_forward_context` takes a
    # `num_tokens_across_dp` argument but does not keep it under that name --
    # it is consumed into DPMetadata, and ForwardContext has no such field. A
    # getattr for it therefore always answered None, which silently sent this
    # check down its single-rank fallback on every rank, which is exactly the
    # cross-rank disagreement it exists to prevent. Read the tensor where it
    # actually lives. It is a CPU tensor, so summing it costs no GPU sync.
    dp_metadata = getattr(get_forward_context(), "dp_metadata", None)
    if dp_metadata is None:
        return None
    return getattr(dp_metadata, "num_tokens_across_dp_cpu", None)


class MlbRoutingRuntime:
    """Holds the MoELoadBalancer instance and the placement geometry."""

    def __init__(
        self,
        algorithm: str,
        *,
        ep_size: int,
        ep_rank: int,
        num_logical_experts: int,
        num_physical_experts: int,
        physical_to_logical_map: torch.Tensor,
        balancer: Any | None = None,
        expert_weights: Any | None = None,
        expert_buffer: Any | None = None,
        communicator: Any | None = None,
    ) -> None:
        from moe_load_balancer import MoELoadBalancer

        self.algorithm = algorithm
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_logical_experts = num_logical_experts
        self.num_physical_experts = num_physical_experts
        # Model-level [layers, num_physical] map.  Kept here rather than on the
        # layer state so that MixtureOfExperts.set_eplb_state -- a public model
        # interface -- does not have to change.
        self.physical_to_logical_map = physical_to_logical_map
        # A logical expert can hold at most one slot plus every redundant one,
        # so this bounds the replica count without reading the map off device.
        self._max_replicas = num_physical_experts - num_logical_experts + 1
        self._default_replicas: torch.Tensor | None = None
        # Until a placement commits, assume there is a choice to make; the
        # first commit replaces this with what the placement actually offers.
        self._placement_offers_replica_choice: bool = True
        # The same table in the layout the fused kernel reads: one candidate
        # column per logical expert, and a replica count of 1 so the kernel's
        # `hash % count` lands on it. Built once per placement.
        self._fixed_map: torch.Tensor | None = None
        self._fixed_counts: torch.Tensor | None = None
        # [num_physical_experts] — slot i belongs to rank (i // experts_per_rank).
        # Passed to every PlacementSnapshot so LPLB can incorporate cross-GPU
        # transfer cost; SGLang always provides this, we build it from topology.
        from moe_load_balancer.adapters.vllm import build_physical_to_rank_map

        self._physical_to_rank_map = build_physical_to_rank_map(
            num_physical_experts,
            ep_size,
            device=physical_to_logical_map.device,
        )
        # Injected when an engine-scoped integration owns the balancer, so L1
        # and L2 are served by one instance rather than two that cannot see
        # each other.
        self._mlb: Any = (
            balancer
            if balancer is not None
            else MoELoadBalancer.from_algorithm(
                algorithm,
                ep_size=ep_size,
                source_rank=ep_rank,
                experts_per_rank=num_physical_experts // ep_size,
                collectives=VllmRoutingCollectives(),
            )
        )
        # MLB declares what the framework has to prepare for the selected
        # policy, so none of the work below is done unconditionally: `lplb`
        # needs placement state refreshed on commit, `static` needs a per-rank
        # dispatch table, and only a pipeline with a post-TopK policy wants the
        # routing boundary at all.
        caps = self._mlb.routing_capabilities
        self.requires_post_topk_routing = caps.requires_post_topk_routing
        self.requires_placement_state = caps.requires_placement_state
        self.requires_rank_dispatch_map = caps.requires_rank_dispatch_map
        # Whether a policy in this pipeline picks the shared expert's rank.
        # The router still has to have somewhere to put that answer -- it only
        # asks when its layer was built with the EP shared-expert layout
        # (VLLM_FUSE_SHARED_EXPERTS). getattr, because an older MLB has no
        # such field.
        self.routes_shared_expert = getattr(caps, "routes_shared_expert", False)
        # A policy whose answer depends only on the committed placement does
        # not need the per-forward call at all: bake the answer into a map the
        # fused kernel already knows how to read, and the boundary disappears.
        # getattr, because an older MLB has no such field.
        self.dispatch_fixed_by_placement = getattr(
            caps, "dispatch_fixed_by_placement", False
        )
        # These two cannot both hold. Routing a shared expert means choosing a
        # rank per batch from that batch's load; "fixed by placement" means the
        # answer is a function of the committed placement alone and the
        # per-forward call can be skipped. A pipeline claiming both gets its
        # shared-expert decisions computed and then thrown away, and the stale
        # ones from the previous layer read in their place -- with no symptom
        # beyond a quality regression. Checked here rather than trusted: the
        # default above protects against an MLB too old to have the field, but
        # not against one old enough to have it and set it wrong.
        if self.routes_shared_expert and self.dispatch_fixed_by_placement:
            raise ValueError(
                f"MLB routing pipeline {algorithm!r} reports both "
                "routes_shared_expert and dispatch_fixed_by_placement. A "
                "shared-expert rank is a per-batch decision, so it cannot be "
                "baked in at placement-commit time; one of the two is wrong. "
                "An MLB predating the fix to dispatch_fixed_by_placement "
                "reports this for any pipeline containing waterfill."
            )
        # Declared by the pipeline rather than inferred from "has a post-TopK
        # policy": a replica policy can route from placement alone, and
        # gathering the EP-wide load for it costs a collective per step that is
        # then discarded.
        self.consumes_global_logical_count = getattr(
            caps, "consumes_global_logical_count", caps.requires_post_topk_routing
        )
        self.max_input_staleness_steps = getattr(caps, "max_input_staleness_steps", 0)
        self.supports_concurrent_microbatches = getattr(
            caps, "supports_concurrent_microbatches", False
        )
        self.graph_stability = getattr(caps, "graph_stability", "stable")
        logger.info(
            "MLB L2 routing enabled (algorithm=%s, ep_size=%d, logical=%d, "
            "physical=%d, caps: post_topk=%s placement_state=%s "
            "rank_dispatch_map=%s)",
            algorithm,
            ep_size,
            num_logical_experts,
            num_physical_experts,
            self.requires_post_topk_routing,
            self.requires_placement_state,
            self.requires_rank_dispatch_map,
        )

        # Graph-safe global count cache for LPLB.
        #
        # LPLB's LP solve needs global_logical_count: how many tokens each
        # logical expert received across all EP ranks.  The naive path inside
        # MLB does count_logical_experts(topk_ids) + EP all_reduce per layer
        # per forward, which is 40 NCCL collectives per step -- and NCCL
        # cannot be captured in a CUDA graph.
        #
        # The fix: split "compute local count" (CUDA kernel, graph-safe) from
        # "all_reduce" (outside the graph, once per step).
        #
        # _logical_count_local[layer, expert]: local counts accumulated per-layer
        #   inside the graph by count_logical_experts; reset each step.
        # _logical_count_global[layer, expert]: all-reduced result of the previous
        #   step, stable GPU address, read by LP solve inside the graph.
        #
        # The LP solve therefore uses counts stale by one step, which is an
        # excellent approximation for decode (distribution barely changes) and
        # acceptable for prefill (first step uses zeros → hash routing, then
        # converges).
        num_moe_layers = physical_to_logical_map.shape[0]
        dev = physical_to_logical_map.device
        # Gated on redundancy as well as on the policy.  With no redundant
        # experts every logical has exactly one replica, the LP is degenerate,
        # and LPLB answers with identity dispatch without ever reading the
        # count -- so keeping these buffers would buy a per-layer count kernel
        # plus one collective per step for a result that is thrown away.
        # Leaving them None also makes MLB decline to gather the count itself
        # (needs_global_logical_count), so nothing downstream pays for it.
        # MLB_KEEP_ZERO_REDUNDANCY_COUNTS=1 restores the ungated behaviour so the
        # cost of that discarded work can be measured against this, rather than
        # only argued for.
        keep_degenerate = envs.MLB_KEEP_ZERO_REDUNDANCY_COUNTS
        if self.consumes_global_logical_count and (
            self._max_replicas > 1 or keep_degenerate
        ):
            self._logical_count_local: torch.Tensor | None = torch.zeros(
                num_moe_layers, num_logical_experts, dtype=torch.float32, device=dev
            )
            self._logical_count_global: torch.Tensor | None = torch.zeros(
                num_moe_layers, num_logical_experts, dtype=torch.float32, device=dev
            )
        else:
            self._logical_count_local = None
            self._logical_count_global = None
        # True once finalize_step_counts has run at least once (first step uses
        # zeros → hash routing fallback via selection is None).
        self._logical_count_ready = False
        # Reduced by the dummy path so the collective stays symmetric.
        self._count_scratch: torch.Tensor | None = None

        # MLB_FRESH_COUNTS=1 withholds the pre-computed count so MLB gathers it
        # itself, per layer per forward. That is what the SGLang adapter does and
        # what LPLB's published numbers were measured against; the count here is
        # one step stale so the collective can live outside the CUDA graph.
        # Staleness is a good approximation in decode, where the routing
        # distribution moves slowly, and a worse one in chunked prefill, where
        # consecutive chunks can differ sharply -- so a measured LPLB regression
        # cannot be attributed to the algorithm without checking this. Read once:
        # replica_shares runs per layer per forward.
        self._fresh_counts = envs.MLB_FRESH_COUNTS

        # Diagnostic capture of the LP's inputs and output. Off unless a
        # directory is named; bounded so a long run cannot fill the disk.
        self._dump_lp_dir = envs.MLB_DUMP_LP
        self._dump_lp_left = envs.MLB_DUMP_LP_N
        # Skip the opening calls: the first steps legitimately carry zeros
        # while the one-step-stale pipeline fills, and sampling only those
        # would mistake a warm-up transient for steady state.
        self._dump_lp_skip = envs.MLB_DUMP_LP_SKIP

        # MLB_TIME_L2=<n> times n route_tokens calls with a device sync on each
        # side. The sync makes the number meaningful and the measurement
        # invasive, so it is off unless asked for -- diagnostic only.
        # Set by replica_shares when a policy answers with ids; read by
        # resolve_routing in the same call.
        self._last_physical_ids: torch.Tensor | None = None
        # Same one-call handoff for a shared-expert decision. Separate from the
        # ids because the router consumes them at different points: the ids
        # feed the EPLB mapping kernel, the rank feeds the append that happens
        # after it.
        self._last_shared_expert_rank: torch.Tensor | None = None
        # Set by plan_placement() after an UltraEP L1 solve, read by
        # _snapshot() on every forward until the next solve replaces them.
        # None for every other L1 policy, and before the first solve.
        #
        # The candidate table and quota must come from the same solve: the
        # quota indexes replicas by column against MLB's own candidate
        # ordering, which is not guaranteed to be the same table -- same
        # width, even -- as vLLM's own logical_to_physical_map, independently
        # rebuilt from physical_to_logical_map through compute_logical_maps.
        # So route with MLB's own table for this policy, not vLLM's.
        self._ultraep_rank_quota_prefix: torch.Tensor | None = None
        self._ultraep_logical_to_physical: torch.Tensor | None = None
        self._ultraep_replica_counts: torch.Tensor | None = None
        self._time_l2 = envs.MLB_TIME_L2
        # MLB_TIME_REFRESH=<n> times n fast refreshes -- the placement solve and
        # the weight commit separately -- with a device sync on each, then
        # reports and disarms. Same shape as MLB_TIME_L2 above.
        self._time_refresh = envs.MLB_TIME_REFRESH
        # How UltraEP's fast refresh lands re-solved replica weight. Upstream
        # UltraEP does this with a one-sided weight_sync (masters stay put,
        # replica slots are refilled from the owning master); the movers below
        # are that idea on vLLM's own tensors, best first:
        #   symm    -- one-sided NVLink puts through torch symmetric memory,
        #              completed with pairwise signals (_commit_symm)
        #   direct  -- pairwise send/recv over the EPLB communicator, no
        #              staging buffer (_commit_direct)
        #   generic -- vLLM's periodic-rearrangement mover (move_to_buffer),
        #              which also handles the one case the others do not: the
        #              first refresh after a checkpointed placement, when the
        #              masters themselves move
        # MLB_ULTRAEP_MOVER (default direct). `auto` and `symm` try symmetric
        # memory first, but only after every EP rank has agreed it can (see
        # _symm_setup): a per-rank try/except around the collective rendezvous
        # let some ranks fall back while the others waited in it, and the
        # server deadlocked on its first refresh whenever the EP group spanned
        # hosts without a multi-node NVLink fabric. Measured on a Blackwell
        # multi-node NVLink EP16 deployment (DeepSeek-V3, one allocation, arms
        # back to back) symm is within 1-2%
        # of direct, so direct is the default and symm an explicit choice.
        # Every mover leaves a layer whose masters move to generic.
        self._ultraep_mover = envs.MLB_ULTRAEP_MOVER
        self._ultraep_mover_warned = False
        self._symm_hdl: Any = None
        self._symm_buf: torch.Tensor | None = None
        self._symm_layout: list[tuple[int, int, torch.dtype, tuple[int, ...]]] = []
        self._symm_expert_nbytes = 0
        # MLB_ULTRAEP_LAGGED_APPLY (default 0): with 1 the plan solved on this
        # forward lands (routing snapshot + weights) on the layer's next
        # forward, which keeps the plan's device->host copy off the critical
        # path -- no per-layer GPU sync -- at the price of one step of
        # staleness: the first batch of a burst runs on the old placement and
        # the second on a plan fitted to that first batch. Measured on
        # DeepSeek-V3 EP16 (same allocation, arms back to back): at interval 8
        # it changes nothing (Blackwell, 139.0k vs 139.0k tok/s, mean imbalance
        # 1.30 vs 1.29); at interval 1 it buys +3% (HybridEP) to +8% (DeepEP
        # normal) on Blackwell for +0.05 of steady imbalance, and on Hopper it
        # buys nothing and costs the same balance. Off by default; set 1 for
        # interval-1 deployments with Blackwell-class step times.
        self._ultraep_lagged = envs.MLB_ULTRAEP_LAGGED_APPLY
        self._ultraep_pending: dict[int, tuple[Any, ...]] = {}
        self._ultraep_pending_host: dict[int, torch.Tensor] = {}
        self._solve_us: list[float] = []
        self._commit_us: list[tuple[int, float]] = []
        self._commit_host_us: list[float] = []
        self._l2_us: list[float] = []

        # Real, traffic-driven placement refresh with real weight transfer,
        # additive to plan_placement()'s slow, wall-clock-cadence path above:
        # that path still owns this model's *initial* placement (the buffers
        # this refresh later writes per-layer slices into do not exist
        # before its first solve -- see the None-guard in
        # _ultraep_fast_refresh), this only adds much more frequent updates
        # on top of it. Making memory match a placement the solve decided is
        # the framework adapter's job and stays here: the solve produces an
        # explicit before/after map pair and hands it to EplbState's own P2P
        # mover (_commit_placement_weights). MLB supplies the placement
        # algorithm and nothing else -- no policy brings its own weight
        # mover, and no second component re-derives placement from raw
        # routing to agree with this one by construction.
        self._ultraep_expert_weights: Any = None
        self._ultraep_expert_buffer: Any = None
        self._ultraep_communicator: Any = None
        self._ultraep_num_local_physical: int | None = None
        # Layers whose MLB-side placement tables have actually been written by
        # a refresh. Allocation of the quota tensor is not the same event: the
        # bootstrap solve allocates it whole, layer contents arrive one refresh
        # at a time. See the guard in _snapshot.
        self._ultraep_committed_layers: set[int] = set()
        self._ultraep_refresh_gate: Any = None
        # Declared here (not just where it's actually populated, in
        # _init_ultraep_fast_refresh below) because finish_pending_
        # ultraep_transfer() is called unconditionally from vLLM's own FFN
        # forward for every MoE layer regardless of which L1 policy is
        # active -- an L1 policy other than ultraep must see this attribute
        # exist (as None) rather than raise AttributeError.
        # 512 is the reference inference integration's default (SGLang's
        # `moe_balance_refresh_min_tokens`). The solve reads only the batch in
        # front of it and the answer is then held for a whole refresh interval,
        # so re-solving on a batch too small to be representative fits the
        # placement to noise and keeps it there. It bites only where batches
        # are genuinely small: a prefill-shaped step is orders of magnitude
        # above either threshold.
        self._ultraep_min_representative_tokens = envs.MLB_ULTRAEP_REFRESH_MIN_TOKENS
        # Parsed, not compared as a string. `algorithm` is the whole L2
        # expression, so an equality test here silently misses every
        # composition that merely *contains* ultraep -- `ultraep+waterfill`
        # among them. The failure was not a graceful degradation: the refresh
        # never ran, while `_snapshot` still saw an allocated quota and routed
        # through it, so tokens went to slots holding other experts. Everywhere
        # else in this file already asks the parsed pipeline (see
        # l2_pipeline_capabilities / l2_inapplicable_reason); this was the one
        # place that did not.
        if _algorithm_places_with_ultraep(algorithm) and expert_weights is not None:
            self._init_ultraep_fast_refresh(expert_weights, expert_buffer, communicator)

    def _init_ultraep_fast_refresh(
        self,
        expert_weights: Any,
        expert_buffer: Any,
        communicator: Any,
    ) -> None:
        """One-time setup for the traffic-driven placement refresh.

        Holds on to the model's own expert weight tensors plus ``EplbState``'s
        staging buffer and P2P communicator -- the same three things vLLM's own
        periodic rearrangement moves weight with. This policy re-plans placement
        far more often than that cadence, but it re-plans it into the same
        memory using the same machinery; it brings no mover of its own.

        Weights are taken eagerly rather than on first use: vLLM hands this
        runtime ``model.expert_weights`` at the same call site that constructs
        it, so there is no "not loaded yet" window to defer past.

        Assumes the per-layer weight tensors are each shaped
        ``[num_local_physical_experts, ...]`` -- true for the unquantized case
        this was verified against. A model whose ``expert_weights`` also carries
        separate quantization-scale tensors would need those moved too; not
        attempted here.
        """
        if expert_buffer is None or communicator is None:
            logger.warning_once(
                "ultraep fast refresh needs EplbState's expert buffer and "
                "communicator to move weight; neither was supplied, so the "
                "policy stays on the slow EplbState.rearrange() cadence only."
            )
            return

        from moe_load_balancer.policies.l1 import RefreshGate

        self._ultraep_expert_weights = expert_weights
        self._ultraep_expert_buffer = expert_buffer
        self._ultraep_communicator = communicator
        self._ultraep_num_local_physical = self.num_physical_experts // self.ep_size

        interval = envs.MLB_ULTRAEP_REFRESH_INTERVAL
        self._ultraep_refresh_gate = RefreshGate(interval)
        logger.info(
            "UltraEP fast refresh enabled (interval=%d representative "
            "batches, min_tokens=%d, num_layers=%d, "
            "num_local_physical=%d, mover=%s, lagged_apply=%s)",
            interval,
            self._ultraep_min_representative_tokens,
            len(expert_weights),
            self._ultraep_num_local_physical,
            self._ultraep_mover,
            self._ultraep_lagged,
        )

    def finalize_step_counts(self) -> None:
        """All-reduce per-layer local counts and update the stable LP input buffer.

        Call at the START of each forward pass (from EplbState.prepare_forward).
        By the time this runs, the previous step's count_logical_experts results
        are already in _logical_count_local (written per-layer inside the graph or
        in the eager forward).

        One EP collective for all 40 layers combined replaces the 40 per-layer
        collectives that MLB's _global_logical_count would otherwise issue.
        Running outside the graph means NCCL is never captured -- graphs only
        see the LP solve kernels reading from _logical_count_global.
        """
        if self._logical_count_local is None or self._logical_count_global is None:
            return
        # Never run inside a CUDA graph capture stream.  prepare_forward is
        # normally called outside graph context, but guard explicitly.
        if torch.cuda.is_current_stream_capturing():
            return
        from vllm.distributed import get_ep_group

        ep_group = get_ep_group()
        ep_group.all_reduce(self._logical_count_local)
        self._logical_count_global.copy_(self._logical_count_local)
        self._logical_count_local.zero_()
        self._logical_count_ready = True

    def match_step_counts_collective(self) -> None:
        """Issue the count collective without touching the count pipeline.

        A rank running a dummy batch has to take part in the collective, or the
        EP group falls out of order and its peers wait in dispatch until DeepEP
        times out. It must not run `finalize_step_counts`, though: that copies
        the local buffer into the global one and then clears the local. Called a
        second time within a step -- which is exactly what a dummy forward
        beside a real one does -- it copies the freshly cleared buffer over the
        counts that were just published, and the LP then solves against an
        all-zero load for the rest of the run.

        So the dummy path reduces a scratch tensor of the same shape instead:
        same collective, same size, no effect on what the solve reads.
        """
        if self._logical_count_local is None:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        from vllm.distributed import get_ep_group

        if self._count_scratch is None:
            self._count_scratch = torch.zeros_like(self._logical_count_local)
        get_ep_group().all_reduce(self._count_scratch)

    def set_physical_to_logical_map(self, mapping: torch.Tensor) -> None:
        self.physical_to_logical_map = mapping

    def _snapshot(
        self,
        layer_state: EplbLayerState,
        layer_id: int,
    ) -> Any:
        from moe_load_balancer.adapters.vllm import (
            collapse_candidates_to_local,
            to_placement_snapshot,
        )

        # Narrowing an expert's candidates to its local replica is a *routing
        # decision* ("never pay for a cross-GPU hop"), not a view of the
        # placement, so it belongs to the policy that asked for it rather than
        # to every policy.  SGLang draws the line the same way: the collapse
        # lives in `logical_to_rank_dispatch_physical_map`, which it only builds
        # when the pipeline reports `requires_rank_dispatch_map` -- while
        # `init_by_eplb`, the path taken on every rebalance, hands the policy
        # the full global list.
        #
        # Applying it unconditionally answers the question LPLB exists to ask:
        # with a local replica always winning, a rank holding a copy has nothing
        # left to solve, so the LP is skipped outright and the rest see only the
        # experts they do not host.
        # vLLM pads the candidate map to MAX_EXPERT_REDUNDANCY + 1 (1024)
        # columns whatever the configured redundancy, and MLB answers with a
        # table the same width as the candidates it was given. The kernel scans
        # that table with a compile-time loop, so handing over the padded width
        # would have made a share-table answer impossible to compile. Every
        # column past `_max_replicas` is padding on both sides, so trimming to
        # it changes no decision.
        assert layer_state.logical_to_physical_map is not None
        assert layer_state.logical_replica_count is not None
        candidates = layer_state.logical_to_physical_map[:, : self._max_replicas]
        counts = layer_state.logical_replica_count
        # `requires_rank_dispatch_map` is true only for `static`, and static no
        # longer calls this method at all -- its answer is resolved once per
        # placement instead (see `dispatch_fixed_by_placement`). Kept rather
        # than deleted: it is what a future rank-dispatch-map policy would need,
        # and removing it now would be removing untested surface, not dead code.
        if self.requires_rank_dispatch_map:
            candidates, counts = collapse_candidates_to_local(
                candidates,
                counts,
                ep_rank=self.ep_rank,
                num_local_physical_experts=(self.num_physical_experts // self.ep_size),
            )

        defaults = self._default_replicas
        quota = self._ultraep_rank_quota_prefix
        # Allocated is not the same as committed. The bootstrap solve allocates
        # the whole quota tensor before any refresh has run, and what it leaves
        # in it describes a layout vLLM never applied -- all-zero quota,
        # single-replica counts, candidates pointing at slots that hold other
        # experts. Routing through that is worse than not routing at all: it
        # sends tokens to the wrong expert instead of falling back. So the
        # MLB-side tables are used only for layers a refresh has actually
        # written; every other layer takes the framework's own candidates, which
        # is the same path a non-UltraEP policy takes and is always consistent
        # with physical_to_logical_map.
        if quota is not None and layer_id in self._ultraep_committed_layers:
            # The quota's replica columns are only meaningful against the
            # candidate table MLB solved them from -- not vLLM's own
            # candidates, independently rebuilt from physical_to_logical_map
            # and not guaranteed to share its width, let alone its ordering.
            assert self._ultraep_logical_to_physical is not None
            assert self._ultraep_replica_counts is not None
            candidates = self._ultraep_logical_to_physical[layer_id]
            counts = self._ultraep_replica_counts[layer_id]
        return to_placement_snapshot(
            layer_state,
            layer_id,
            physical_to_logical_map=self.physical_to_logical_map[layer_id],
            num_logical_experts=self.num_logical_experts,
            num_physical_experts=self.num_physical_experts,
            ep_size=self.ep_size,
            candidates=candidates,
            counts=counts,
            default_physical_for_logical=(
                None if defaults is None else defaults[layer_id]
            ),
            physical_to_rank_map=self._physical_to_rank_map,
            rank_quota_prefix=None if quota is None else quota[layer_id],
        )

    def _rebuild_default_replicas(self) -> None:
        """Synthesize the per-rank default replica table vLLM does not keep.

        Only `static` replica routing reads it (MLB reports that through
        ``requires_rank_dispatch_map``), and it is recomputed only when a
        placement is committed -- never per forward.
        """
        if not self.requires_rank_dispatch_map:
            self._default_replicas = None
            self._fixed_map = None
            self._fixed_counts = None
            return

        from moe_load_balancer.adapters.vllm import nearest_replica_table

        self._default_replicas = torch.stack(
            [
                nearest_replica_table(
                    self._logical_to_physical_map[layer],
                    self._logical_replica_count[layer],
                    ep_rank=self.ep_rank,
                    num_local_physical_experts=(
                        self.num_physical_experts // self.ep_size
                    ),
                    # Passed rather than inferred: inference reads the largest
                    # physical id present, which undercounts whenever the top
                    # ranks hold no replica, and this rank can then fall outside
                    # the table it just built.
                    ep_size=self.ep_size,
                )
                for layer in range(self._logical_to_physical_map.shape[0])
            ]
        )

        if not self.dispatch_fixed_by_placement:
            self._fixed_map = None
            self._fixed_counts = None
            return

        # [layers, num_logical] -> [layers, num_logical, 1]. One column, so the
        # kernel's gather strides by 1 instead of by the candidate map's padded
        # width of MAX_EXPERT_REDUNDANCY + 1, and reads a table small enough to
        # stay cached.
        self._fixed_map = (
            self._default_replicas.to(self._logical_to_physical_map.dtype)
            .unsqueeze(-1)
            .contiguous()
        )
        # Every expert has exactly one candidate in that layout, so `hash %
        # count` is 0 for every token and the kernel returns column 0 -- which
        # is the replica this policy chose. Same answer as calling the policy
        # per forward, without the call.
        self._fixed_counts = torch.ones(
            self._default_replicas.shape[1],
            dtype=self._logical_replica_count.dtype,
            device=self._default_replicas.device,
        ).contiguous()

    def fixed_dispatch_maps(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The placement-resolved map and counts for a layer, if there is one.

        Returned in place of calling the routing pipeline: for a policy that
        decides from placement alone, the per-forward call produced this same
        answer at a measured 426 us per layer against a policy-free path that
        costs nothing.
        """
        if self._fixed_map is None or self._fixed_counts is None:
            return None
        return self._fixed_map[layer_id], self._fixed_counts

    def register_logical_maps(
        self, logical_to_physical_map: torch.Tensor, logical_replica_count: torch.Tensor
    ) -> None:
        """Keep the model-level maps so snapshots can be built without a layer.

        vLLM commits a rearrangement with in-place ``copy_`` into these
        tensors, so holding them here observes every future placement without
        re-registration.
        """
        self._logical_to_physical_map = logical_to_physical_map
        self._logical_replica_count = logical_replica_count
        # Does this placement give any expert a second copy? If not, every
        # replica policy -- static, dynamic, dynamic_random, lplb alike -- can
        # only return the identity, and consulting one is pure cost.
        #
        # Read once here rather than per forward: `.any()` on a CUDA tensor is
        # a device sync, which is cheap at a rearrangement and ruinous inside a
        # forward. Kept as a Python bool for that reason.
        #
        # This belongs to the placement, not to a policy. LPLB happens to
        # short-circuit itself (`num_red_log == 0` leaves its LP with nothing
        # to solve), static does not -- it consults its nearest-replica table
        # whatever the redundancy is. Leaving each policy to notice the
        # degenerate case independently is how that asymmetry arose, and it is
        # exactly what made a red0 row -- where by construction no policy can
        # differ -- report a 2.6% spread between them.
        self._placement_offers_replica_choice = bool(
            (logical_replica_count > 1).any().item()
        )
        self._rebuild_default_replicas()

    def announce_initial_placement(self) -> None:
        """Tell MLB about the placement the model just loaded with.

        The contract is "after initial weight loading **and** after each
        committed update"; SGLang does the first half in
        ``ModelRunner._prepare_moe_topk``.  Skipping it leaves a policy to build
        its per-layer state lazily inside the first forward, from whatever
        placement that forward happens to see.
        """
        if not self.requires_placement_state:
            return
        num_layers = self._logical_to_physical_map.shape[0]
        self._commit_layers(range(num_layers))
        logger.info("MLB: announced initial placement for %d layers", num_layers)

    def _commit_layers(self, layer_ids) -> None:
        """Hand MLB the committed placement of the given layers."""
        from moe_load_balancer.core.types import PlacementSnapshot

        for layer_id in layer_ids:
            self._mlb.on_placement_committed(
                PlacementSnapshot(
                    layer_id=layer_id,
                    num_logical_experts=self.num_logical_experts,
                    num_physical_experts=self.num_physical_experts,
                    ep_size=self.ep_size,
                    num_local_physical_experts=(
                        self.num_physical_experts // self.ep_size
                    ),
                    physical_to_logical_map=self.physical_to_logical_map[layer_id],
                    # Same trim as `_snapshot`, and it has to be the same: the
                    # policy sizes its per-layer state from whichever snapshot
                    # reaches it first, and this one does -- it runs for every
                    # layer at startup. Committing the padded width here left
                    # the solver emitting 1024-wide tables that the mapping
                    # kernel then could not compile.
                    logical_to_physical_candidates=self._logical_to_physical_map[
                        layer_id
                    ][:, : self._max_replicas],
                    logical_to_physical_count=self._logical_replica_count[layer_id],
                    default_physical_for_logical=(
                        None
                        if self._default_replicas is None
                        else self._default_replicas[layer_id]
                    ),
                )
            )

    def on_placement_committed(
        self, changed_layer_ids: list[int] | None = None
    ) -> None:
        """Refresh policy state for layers whose placement just changed.

        Must be called after vLLM has moved weights and updated its live
        metadata -- never from an uncommitted plan.

        ``changed_layer_ids`` matters for cost, not just tidiness.  A policy
        whose per-layer state layout changes has to rebuild it, and for `lplb`
        that means a JIT build plus warmup -- about 14.5 s per layer measured
        here.  Refreshing all 40 layers stalled the worker for ~102 s, long
        enough for the peer's collective to time out and kill it.  SGLang
        avoids this by passing the framework's own ``update_layer_ids``
        (``ModelRunner._notify_mlb_placement_committed``); vLLM does not track
        them, so Glue diffs the maps before committing and passes the result.
        """
        # The nearest-replica table is *this* side's derived state, not the
        # policy's, so it is refreshed on every commit. Gating it on
        # requires_placement_state left a policy that keeps no state of its
        # own -- static, which reads this table and nothing else -- routing
        # by the placement the run started with. After a rearrangement most
        # defaults are no longer in their expert's candidate list, the share
        # table falls back to column 0 for them, and the traffic those
        # replicas exist to spread lands on one of them.

        self._rebuild_default_replicas()

        if not self.requires_placement_state:
            return
        if changed_layer_ids is None:
            changed_layer_ids = list(range(self._logical_to_physical_map.shape[0]))
        if not changed_layer_ids:
            return
        self._commit_layers(changed_layer_ids)
        logger.info(
            "MLB: refreshed placement state for %d/%d layers",
            len(changed_layer_ids),
            self._logical_to_physical_map.shape[0],
        )

    def resolve_routing(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        layer_state: EplbLayerState,
        num_unpadded_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Ask MLB to route.

        Returns ``(replica_shares, physical_ids)``. Every replica policy
        resolves its own ids now, so the first element is always ``None`` --
        kept in the return shape because the caller's fused kernel still has a
        share-table mode (``HAS_REPLICA_PROB``) it did not stop supporting,
        even though nothing on the MLB side asks for it any more.
        """
        shares = self.replica_shares(
            topk_ids, topk_weights, layer_state, num_unpadded_tokens
        )
        ids = None if shares is not None else self._last_physical_ids
        return shares, ids

    def resolve_shared_expert_rank(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        layer_state: EplbLayerState,
        num_unpadded_tokens: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """The per-token home rank for the shared expert, or None.

        Reads what the pipeline just decided rather than running it again:
        ``resolve_routing`` has already crossed the routing boundary for this
        layer, and Waterfill's answer came back in the same decision. Calling
        MLB twice would double the only per-layer cost this integration has.

        None means "no policy answered", which the caller reads as "this
        rank" -- the placement the replicated MLP had.
        """
        del topk_ids, topk_weights, layer_state, num_unpadded_tokens
        return self._last_shared_expert_rank

    def _ultraep_fast_refresh(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        num_unpadded_tokens: torch.Tensor | None,
    ) -> None:
        """Real, traffic-driven placement refresh + weight transfer, one layer.

        Not CUDA-graph-safe: issues real collectives (all_gather, a
        distributed weight transfer, a barrier) whenever it actually
        refreshes, so it must never run under capture -- same reason
        ``finalize_step_counts`` guards on the same check.

        Gated on ``_ultraep_rank_quota_prefix`` already being allocated:
        those whole-model buffers are only created by ``plan_placement()``'s
        slow-path solve (see ``mlb_runtime.py``'s ``plan_placement``), which
        still owns this model's *initial* placement. This refresh writes
        only a per-layer slice into buffers that solve already allocated; it
        does nothing before that first solve has run.

        Writes its result directly into the same whole-model buffers
        ``_snapshot()`` reads every forward (``physical_to_logical_map``,
        ``_ultraep_logical_to_physical``, ``_ultraep_replica_counts``,
        ``_ultraep_rank_quota_prefix``) rather than a separate pending store:
        ``replica_shares()`` calls ``_snapshot()`` again immediately after
        this returns, on its way to dispatch, so there is no window where a
        separate "pending" representation could go stale relative to what
        this just wrote.

        The placement solve is this file's own (MLB's L1, through
        ``self._mlb.plan_placement``); physically moving weight data to match
        it is ``_commit_placement_weights`` below, over ``EplbState``'s own P2P
        mover. The bridge between the two is explicit and is the solve's actual
        output: the before/after physical-to-logical map pair. No second
        component re-derives placement from raw routing to arrive at the same
        answer independently, so there is no shared-determinism assumption to
        hold -- the mover cannot disagree with the solve, because it is handed
        the solve.
        """
        if torch.cuda.is_current_stream_capturing():
            return
        if self._ultraep_rank_quota_prefix is None:
            return
        if self._ultraep_lagged:
            # The plan solved on this layer's previous forward lands now
            # (routing snapshot, then weights), before this forward's routing.
            self._ultraep_apply_pending(layer_id)
        if _current_stage() == "decode":
            # UltraEP is explicitly a prefill-time algorithm (the paper's own
            # scoping; SGLang's reference should_refresh gates the same way,
            # rejecting only its provable "decode" and accepting everything
            # else, including its own "mixed"). A decode-only batch must not
            # count toward this layer's refresh interval at all -- it never
            # reaches is_due(), the same way SGLang's gate never touches its
            # own batch counter for a rejected stage.
            #
            # Everything past this point issues collectives, so this verdict
            # has to be identical on every EP rank or the ranks that continue
            # will wait in the all_gather below for ranks that returned here.
            # That is what `_current_stage()` reads the DP-reduced
            # `uniform_decode_across_dp` for rather than this rank's own batch
            # descriptor: a rank with no requests runs `execute_dummy_batch()`
            # -- `_dummy_run(uniform_decode=True)`, locally a perfect "decode"
            # -- while its peers run real prefill. Gating on the local view
            # splits the group, and permanently desynchronises is_due()'s
            # per-layer counter as well, since a rank that returns here never
            # advances it.
            return

        # is_due() is pure Python counter bookkeeping -- no GPU access, never
        # syncs -- and is always cheap to call. Computing `representative`
        # below is not free either: the fallback source is a device tensor and
        # its `.item()` is a real CPU-GPU sync (the DP source is a CPU tensor
        # and is not),
        # and paying it on every one of interval-1-out-of-interval calls
        # where should_refresh could not possibly say yes anyway is exactly
        # the class of per-forward cost this file goes to real lengths to
        # avoid elsewhere (see finalize_step_counts's whole reason for
        # existing). Check is_due first; only compute representative and
        # call should_refresh when it says a refresh could happen this call.
        if not self._ultraep_refresh_gate.is_due(layer_id):
            return

        # Under DP, this rank's own num_unpadded_tokens says nothing about
        # what its peers are doing this step -- a rank running a dummy batch
        # has no local token count at all, and would otherwise have to guess.
        # dp_counts is the same array on every rank (vLLM's own DP
        # coordination gathers it before any MoE layer runs), so summing it
        # gives every rank -- real or dummy -- an identical verdict. Only
        # fall back to the local, per-rank count outside DP, where no dummy
        # peer exists that could disagree with it.
        dp_counts = _dp_token_counts()
        if dp_counts is not None:
            representative = (
                int(dp_counts.sum().item()) >= self._ultraep_min_representative_tokens
            )
        else:
            representative = (
                num_unpadded_tokens is not None
                and int(num_unpadded_tokens.item())
                >= self._ultraep_min_representative_tokens
            )
        if not self._ultraep_refresh_gate.should_refresh(
            layer_id, representative=representative
        ):
            return

        from moe_load_balancer.adapters.vllm import to_placement_request
        from moe_load_balancer.kernels.ops.counting import count_logical_experts

        from vllm.distributed import get_ep_group

        ep_group = get_ep_group().device_group
        local_count = count_logical_experts(
            topk_ids, self.num_logical_experts, dtype=torch.int32
        )
        per_rank_count = local_count.new_empty((self.ep_size, self.num_logical_experts))
        torch.distributed.all_gather_into_tensor(
            per_rank_count, local_count.contiguous(), group=ep_group
        )

        request = to_placement_request(
            per_rank_count[:, None, :],
            num_replicas=self.num_physical_experts,
            num_ranks=self.ep_size,
            num_groups=1,
            num_nodes=1,
            algorithm="ultraep",
            ep_rank=self.ep_rank,
        )
        if self._time_refresh:
            torch.accelerator.synchronize()
            _ts = time.perf_counter()
        plan = self._mlb.plan_placement(request)
        if self._time_refresh:
            torch.accelerator.synchronize()
            self._solve_us.append((time.perf_counter() - _ts) * 1e6)
        if self._ultraep_lagged:
            self._ultraep_stage_pending(layer_id, plan)
            return

        # The request carried one synthetic layer (dim 0 of per_rank_count),
        # so the plan's own layer index is always 0 regardless of layer_id --
        # layer_id only selects where in *our* whole-model buffers this one
        # layer's slice of the plan lands.
        #
        # Where each physical slot's weight lives now, and where the solve wants
        # it to live. Those two rows *are* the transfer plan handed to the mover
        # below -- nothing re-derives placement from raw routing a second time,
        # so the mover cannot disagree with the solve. Stacked into one tensor
        # because the mover reads them on the host: one device-to-host copy for
        # the pair rather than one each, and the sync is visible here, at the
        # only place in this path that pays it.
        live = self.physical_to_logical_map[layer_id]
        solved = plan.physical_to_logical_map[0]
        placement_change = torch.stack((live, solved.to(live.dtype))).cpu().numpy()

        assert self._ultraep_logical_to_physical is not None
        assert self._ultraep_replica_counts is not None
        assert self._ultraep_rank_quota_prefix is not None
        self.physical_to_logical_map[layer_id].copy_(solved)
        self._ultraep_logical_to_physical[layer_id].copy_(
            plan.logical_to_all_physical_map[0]
        )
        self._ultraep_replica_counts[layer_id].copy_(plan.logical_to_physical_count[0])
        self._ultraep_rank_quota_prefix[layer_id].copy_(
            plan.metadata["rank_quota_prefix"][0]
        )
        # All four tables for this layer now describe the same solve, and the
        # weight move below makes memory match it. Only from here is it sound
        # for _snapshot to route through them.
        self._ultraep_committed_layers.add(layer_id)

        self._timed_commit(layer_id, placement_change[0], placement_change[1])

    def _timed_commit(self, layer_id: int, old_np: Any, new_np: Any) -> None:
        """``_commit_placement_weights``, wrapped in the MLB_TIME_REFRESH timing
        when it is armed: host issue time and device completion time of each
        commit, reported once with the solve times and then disarmed. Shared by
        the same-forward and the next-forward (lagged) apply paths."""
        if not self._time_refresh:
            self._commit_placement_weights(layer_id, old_np, new_np)
            return
        torch.accelerator.synchronize()
        _tc = time.perf_counter()
        self._commit_placement_weights(layer_id, old_np, new_np)
        _host = (time.perf_counter() - _tc) * 1e6
        torch.accelerator.synchronize()
        _dt = (time.perf_counter() - _tc) * 1e6
        self._commit_host_us.append(_host)
        # How much this commit actually had to move: slots whose occupant
        # changed. A refresh that re-solves to the same placement moves
        # nothing, and the first commit for a layer moves everything, so
        # the duration alone says very little without this.
        _moved = int((old_np != new_np).sum())
        self._commit_us.append((_moved, _dt))
        if len(self._commit_us) >= self._time_refresh:
            import statistics

            def _q(v, f):
                return sorted(v)[min(int(f * len(v)), len(v) - 1)] if v else 0.0

            idle = [d for m, d in self._commit_us if m == 0]
            moving = [(m, d) for m, d in self._commit_us if m > 0]
            logger.info(
                "MLB refresh cost over %d refreshes: solve median=%.0f us "
                "p90=%.0f us | commits that moved nothing: n=%d median=%.0f us "
                "| commits that moved slots: n=%d median slots=%.0f "
                "median=%.0f us p90=%.0f us max=%.0f us",
                len(self._commit_us),
                statistics.median(self._solve_us),
                _q(self._solve_us, 0.9),
                len(idle),
                statistics.median(idle) if idle else 0.0,
                len(moving),
                statistics.median([m for m, _ in moving]) if moving else 0.0,
                statistics.median([d for _, d in moving]) if moving else 0.0,
                _q([d for _, d in moving], 0.9),
                max([d for _, d in moving], default=0.0),
            )
            logger.info(
                "MLB commit host-vs-device: host median=%.0f us | "
                "device-complete median=%.0f us -- if the host figure is "
                "well below the device one, the device is the binding "
                "constraint and shaving host work changes nothing",
                statistics.median(self._commit_host_us),
                statistics.median([d for _, d in self._commit_us]),
            )
            self._commit_host_us.clear()
            self._solve_us.clear()
            self._commit_us.clear()
            self._time_refresh = 0

    def _commit_placement_weights(
        self,
        layer_id: int,
        old_np: Any,
        new_np: Any,
    ) -> None:
        """Move expert weight so this layer's memory matches the placement the
        solve just decided.

        Both arrays are logical-expert ids per physical slot, EP-wide -- exactly
        the pair ``EplbState`` hands its own periodic rearrangement, so this is
        that same P2P move driven at this policy's cadence instead of at the
        rearrangement interval. ``move_to_buffer`` talks to the other ranks and
        ``move_from_buffer`` lands the result locally; both run here, in order,
        on the calling stream.
        """
        if (old_np == new_np).all():
            # The solve landed on the placement already in memory. Nothing to
            # move, and skipping is safe for the EP group precisely because
            # every rank compares the same two EP-wide arrays and so reaches
            # this the same way.
            return
        if self._ultraep_mover in ("auto", "symm") and self._commit_symm(
            layer_id, old_np, new_np
        ):
            return
        if self._ultraep_mover in ("auto", "direct") and self._commit_direct(
            layer_id, old_np, new_np
        ):
            return

        from vllm.distributed.eplb.rebalance_execute import (
            move_from_buffer,
            move_to_buffer,
        )

        assert self._ultraep_num_local_physical is not None
        metadata = move_to_buffer(
            num_local_experts=self._ultraep_num_local_physical,
            old_indices=old_np,
            new_indices=new_np,
            expert_weights=self._ultraep_expert_weights[layer_id],
            expert_weights_buffers=self._ultraep_expert_buffer,
            stream=None,
            ep_rank=self.ep_rank,
            communicator=self._ultraep_communicator,
            layer_idx=layer_id,
        )
        move_from_buffer(
            expert_weights=self._ultraep_expert_weights[layer_id],
            expert_weights_buffers=self._ultraep_expert_buffer,
            transfer_metadata=metadata,
            new_indices=new_np,
            ep_rank=self.ep_rank,
        )

    def _ultraep_stage_pending(self, layer_id: int, plan: Any) -> None:
        """Park a solved plan for this layer until its next forward.

        The plan's GPU tensors are kept as-is (plan() allocates fresh ones per
        solve, so nothing is overwritten underneath us) and the (live, solved)
        physical->logical pair is copied to pinned host memory asynchronously
        on the current stream, with an event marking completion. Nothing here
        waits on the device.
        """
        live = self.physical_to_logical_map[layer_id]
        solved = plan.physical_to_logical_map[0].to(live.dtype)
        host = self._ultraep_pending_host.get(layer_id)
        if host is None:
            host = torch.empty((2, live.numel()), dtype=live.dtype, pin_memory=True)
            self._ultraep_pending_host[layer_id] = host
        host.copy_(torch.stack((live, solved)), non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._ultraep_pending[layer_id] = (
            solved,
            plan.logical_to_all_physical_map[0],
            plan.logical_to_physical_count[0],
            plan.metadata["rank_quota_prefix"][0],
            host,
            ev,
        )

    def _ultraep_apply_pending(self, layer_id: int) -> None:
        """Land the plan staged on this layer's previous forward: snapshot first
        (the routing that follows reads it), then the weight sync on the same
        stream, so the experts kernel downstream sees matching weights. By now
        the pinned copy finished a whole step ago, so the event wait is free.
        """
        pending = self._ultraep_pending.pop(layer_id, None)
        if pending is None:
            return
        solved, l2p, counts, quota, host, ev = pending
        ev.synchronize()
        change = host.numpy()
        old_np = change[0].copy()
        new_np = change[1].copy()
        assert self._ultraep_logical_to_physical is not None
        assert self._ultraep_replica_counts is not None
        assert self._ultraep_rank_quota_prefix is not None
        self.physical_to_logical_map[layer_id].copy_(solved)
        self._ultraep_logical_to_physical[layer_id].copy_(l2p)
        self._ultraep_replica_counts[layer_id].copy_(counts)
        self._ultraep_rank_quota_prefix[layer_id].copy_(quota)
        self._ultraep_committed_layers.add(layer_id)
        self._timed_commit(layer_id, old_np, new_np)

    def _replica_changes(self, layer_id: int, old_np: Any, new_np: Any) -> list | None:
        """Replica slots that acquired a new expert, as ``(dst_rank, k)`` pairs
        with ``k`` the slot's index among that rank's replica slots -- or None
        when either placement is not on the fixed-master layout.

        Both fast movers rest on one assumption: the first ``masters_per_rank``
        slots of every rank hold that rank's own logical experts, in order, on
        both sides of the change. The first refresh after loading a
        checkpointed (EPLB-permuted) placement breaks it -- that refresh moves
        every master home -- and is left to the generic mover.
        """
        ep_size = self.ep_size
        num_local = self._ultraep_num_local_physical
        masters_per_rank = self.num_logical_experts // ep_size
        new_grid = np.asarray(new_np).reshape(ep_size, num_local)
        old_grid = np.asarray(old_np).reshape(ep_size, num_local)
        expected = (
            np.arange(ep_size)[:, None] * masters_per_rank
            + np.arange(masters_per_rank)[None, :]
        )
        if not (
            np.array_equal(new_grid[:, :masters_per_rank], expected)
            and np.array_equal(old_grid[:, :masters_per_rank], expected)
        ):
            if not self._ultraep_mover_warned:
                self._ultraep_mover_warned = True
                logger.info(
                    "MLB weight sync: layer %d moves master slots (placement not "
                    "on the fixed-master layout); the generic mover handles it and "
                    "the fast path takes over from the next refresh",
                    layer_id,
                )
            return None
        return np.argwhere(
            (new_grid[:, masters_per_rank:] != old_grid[:, masters_per_rank:])
            & (new_grid[:, masters_per_rank:] >= 0)
        ).tolist()

    def _commit_direct(self, layer_id: int, old_np: Any, new_np: Any) -> bool:
        """Weight sync over the EPLB communicator: masters stay put, and every
        replica slot whose expert changed receives it straight from the rank
        holding the master, in one send/recv group per layer -- no staging
        buffer, no second local copy. Sources are master slots (never written
        by a refresh) and destinations are replica slots (each written by one
        recv), so stream order is the only ordering needed.

        Returns False to hand the layer to the generic mover.
        """
        changed = self._replica_changes(layer_id, old_np, new_np)
        if changed is None:
            return False
        masters_per_rank = self.num_logical_experts // self.ep_size
        new_grid = np.asarray(new_np).reshape(
            self.ep_size, self._ultraep_num_local_physical
        )
        weights = self._ultraep_expert_weights[layer_id]
        comm = self._ultraep_communicator
        posted = False
        for dst_rank, k in changed:
            dst_row = masters_per_rank + k
            expert = int(new_grid[dst_rank, dst_row])
            src_rank, src_row = divmod(expert, masters_per_rank)
            if dst_rank == src_rank:
                if self.ep_rank == dst_rank:
                    for w in weights:
                        w[dst_row].copy_(w[src_row], non_blocking=True)
                continue
            if self.ep_rank == src_rank:
                comm.add_send([w[src_row] for w in weights], dst_rank, expert_id=expert)
                posted = True
            elif self.ep_rank == dst_rank:
                comm.add_recv([w[dst_row] for w in weights], src_rank, expert_id=expert)
                posted = True
        if posted:
            comm.execute()
        return True

    def _symm_setup(self) -> bool:
        """Allocate and rendezvous the per-rank staging buffer (collective on the
        EP group, so every rank must reach its first commit together -- they do,
        the refresh runs in lockstep).

        Layout: ``[2 sets, replica slots per rank, expert bytes]``. An expert's
        tensors (weights and scales) are packed back to back as raw bytes in the
        order ``expert_weights[layer]`` lists them; every layer has the same
        shapes, so one layout serves all layers. Two sets alternate with the
        layer index: the unpack of layer L is stream-ordered before this rank's
        all-gather for layer L+1, and every peer's puts for layer L+1 are issued
        after that all-gather completes, so set L%2 is never overwritten while
        it is still being read.

        The decision to use symmetric memory is collective. Each rank first
        judges locally whether it can take part -- the module imports, and the
        EP group either sits on one host or runs on Blackwell-class devices
        (taken as a multi-node NVLink fabric; ``MLB_ULTRAEP_MOVER=symm`` skips
        that heuristic) -- and the verdicts are combined with an all-reduce
        (min) over the EP group, so every rank either enters the rendezvous or
        none does. Without that step a rank whose handle exchange failed fell
        back to direct while its peers waited in the rendezvous, and the
        server deadlocked on its first refresh whenever the EP group spanned
        hosts without such a fabric.

        Returns False (and drops to the direct mover) when the group cannot use
        symmetric memory.
        """
        if self._symm_hdl is not None:
            return True
        from vllm.distributed import get_ep_group
        from vllm.platforms import current_platform

        ep = get_ep_group()
        device = self._ultraep_expert_weights[0][0].device
        reason = ""
        try:
            import torch.distributed._symmetric_memory as symm  # noqa: F401
        except Exception as exc:  # pragma: no cover - build without symm
            reason = f"symmetric memory unavailable ({exc})"
        if not reason and self._ultraep_mover != "symm":
            import socket

            hosts: list[str | None] = [None] * ep.world_size
            torch.distributed.all_gather_object(
                hosts, socket.gethostname(), group=ep.cpu_group
            )
            capability = current_platform.get_device_capability()
            multi_node_nvlink = capability is not None and capability.major >= 10
            if len(set(hosts)) > 1 and not multi_node_nvlink:
                reason = "EP group spans hosts without a multi-node NVLink fabric"
        verdict = torch.tensor([0 if reason else 1], device=device, dtype=torch.int32)
        torch.distributed.all_reduce(
            verdict, op=torch.distributed.ReduceOp.MIN, group=ep.device_group
        )
        if int(verdict.item()) == 0:
            logger.info(
                "MLB weight sync: %s; using the direct send/recv mover on every rank",
                reason or "another EP rank cannot use symmetric memory",
            )
            self._ultraep_mover = "direct"
            return False
        try:
            import torch.distributed._symmetric_memory as symm

            weights0 = self._ultraep_expert_weights[0]
            layout: list[tuple[int, int, torch.dtype, tuple[int, ...]]] = []
            off = 0
            for w in weights0:
                row = w[0]
                nbytes = row.numel() * row.element_size()
                layout.append((off, nbytes, w.dtype, tuple(row.shape)))
                off += nbytes
            assert self._ultraep_num_local_physical is not None
            n_rep = (
                self._ultraep_num_local_physical
                - self.num_logical_experts // self.ep_size
            )
            buf = symm.empty(
                (2, n_rep, off), dtype=torch.uint8, device=weights0[0].device
            )
            hdl = symm.rendezvous(buf, group=ep.device_group)
        except Exception as exc:
            logger.warning(
                "MLB weight sync: symmetric-memory rendezvous failed (%s); using "
                "the direct send/recv mover",
                exc,
            )
            self._ultraep_mover = "direct"
            return False
        self._symm_layout = layout
        self._symm_expert_nbytes = off
        self._symm_buf, self._symm_hdl = buf, hdl
        logger.info(
            "MLB weight sync: symmetric-memory mover ready "
            "(%d replica slot(s) x %d MiB, 2 sets, %d tensors per expert)",
            n_rep,
            off >> 20,
            len(layout),
        )
        return True

    def _commit_symm(self, layer_id: int, old_np: Any, new_np: Any) -> bool:
        """UltraEP weight_sync semantics over torch symmetric memory.

        Every rank owning the master of an expert that some replica slot just
        acquired writes that master straight into the acquiring rank's staging
        slot -- a device copy into mapped peer memory, the peer does nothing --
        then signals that peer; each acquiring rank waits for exactly the peers
        it received from and unpacks its staging slots into its replica rows.
        No group-wide barrier, no host round trip.

        Returns False to hand the layer to the generic mover.
        """
        if not self._symm_setup():
            return False
        changed = self._replica_changes(layer_id, old_np, new_np)
        if changed is None:
            return False
        masters_per_rank = self.num_logical_experts // self.ep_size
        assert self._ultraep_num_local_physical is not None
        n_rep = self._ultraep_num_local_physical - masters_per_rank
        new_grid = np.asarray(new_np).reshape(
            self.ep_size, self._ultraep_num_local_physical
        )
        weights = self._ultraep_expert_weights[layer_id]
        set_id = layer_id % 2
        hdl, buf = self._symm_hdl, self._symm_buf
        assert buf is not None
        signal_to: set[int] = set()
        wait_for: set[int] = set()
        # 1. puts: I own the master -> write into the acquirer's staging slot
        for dst_rank, k in changed:
            expert = int(new_grid[dst_rank, masters_per_rank + k])
            src_rank, src_row = divmod(expert, masters_per_rank)
            if src_rank == dst_rank:
                if self.ep_rank == dst_rank:
                    for w in weights:
                        w[masters_per_rank + k].copy_(w[src_row], non_blocking=True)
                continue
            if self.ep_rank == dst_rank:
                wait_for.add(src_rank)
                continue
            if self.ep_rank != src_rank:
                continue
            remote = hdl.get_buffer(
                dst_rank, (2, n_rep, self._symm_expert_nbytes), torch.uint8
            )
            stage = remote[set_id, k]
            for w, (off, nbytes, _dtype, _shape) in zip(weights, self._symm_layout):
                stage[off : off + nbytes].copy_(
                    w[src_row].contiguous().view(-1).view(torch.uint8),
                    non_blocking=True,
                )
            signal_to.add(dst_rank)
        # 2. pairwise completion, stream-ordered after the puts
        for dst_rank in signal_to:
            hdl.put_signal(dst_rank, channel=0)
        for src_rank in wait_for:
            hdl.wait_signal(src_rank, channel=0)
        # 3. unpack what I acquired
        for dst_rank, k in changed:
            if dst_rank != self.ep_rank:
                continue
            expert = int(new_grid[dst_rank, masters_per_rank + k])
            if expert // masters_per_rank == self.ep_rank:
                continue  # local copy done above
            stage = buf[set_id, k]
            for w, (off, nbytes, dtype, shape) in zip(weights, self._symm_layout):
                w[masters_per_rank + k].copy_(
                    stage[off : off + nbytes].view(dtype).view(shape), non_blocking=True
                )
        return True

    def replica_shares(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        layer_state: EplbLayerState,
        num_unpadded_tokens: torch.Tensor | None,
        routed_scaling_factor: float = 1.0,
    ) -> torch.Tensor | None:
        """Ask MLB to route this layer, recording resolved ids as a side effect.

        Always returns ``None``: every replica policy resolves its own ids
        (``self._last_physical_ids``) rather than answering with a share table
        for this method to hand back. The resolved ids are what
        ``resolve_routing`` reads after this call returns.

        ``routed_scaling_factor`` is plumbed but left at its default: MLB only
        reads it when it materializes a shared-expert decision into expanded
        ids, which is SGLang's contract. vLLM takes back the *rank* and builds
        the extra top-k column itself, applying 1/routed_scaling_factor as that
        column's weight in the layer (see shared_expert_fusion.py).
        """
        from moe_load_balancer.adapters.vllm import (
            to_routing_request,
            to_vllm_shared_expert_rank,
        )
        from moe_load_balancer.kernels.ops.counting import count_logical_experts

        self._last_physical_ids = None
        self._last_shared_expert_rank = None
        layer_id = layer_state.moe_layer_idx
        if layer_id is None:
            raise RuntimeError(
                "MLB routing requires EplbLayerState.moe_layer_idx; the layer "
                "was registered by a path that does not set it."
            )

        # Accumulate this layer's LOCAL count into the stable buffer. Skipped
        # during Dynamo tracing, where the counting kernel may not be handled;
        # under graph capture and eager it runs normally and is captured
        # alongside the copy_.
        #
        # This happens BEFORE the readiness check below, not after it. With the
        # write on the far side, the opening step returned early without
        # recording anything, the next step's reduction still saw zeros, and the
        # solve after that was the first to meet real data -- a three-step
        # warm-up where the design calls for two, leaving one step of every run
        # solving against an all-zero load.
        if self._logical_count_local is not None and not torch._dynamo.is_compiling():
            local = count_logical_experts(topk_ids, self.num_logical_experts)
            self._logical_count_local[layer_id].copy_(local)
        # A placement with no redundancy leaves a replica policy nothing to
        # decide: every logical expert has exactly one physical copy, so the
        # only answer any of them can give is the one the framework's own
        # kernel already produces. Return before the pipeline runs.
        #
        # After the count write above, not before it -- the counts feed the
        # next placement and LPLB's solve, and neither stops being wanted just
        # because this placement happens to be degenerate.
        #
        # A shared-expert policy is exempt: it decides which rank runs the
        # shared expert, a choice that exists whether or not any logical
        # expert has a second copy. Returning here for `waterfill` (which
        # expands to `static+waterfill`) would drop its decision on every
        # red0 deployment.
        if not self._placement_offers_replica_choice and not self.routes_shared_expert:
            return None

        # Only policies that read the load have to wait for it. Gating every
        # policy on this readiness flag disabled the ones that never asked for
        # a count: their buffers are not allocated, so the flag never flips and
        # they returned None for the lifetime of the run.
        if self.consumes_global_logical_count and not self._logical_count_ready:
            return None

        if self._ultraep_expert_weights is not None:
            self._ultraep_fast_refresh(layer_id, topk_ids, num_unpadded_tokens)

        # Pass the PREVIOUS step's global count to MLB.  The stable tensor
        # slice has a fixed GPU address, so it is safe to use inside a
        # captured CUDA graph: on replay the LP solve reads whatever value
        # finalize_step_counts deposited before the graph was launched.
        global_count = (
            None
            if self._fresh_counts
            else (
                self._logical_count_global[layer_id]
                if self._logical_count_global is not None
                else None
            )
        )

        if self._time_l2:
            torch.accelerator.synchronize()
            _t0 = time.perf_counter()

        decision = self._mlb.route_tokens(
            to_routing_request(
                layer_id=layer_id,
                logical_topk_ids=topk_ids,
                topk_weights=topk_weights,
                placement=self._snapshot(layer_state, layer_id),
                stage=_current_stage(),
                token_count=num_unpadded_tokens,
                routed_scaling_factor=routed_scaling_factor,
                global_logical_count=global_count,
            )
        )
        if self._time_l2:
            torch.accelerator.synchronize()
            self._l2_us.append((time.perf_counter() - _t0) * 1e6)
            if len(self._l2_us) >= self._time_l2:
                import statistics

                logger.info(
                    "MLB L2 solve cost: n=%d  median=%.0f us  p90=%.0f us  "
                    "mean=%.0f us",
                    len(self._l2_us),
                    statistics.median(self._l2_us),
                    sorted(self._l2_us)[int(0.9 * len(self._l2_us))],
                    statistics.mean(self._l2_us),
                )
                self._l2_us.clear()
                self._time_l2 = 0

        # MLB_DUMP_LP=<dir> captures what the solve was given and what it
        # returned, so the achievable headroom can be computed offline instead
        # of inferred from end-to-end throughput. Diagnostic only. Unconditional
        # on which policy is active -- global_count/candidates/counts come from
        # layer_state regardless, and lp_probability is None for anything but
        # LPLB, which is exactly what "no LP ran here" should look like.
        if self._dump_lp_skip > 0:
            self._dump_lp_skip -= 1
        elif self._dump_lp_dir is not None and self._dump_lp_left > 0:
            import os as _os

            self._dump_lp_left -= 1
            lp_probability = decision.metadata.get("lp_probability")
            assert layer_state.logical_to_physical_map is not None
            assert layer_state.logical_replica_count is not None
            torch.save(
                {
                    "layer_id": layer_id,
                    "ep_size": self.ep_size,
                    "num_local": self.num_physical_experts // self.ep_size,
                    "candidates": layer_state.logical_to_physical_map[
                        :, : self._max_replicas
                    ].cpu(),
                    "counts": layer_state.logical_replica_count.cpu(),
                    "global_count": (
                        None if global_count is None else global_count.float().cpu()
                    ),
                    "probability": (
                        None if lp_probability is None else lp_probability.float().cpu()
                    ),
                },
                _os.path.join(
                    self._dump_lp_dir,
                    f"lp_r{self.ep_rank}_l{layer_id}_{self._dump_lp_left}.pt",
                ),
            )

        # Every replica policy resolves its own ids now; none answers with a
        # share table for this runtime to apply. Record the ids for
        # resolve_routing to pick up.
        #
        # The shared-expert rank is recorded before the `ids is topk_ids`
        # early return below: a pipeline can decide a rank while leaving the
        # routed ids untouched (bare `waterfill` does exactly that), and
        # returning early there would discard the only decision it made.
        self._last_shared_expert_rank = to_vllm_shared_expert_rank(decision)

        ids = decision.routed_physical_topk_ids
        if ids is None or ids is topk_ids:
            return None
        self._last_physical_ids = ids
        return None


class VllmMlbIntegration:
    """One balancer per engine, serving both placement and routing.

    SGLang gives an engine a single MoELoadBalancer and lets L1 and L2 share
    it. vLLM did not: the L1 policy built a throwaway instance on every
    rebalance and L2 held a second one in a module global. Two instances cannot
    see each other's state, which rules out any policy whose placement decision
    depends on what routing observed -- the premise of the predictive layer --
    and a module global also rules out more than one engine in a process.

    The lookup below stays module-level because vLLM's placement policy is a
    classmethod with nowhere to hang an instance. Ownership is not: the
    integration is created and released by EplbState, so its lifetime is the
    engine's.
    """

    def __init__(self, algorithm: str = "") -> None:
        from moe_load_balancer import MoELoadBalancer

        self.algorithm = algorithm
        self._balancer_kwargs: dict | None = None
        self._balancer = None if algorithm else MoELoadBalancer()
        self.routing: MlbRoutingRuntime | None = None

    def balancer(self):
        """The single MoELoadBalancer this engine uses."""
        if self._balancer is None:
            from moe_load_balancer import MoELoadBalancer

            if self._balancer_kwargs is None:
                # Placement can be asked for before the routing geometry is
                # known; a plain planner answers L1 and is replaced in place
                # once routing supplies the topology.
                self._balancer = MoELoadBalancer()
            else:
                self._balancer = MoELoadBalancer.from_algorithm(
                    self.algorithm, **self._balancer_kwargs
                )
        return self._balancer

    def bind_routing(self, **kwargs) -> MlbRoutingRuntime:
        """Create the routing runtime against this engine's balancer."""
        from moe_load_balancer import MoELoadBalancer

        self._balancer_kwargs = {
            "ep_size": kwargs["ep_size"],
            "source_rank": kwargs["ep_rank"],
            "experts_per_rank": kwargs["num_physical_experts"] // kwargs["ep_size"],
            "collectives": VllmRoutingCollectives(),
        }
        self._balancer = MoELoadBalancer.from_algorithm(
            self.algorithm, **self._balancer_kwargs
        )
        self.routing = MlbRoutingRuntime(
            self.algorithm, balancer=self._balancer, **kwargs
        )
        return self.routing


_integration: VllmMlbIntegration | None = None


def get_mlb_integration() -> VllmMlbIntegration:
    """The engine's integration, created on first use."""
    global _integration
    if _integration is None:
        _integration = VllmMlbIntegration(mlb_l2_algorithm())
    return _integration


def set_mlb_integration(integration: VllmMlbIntegration | None) -> None:
    global _integration, _runtime
    _integration = integration
    _runtime = None if integration is None else integration.routing


def l2_pipeline_capabilities(algorithm: str):
    """Capabilities the named L2 pipeline declares, or None if unknown.

    Stateless, so it answers before any engine exists -- configuration
    validation needs it. Living here rather than at each call site keeps the
    balancer a detail of one module: everything else in vLLM asks this file.
    """
    if not algorithm:
        return None
    try:
        from moe_load_balancer.core.routing_pipeline import RoutingPipeline

        return RoutingPipeline.from_value(algorithm).capabilities
    except Exception:
        return None


def l2_inapplicable_reason(algorithm: str, num_redundant_experts: int) -> str | None:
    """Why the named policy cannot act on this deployment, or None.

    The reason is the policy's own words. What this side supplies is what only
    it knows: that the L1 plan's metadata reaches the committed snapshot
    (``to_placement_snapshot`` carries it, which is how ultraep's quota gets
    to its router), and whether the shared expert is dispatched at all: with
    VLLM_FUSE_SHARED_EXPERTS off it is a per-rank replicated MLP, so there is
    no rank for a shared-expert policy to choose and the policy says so
    itself.

    Answered from the env switch rather than from a built layer because this
    runs during configuration validation, before any model exists. A layer
    that then turns out not to be fusible (uneven split, EP disabled) simply
    never asks MLB for a shared-expert rank -- the router checks its own
    geometry before it calls.
    """
    if not algorithm:
        return None
    try:
        from moe_load_balancer import ExpertDeploymentConfig
        from moe_load_balancer.core.routing_pipeline import RoutingPipeline

        pipeline = RoutingPipeline.from_value(algorithm)
    except Exception:
        return None
    from vllm.model_executor.layers.fused_moe.shared_expert_fusion import (
        shared_expert_fusion_enabled,
    )

    return pipeline.is_applicable(
        ExpertDeploymentConfig(
            num_redundant_experts=num_redundant_experts,
            routes_shared_expert=shared_expert_fusion_enabled(),
            carries_placement_metadata=True,
        )
    )


def plan_placement(request):
    """Run L1 through the engine's balancer and hand back vLLM's one map."""
    from moe_load_balancer.adapters.vllm import to_vllm_physical_to_logical

    integration = get_mlb_integration()
    plan = integration.balancer().plan_placement(request)
    # UltraEP is the only L1 policy that publishes this; every other plan's
    # metadata simply lacks the key, so this stays a no-op for them. Routing
    # geometry can lag placement (see balancer()'s docstring), so there may be
    # no runtime to hand the quota to yet -- it reads whatever the next solve
    # after bind_routing() leaves here.
    #
    # The candidate table and replica counts come along too, not just the
    # quota: they must be read from this same plan, not rebuilt from
    # phy2log through vLLM's own compute_logical_maps, or the quota's column
    # ordering and width silently stop matching the table L2 routes against.
    #
    # Stashed here rather than after the caller commits the weight move: no
    # forward can observe this quota paired with the placement it belongs to
    # before that commit happens, because MlbEplbPolicy requires synchronous
    # (use_async=False) rearrangement -- nothing else runs on this thread
    # between this return and register_logical_maps(). An async L1 path would
    # need this to move to the commit step instead.
    quota = plan.metadata.get("rank_quota_prefix")
    if quota is not None and integration.routing is not None:
        integration.routing._ultraep_rank_quota_prefix = quota
        integration.routing._ultraep_logical_to_physical = (
            plan.logical_to_all_physical_map
        )
        integration.routing._ultraep_replica_counts = plan.logical_to_physical_count
        # All three come from the same plan whose physical_to_logical_map the
        # caller commits next, so every layer's tables are consistent with the
        # placement as of this moment. Say so: `_snapshot` routes a layer
        # against MLB's tables only once it appears here, and leaving the set
        # empty left this solve's routing unused until something else added the
        # layer. Nothing was observably wrong -- RefreshGate refreshes on a
        # layer's first call whatever the interval, so every layer was added on
        # its first forward -- but the invariant was being maintained by a
        # second component's incidental behaviour rather than by the code that
        # knows the tables are ready. Stating it here is what makes the gate's
        # meaning ("routable") match the condition it tests.
        integration.routing._ultraep_committed_layers = set(range(quota.shape[0]))
    return to_vllm_physical_to_logical(plan), plan


def placement_request(*args, **kwargs):
    """Translate vLLM's rebalance arguments into the neutral request."""
    from moe_load_balancer.adapters.vllm import to_placement_request

    return to_placement_request(*args, **kwargs)


def _reject_graphs_with_rearranging_placement_state(rearranges: bool) -> None:
    """Refuse the one combination that can fault the GPU.

    A policy that keeps per-layer state rebuilds it when a committed placement
    changes that state's layout, and LPLB *replaces* the tensors in that case
    rather than updating them in place -- its own ``prepare_layer`` docstring
    notes that keeping a previously captured graph valid across such a change
    "requires a future fixed-shape LPLB state representation".

    A CUDA graph captures pointers. Once the solver's buffers move, replaying
    the graph reads freed memory: observed as ``Xid 43`` on two devices and a
    dead worker. It is intermittent -- it needs a rearrangement that actually
    changes the layout, so a short benchmark can pass and a long serving run
    can fault. That is worth failing loudly for.

    Both escapes keep MLB routing available: pin the placement (leave
    rearrangement off, which is how the reported LPLB numbers were measured),
    or run eager.
    """
    if not rearranges:
        return
    from vllm.config import get_current_vllm_config

    try:
        compilation = get_current_vllm_config().compilation_config
    except Exception:  # noqa: BLE001 - no config context: leave the decision alone
        return
    if getattr(compilation, "cudagraph_mode", None) is None:
        return
    if compilation.cudagraph_mode.name == "NONE":
        return
    raise ValueError(
        "VLLM_MLB_L2_ALGORITHM keeps per-layer solver state, and EPLB "
        "rearrangement can change that state's layout. LPLB replaces its "
        "tensors on such a change, which a captured CUDA graph cannot follow "
        "-- replay then reads freed memory (Xid 43). Either disable "
        "rearrangement (eplb_config step_interval high, or policy that does "
        "not re-plan), or set enforce_eager=True. Fixing this properly needs "
        "a fixed-shape solver state in moe_load_balancer."
    )


def init_mlb_routing(
    *,
    algorithm: str,
    ep_size: int,
    ep_rank: int,
    num_logical_experts: int,
    num_physical_experts: int,
    physical_to_logical_map: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    rearranges: bool = False,
    expert_weights: Any | None = None,
    expert_buffer: Any | None = None,
    communicator: Any | None = None,
) -> MlbRoutingRuntime | None:
    """Create the routing runtime for a configured L2 algorithm.

    ``algorithm`` comes from ``EPLBConfig.l2_algorithm``, which has already
    resolved the environment default and cleared itself for placements no L2
    policy can act on. Passing it in rather than re-reading the environment is
    what makes that decision binding.

    ``expert_weights`` is the model's own ``expert_weights`` (one entry per
    MoE layer, present at this same call site); ``expert_buffer`` and
    ``communicator`` are ``EplbState``'s own staging buffer and P2P backend
    for moving expert weight between ranks. Only ``ultraep`` reads them -- it
    is the one policy that re-plans placement often enough to need weight
    moved mid-run rather than at the rearrangement cadence -- and it moves it
    with this framework machinery rather than any of its own. Every other
    algorithm ignores all three, so they are safe to leave unset.
    """
    global _runtime
    if not algorithm:
        return None
    integration = get_mlb_integration()
    integration.algorithm = algorithm
    _runtime = integration.bind_routing(
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_logical_experts=num_logical_experts,
        num_physical_experts=num_physical_experts,
        physical_to_logical_map=physical_to_logical_map,
        expert_weights=expert_weights,
        expert_buffer=expert_buffer,
        communicator=communicator,
    )
    _runtime.register_logical_maps(logical_to_physical_map, logical_replica_count)
    # Keyed on the declared stability of the policy's state, not on whether it
    # keeps state at all. A policy whose placement-derived tensors keep their
    # addresses across a rearrangement is safe to capture; refusing it because
    # some other policy is not would bar a combination that never faults.
    if _runtime.graph_stability == "realloc_on_placement_change":
        _reject_graphs_with_rearranging_placement_state(rearranges)
    return _runtime


def get_mlb_routing() -> MlbRoutingRuntime | None:
    return _runtime


def reset_mlb_routing() -> None:
    """Test hook. Releases the engine's integration along with the runtime."""
    global _runtime, _integration
    _runtime = None
    _integration = None
