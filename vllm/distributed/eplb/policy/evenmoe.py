# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-owned glue for EvenMoE (the framework-decoupled MoE Load Balancer).

EvenMoE (``evenmoe``) implements expert-placement and token-routing
policies once, framework-neutrally, and each serving framework keeps a thin
adapter.  This module is that adapter for vLLM's L1 placement slot: it
translates :class:`AbstractEplbPolicy` arguments into an EvenMoE
``PlacementRequest``, runs the selected EvenMoE policy, and translates the result
back into the ``physical_to_logical_map`` vLLM expects.

Enable with ``--eplb-config '{"policy": "evenmoe", "use_async": false}'``.
The EvenMoE placement algorithm is selected with ``VLLM_EVENMOE_L1_ALGORITHM``
(default ``auto``); see ``evenmoe`` for the supported names.

Note that ``use_async`` must be disabled: vLLM's async rearrangement path is
validated only against the built-in policy (see ``EPLBConfig``), and EvenMoE's
planner runs synchronously on the caller's thread.

Two optional hooks let a placement policy ask ``EplbState`` for more than the
built-in contract offers, and both are answered by EvenMoE's own declarations
rather than by this module knowing which algorithm is which:

* :meth:`wants_per_rank_weight` -- a planner that solves from each source
  rank's own counts (``ultraep`` places redundant replicas by which *rank* is
  overloaded, not just which logical expert is hot) makes ``rearrange`` gather
  instead of reduce and hand the per-rank breakdown back through
  ``per_rank_weight`` together with the caller's own ``ep_rank`` (its placement
  kernel solves once per rank and validates ``rank`` is in range).
* :meth:`initial_placement` -- a planner whose placement before any
  statistics is its own layout (``ultraep``: each rank's block of master
  experts followed by its redundant slots) supplies it, and ``EplbState``
  moves the loaded weights into it at start-up the way it installs a
  checkpoint, so the first traffic-driven solve only has to touch the
  redundant slots.
"""

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm.distributed.eplb.policy.abstract import AbstractEplbPolicy, EplbStats
from vllm.distributed.eplb.policy.default import DefaultEplbPolicy
from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_ALGORITHM = "auto"
ALGORITHM_ENV = "VLLM_EVENMOE_L1_ALGORITHM"


def l1_algorithm() -> str:
    """The L1 planner vLLM runs: ``VLLM_EVENMOE_L1_ALGORITHM``, else ``auto``.

    Public because the L2 side checks the pairing against it: a routing
    pipeline that works from state its own planner publishes (``ultraep``)
    names the planner it requires, and only this module knows which one is
    selected.
    """
    return envs.VLLM_EVENMOE_L1_ALGORITHM or DEFAULT_ALGORITHM


class EvenMoeEplbPolicy(AbstractEplbPolicy):
    """Delegate expert placement to the EvenMoE core."""

    @classmethod
    def wants_per_rank_weight(cls) -> bool:
        """Whether the selected planner needs ``[ranks, layers, experts]`` counts.

        Asked of the balancer, which reads the planner's own declaration
        (``PlacementPlanner.logical_count``, "global" or "per_source_rank"): a
        name comparison here would have to be kept in step with every planner
        that makes the same choice, and would miss one spelled differently.
        """
        from vllm.distributed.eplb.evenmoe_runtime import placement_logical_count

        return placement_logical_count(l1_algorithm()) == "per_source_rank"

    @classmethod
    def initial_placement(
        cls,
        *,
        num_moe_layers: int,
        num_logical_experts: int,
        num_physical_experts: int,
        num_ranks: int,
        num_groups: int,
        num_nodes: int,
    ) -> torch.Tensor | None:
        """The selected planner's own start-up layout, or None to keep vLLM's.

        Returns ``[num_moe_layers, num_physical_experts]`` on the CPU, the
        shape ``EplbState`` installs a placement checkpoint from. Every rank
        computes it from the geometry alone, so the result is identical EP-wide
        without a collective.
        """
        from vllm.distributed.eplb.evenmoe_runtime import plan_initial_placement

        phy2log = plan_initial_placement(
            l1_algorithm(),
            num_moe_layers=num_moe_layers,
            num_logical_experts=num_logical_experts,
            num_physical_experts=num_physical_experts,
            num_ranks=num_ranks,
            num_groups=num_groups or 1,
            num_nodes=num_nodes,
        )
        if phy2log is None:
            return None
        logger.info_once(
            "EPLB initial placement delegated to EvenMoE (algorithm=%s)", l1_algorithm()
        )
        return phy2log.cpu().to(torch.int64)

    def rebalance_experts(
        self,
        stats: EplbStats,
        old_global_expert_indices: torch.Tensor | None = None,
        per_rank_weight: torch.Tensor | None = None,
        ep_rank: int | None = None,
    ) -> torch.Tensor:
        weight = stats.global_expert_load_window
        num_replicas = stats.num_replicas
        num_groups = stats.num_groups
        num_nodes = stats.num_nodes
        num_ranks = stats.num_gpus
        try:
            from vllm.distributed.eplb.evenmoe_runtime import (
                placement_request,
                plan_placement,
            )
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "EPLB policy 'evenmoe' requires the evenmoe package (formerly "
                "moe-load-balancer). Install it into this environment with "
                "`pip install -e /path/to/evenmoe --no-deps`."
            ) from exc

        algorithm = l1_algorithm()

        # ultraep's placement kernel derives the cross-rank sum itself from
        # the per-rank breakdown, so hand that over whole instead of the
        # already-summed weight every other algorithm here expects. Unlike
        # every other algorithm's CPU numpy solve, ultraep's is a CUDA
        # kernel -- it stays on-device rather than following weight.cpu().
        logical_count = (
            weight.float().cpu() if per_rank_weight is None else per_rank_weight.float()
        )
        # vLLM passes num_groups=0 for models without expert groups; EvenMoE
        # expresses "no grouping" as a single group.
        request = placement_request(
            logical_count,
            num_replicas=num_replicas,
            num_ranks=num_ranks,
            num_groups=num_groups or 1,
            num_nodes=num_nodes,
            algorithm=algorithm,
            old_physical_to_logical_map=(
                None
                if old_global_expert_indices is None
                else old_global_expert_indices.cpu()
            ),
            ep_rank=ep_rank,
        )

        # The engine's balancer, not a fresh one per rebalance: L1 and L2 must
        # be the same instance for any policy whose placement depends on what
        # routing observed.
        phy2log, plan = plan_placement(request)
        phy2log = phy2log.cpu().to(torch.int64)

        # EvenMoE's L1 policies do not consume a previous placement, so the
        # slot-preservation pass that the built-in policy performs inside
        # rebalance_experts is applied here instead.  It only permutes slots
        # within a rank, so the placement EvenMoE decided (which logical experts
        # live on which rank, and how many replicas each gets) is unchanged --
        # only their slot positions are, which is what avoids weight copies.
        if old_global_expert_indices is not None:
            phy2log = torch.from_numpy(
                DefaultEplbPolicy.preserve_intragpu_slots(
                    phy2log.numpy(),
                    num_ranks,
                    old_global_expert_indices.cpu().numpy(),
                )
            )

        logger.info_once(
            "EPLB placement delegated to EvenMoE (algorithm=%s, resolved=%s)",
            algorithm,
            plan.metadata.get("resolved_policy", "unknown"),
        )
        return phy2log
