# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-owned glue for the framework-decoupled MoE Load Balancer (MLB).

MLB (``moe-load-balancer``) implements expert-placement and token-routing
policies once, framework-neutrally, and each serving framework keeps a thin
adapter.  This module is that adapter for vLLM's L1 placement slot: it
translates :class:`AbstractEplbPolicy` arguments into an MLB
``PlacementRequest``, runs the selected MLB policy, and translates the result
back into the ``physical_to_logical_map`` vLLM expects.

Enable with ``--eplb-config '{"policy": "mlb", "use_async": false}'``.
The MLB placement algorithm is selected with ``VLLM_MLB_L1_ALGORITHM``
(default ``auto``); see ``moe_load_balancer`` for the supported names.

Note that ``use_async`` must be disabled: vLLM's async rearrangement path is
validated only against the built-in policy (see ``EPLBConfig``), and MLB's
planner runs synchronously on the caller's thread.

``ultraep`` places redundant replicas by which *rank* is overloaded, not just
which logical expert is hot, so it needs every rank's own count rather than
the cross-rank sum every other algorithm here is happy with.
:meth:`wants_per_rank_weight` tells ``EplbState.rearrange`` to gather instead
of reduce and hand the per-rank breakdown back through ``per_rank_weight``.
It also needs to know which rank *this call* is running on -- its placement
kernel solves once per rank, not once globally, and validates ``rank`` is
in range -- so ``EplbState.rearrange`` passes its own ``ep_rank`` alongside
``per_rank_weight`` for the same policies.
"""

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm.distributed.eplb.policy.abstract import AbstractEplbPolicy
from vllm.distributed.eplb.policy.default import DefaultEplbPolicy
from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_ALGORITHM = "auto"
ALGORITHM_ENV = "VLLM_MLB_L1_ALGORITHM"


def _algorithm() -> str:
    return envs.VLLM_MLB_L1_ALGORITHM or DEFAULT_ALGORITHM


class MlbEplbPolicy(AbstractEplbPolicy):
    """Delegate expert placement to the MoE Load Balancer core."""

    @classmethod
    def wants_per_rank_weight(cls) -> bool:
        return _algorithm() == "ultraep"

    @classmethod
    def rebalance_experts(
        cls,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
        per_rank_weight: torch.Tensor | None = None,
        ep_rank: int | None = None,
    ) -> torch.Tensor:
        try:
            from vllm.distributed.eplb.mlb_runtime import (
                placement_request,
                plan_placement,
            )
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "EPLB policy 'mlb' requires the moe-load-balancer package. "
                "Install it into this environment with "
                "`pip install -e /path/to/moe_load_balancer --no-deps`."
            ) from exc

        algorithm = _algorithm()

        # ultraep's placement kernel derives the cross-rank sum itself from
        # the per-rank breakdown, so hand that over whole instead of the
        # already-summed weight every other algorithm here expects. Unlike
        # every other algorithm's CPU numpy solve, ultraep's is a CUDA
        # kernel -- it stays on-device rather than following weight.cpu().
        logical_count = (
            weight.float().cpu() if per_rank_weight is None else per_rank_weight.float()
        )
        # vLLM passes num_groups=0 for models without expert groups; MLB
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

        # MLB's L1 policies do not consume a previous placement, so the
        # slot-preservation pass that the built-in policy performs inside
        # rebalance_experts is applied here instead.  It only permutes slots
        # within a rank, so the placement MLB decided (which logical experts
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
            "EPLB placement delegated to MLB (algorithm=%s, resolved=%s)",
            algorithm,
            plan.metadata.get("resolved_policy", "unknown"),
        )
        return phy2log
