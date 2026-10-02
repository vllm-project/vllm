# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""EP-dispatched shared experts.

By default vLLM runs a MoE layer's shared expert as a plain MLP replicated on
every rank: every rank computes it for its own tokens and nothing crosses the
all-to-all. That is cheap, but it means a shared-expert token has no *choice*
of rank, so a load balancer has nothing to decide -- which is why MLB's
Waterfill L2 policy (``RoutingPhase.SHARED_EXPERT``) is inapplicable on vLLM
today.

This module gives the shared expert a home rank so it can travel through the
same dispatch as the routed experts. The layout it uses is the one property
everything downstream depends on:

    rank r owns global expert ids [r*S, (r+1)*S)  where S = R + n_shared
        ids [r*S,      r*S + R)  -- that rank's R routed physical experts
        ids [r*S + R,  (r+1)*S)  -- that rank's *own copy* of the shared expert

i.e. every rank gets its own distinct global id for the shared expert. That is
what keeps DeepEP's rank derivation (``expert_id // (num_experts // ep_size)``,
hard-coded in its CUDA kernels) correct, and it is why this cannot reuse
vLLM's existing ``num_fused_shared_experts`` machinery: that one gives the
shared expert a *single* global id that every rank maps to a local slot, and
de-duplicates the resulting replication with a token-level round-robin mask.
That layout suits AITER's "every rank computes the full batch locally, then
reduce" model and is unroutable under dispatch/combine -- ROCm's own MoRI
backend asserts it off for exactly this reason.

Because the routed experts no longer occupy a contiguous ``[0, R*ep_size)``
range, their physical ids must be re-mapped into the widened space before
dispatch. ``remap_routed_ids`` does that; ``append_shared_expert`` does both
the re-map and the extra top-k column in one go.
"""

from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class SharedExpertFusion:
    """Geometry of the widened expert layout, and the ops that use it.

    Immutable and built once per MoE layer: every field is fixed at load time,
    which matters because ``num_local_experts`` sizes the weight tensors and
    ``global_num_experts`` is baked into the DeepEP call.
    """

    ep_size: int
    ep_rank: int
    num_shared_experts: int
    """Shared experts per rank -- ``config.n_shared_experts``, not times ep_size."""
    routed_slots_per_rank: int
    """Physical routed experts per rank: ``(n_routed + n_redundant) // ep_size``."""
    shared_expert_weight: float
    """Top-k weight for the shared slot. 1/routed_scaling_factor when the runner
    scales the combined output, so the shared expert's net contribution stays
    1.0 -- the same correction the AITER path applies."""

    @property
    def slots_per_rank(self) -> int:
        return self.routed_slots_per_rank + self.num_shared_experts

    @property
    def global_num_experts(self) -> int:
        """Widened physical expert count handed to DeepEP and ``expert_map``."""
        return self.slots_per_rank * self.ep_size

    @property
    def num_routed_physical_experts(self) -> int:
        """Expert count in the *un-widened* space the router still works in."""
        return self.routed_slots_per_rank * self.ep_size

    def shared_expert_global_ids(self, ep_rank: int) -> list[int]:
        """Global ids of the shared-expert slots owned by ``ep_rank``."""
        base = ep_rank * self.slots_per_rank + self.routed_slots_per_rank
        return [base + j for j in range(self.num_shared_experts)]

    def remap_routed_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Lift routed physical ids into the widened space.

        ``id -> id + (id // R) * n_shared``: rank r's block ``[r*R, r*R+R)``
        slides to ``[r*S, r*S+R)``, leaving its shared slots free at the end.
        Negative (invalid) ids are passed through untouched.
        """
        shifted = topk_ids + (topk_ids // self.routed_slots_per_rank) * (
            self.num_shared_experts
        )
        return torch.where(topk_ids >= 0, shifted, topk_ids)

    def append_shared_expert(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        shared_expert_rank: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-map the routed columns and add the shared-expert column(s).

        ``shared_expert_rank`` is the per-token home rank, as chosen by an L2
        policy. ``None`` means "this rank", which reproduces the replicated
        MLP's placement exactly while still routing through dispatch -- that is
        the configuration to A/B against, since it isolates the cost of the
        detour from the effect of the balancing decision.
        """
        num_tokens = topk_ids.shape[0]
        ids = self.remap_routed_ids(topk_ids)

        if shared_expert_rank is None:
            home = torch.full(
                (num_tokens, 1),
                self.ep_rank,
                dtype=ids.dtype,
                device=ids.device,
            )
        else:
            home = shared_expert_rank.reshape(num_tokens, 1).to(ids.dtype)

        offsets = torch.arange(
            self.num_shared_experts, dtype=ids.dtype, device=ids.device
        )
        # A policy answers -1 for a token it will not place -- a padded row, or
        # one whose routed ids are all invalid. Arithmetic on that lands
        # somewhere real: -1 * slots + routed + 0 is -1 only because the two
        # terms happen to cancel at num_shared_experts == 1, and at 2 it yields
        # -2, which torch reads as an index from the end rather than as "no
        # expert". Carry the sentinel through explicitly instead of relying on
        # that, and zero the weight so a row that reaches a kernel anyway
        # contributes nothing. remap_routed_ids above guards the routed columns
        # the same way; this is the matching guard for the shared one.
        placed = home >= 0
        shared_ids = torch.where(
            placed,
            home * self.slots_per_rank + self.routed_slots_per_rank + offsets,
            torch.full_like(home, -1),
        )
        shared_weights = torch.where(
            placed,
            torch.full(
                (num_tokens, self.num_shared_experts),
                self.shared_expert_weight,
                dtype=topk_weights.dtype,
                device=topk_weights.device,
            ),
            torch.zeros(
                (num_tokens, self.num_shared_experts),
                dtype=topk_weights.dtype,
                device=topk_weights.device,
            ),
        )
        return (
            torch.cat((ids, shared_ids), dim=-1),
            torch.cat((topk_weights, shared_weights), dim=-1),
        )


def maybe_build_shared_expert_fusion(
    *,
    n_shared_experts: int | None,
    num_physical_experts: int,
    ep_size: int,
    ep_rank: int,
    use_ep: bool,
    is_act_and_mul: bool,
    shared_expert_weight: float,
    layer_name: str = "",
    warn_on_uneven: bool = True,
) -> SharedExpertFusion | None:
    """Build the geometry when this layer can and should dispatch its shared expert.

    Returns ``None`` -- meaning "nothing changes" -- for every configuration
    the widened layout cannot serve, rather than raising: a shared expert is
    not required to be fusible, and the caller falls back to the replicated
    MLP. The one case worth a log is an uneven split, because that one looks
    like it should have worked.
    """
    if not shared_expert_fusion_enabled():
        return None
    if not n_shared_experts:
        return None
    if not is_act_and_mul:
        # The shared expert has to be the same gated-MLP shape as a routed one
        # to occupy an expert slot at all.
        return None
    if not use_ep or ep_size <= 1:
        # Without EP every rank is the only rank: dispatching the shared expert
        # would be a no-op detour, and there is no rank for a policy to choose.
        return None
    if num_physical_experts % ep_size != 0:
        if not warn_on_uneven:
            return None
        logger.warning(
            "Shared-expert fusion is on but %s has %d physical experts "
            "over %d EP ranks, which does not divide evenly. Falling back to "
            "the replicated shared-expert MLP.",
            layer_name or "this MoE layer",
            num_physical_experts,
            ep_size,
        )
        return None

    return SharedExpertFusion(
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_shared_experts=int(n_shared_experts),
        routed_slots_per_rank=num_physical_experts // ep_size,
        shared_expert_weight=shared_expert_weight,
    )


def _l2_pipeline_needs_dispatched_shared_expert() -> bool:
    """Whether the configured MLB L2 pipeline picks the shared expert's rank."""
    from vllm.distributed.eplb.mlb_runtime import mlb_l2_algorithm

    algorithm = mlb_l2_algorithm()
    if not algorithm:
        return False
    try:
        from moe_load_balancer.core.routing_pipeline import RoutingPipeline

        caps = RoutingPipeline.from_value(algorithm).capabilities
    except Exception:
        return False
    return bool(getattr(caps, "routes_shared_expert", False))


def shared_expert_fusion_enabled() -> bool:
    """Whether the shared expert is dispatched rather than replicated per rank.

    The env switch turns it on, and so does asking for an L2 pipeline that
    routes the shared expert: waterfill picks a rank for it, and there is no
    rank to pick unless it goes through dispatch. Selecting that pipeline and
    leaving the switch off used to leave MLB's L2 quietly disabled -- the
    configuration was refused, a line went by in the log, and the run carried
    on doing none of what was asked for. Turning it on and saying so is how
    SGLang resolves the same pair (`enable_deepep_waterfill` forces shared
    expert fusion there and warns), and a loud change of configuration beats a
    silent cancellation of one.

    Says nothing about whether any particular layer *can* fuse -- that is
    ``maybe_build_shared_expert_fusion``'s answer, and it needs a layer's
    geometry to give it.
    """
    if envs.VLLM_FUSE_SHARED_EXPERTS:
        return True
    if _l2_pipeline_needs_dispatched_shared_expert():
        logger.warning_once(
            "Enabling shared-expert fusion: the configured MLB L2 pipeline "
            "(VLLM_MLB_L2_ALGORITHM) assigns the shared expert to a rank, "
            "which requires it to go through expert-parallel dispatch. Set "
            "VLLM_FUSE_SHARED_EXPERTS=1 to make this explicit, or drop "
            "waterfill from the pipeline to keep the shared expert "
            "replicated per rank."
        )
        return True
    return False
