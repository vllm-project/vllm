# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v41/mono/kernels/moe_shape.py
"""The MoE stages' build key and route shape: what they are built for (``MoeBuild``),
and the task counts, placements and LDS route table that follow from its
routing (``RouteShape``), with the table's readers. Plain Python but for the
readers; the kernel's stages are in ``moe``."""

from dataclasses import dataclass, field

import flydsl.expr as fx

from vllm.models.deepseek_v41.amd.mono.common.plan import BLOCKS, WAVES
from vllm.models.deepseek_v41.amd.mono.stages import moe_plan as mplan
from vllm.models.deepseek_v41.amd.mono.stages.dims import UG_PART, Dims
from vllm.models.deepseek_v41.amd.mono.stages.gemv import ROWS, TILE, token_tiles
from vllm.models.deepseek_v41.amd.mono.stages.router import PARTS as ROUTER_PARTS

EXPERTS, TOPK = 384, 6  # the target's routing (a build's: ``MoeBuild``)
ROUTER0 = 8  # the router's first task: past the xq tasks (a token each)
MASK_WORDS = 2  # an expert's token bitmap in the route table: tokens <= 64
# a route lane's experts (EXPERTS / 64) sorted descending, ties kept in index
# order: the network by their count
SORT_NETS = {
    6: [
        (0, 1),
        (2, 3),
        (4, 5),
        (0, 2),
        (1, 4),
        (3, 5),
        (0, 1),
        (2, 3),
        (4, 5),
        (1, 2),
        (3, 4),
        (2, 3),
    ],
    2: [(0, 1)],
}


@dataclass(frozen=True)
class MoeBuild:
    tokens: int
    # routed experts and a token's top-k: the target's, or the DSpark draft's
    experts: int = EXPERTS
    topk: int = TOPK
    # the TP size: every width a rank holds (``Dims``)
    tp: int = 4
    timeline: bool = field(default=False, metadata={"sym": "tl"})
    # FP4 groups an ug task (``RouteShape.ug_groups``)
    ug_groups: int = 2


@dataclass(frozen=True)
class RouteShape:
    """A build's routing: ``experts`` routed experts, ``topk`` a token of
    ``tokens``, the task counts and placements that follow, and the route table
    ``load_route`` lays out in LDS: every token's (ids, weights), |U|, the
    experts of U, each (t, k)'s slot in U, each expert's slot, the picks in slot
    order and each slot's first position in it (past one token tile also each
    slot's first pick tile and each pick tile's slot)."""

    experts: int
    topk: int
    tokens: int
    tp: int
    # FP4 groups (32 intermediate columns) an ug task, its waves splitting K in
    # halves (2) or quarters (1). 1 is twice the tasks, so the CTAs' ug ends
    # closer together: faster at the draft's widths (40 rows -4 us), slower at
    # the target's (48 rows +3 us; the runners pick)
    ug_groups: int = 2

    @property
    def dims(self) -> Dims:
        return Dims(self.tp)

    @property
    def ug_parts(self) -> int:
        """An expert's ug tasks (``moe_plan.ug_parts``)."""
        return mplan.ug_parts(self.dims.inter_real // UG_PART, self.ug_groups)

    @property
    def ug_k_slices(self) -> int:
        """The K slices of an ug task's waves: a wave a (group, 16-column half,
        slice)."""
        return WAVES // (2 * self.ug_groups)

    @property
    def picks(self) -> int:
        return self.tokens * self.topk

    @property
    def max_u(self) -> int:
        return min(self.picks, self.experts)

    @property
    def router_groups(self) -> tuple[int, int]:
        """``stage_router``'s token groups and their tokens: a group a token
        tile, as many as one round of CTAs holds (a CTA running two serialized
        them: the route waited on the 32 that did at 48 rows)."""
        per_group = self.experts // ROWS * ROUTER_PARTS
        groups = max(1, min(len(token_tiles(self.tokens)), BLOCKS // per_group))
        tokens = -(-self.tokens // groups)
        assert (groups - 1) * tokens < self.tokens
        return groups, tokens

    @property
    def router_tasks(self) -> int:
        # (token group, 16-row tile, K part): ``stage_router``
        return self.router_groups[0] * self.experts // ROWS * ROUTER_PARTS

    @property
    def router0(self) -> int:
        """Past the xq tasks (a token each from CTA 0)."""
        return max(ROUTER0, self.tokens)

    @property
    def route0(self) -> int:
        return self.router0 + self.router_tasks

    @property
    def shared0(self) -> int:
        return self.route0 + max(8, self.tokens)

    @property
    def shared_units(self) -> int:
        """The shared expert's (task, token tile) units (``stage_shared``)."""
        return self.dims.shared_tasks * len(token_tiles(self.tokens))

    @property
    def ug0(self) -> int:
        """The first ug round's placement: the shared units' CTAs last."""
        return self.shared0 + self.shared_units

    @property
    def sort_net(self):
        return SORT_NETS[self.experts // 64]

    @property
    def tab_words(self) -> int:
        return self.tile_slot0 + self.max_tiles if self.multi else self.queue0 + 1

    @property
    def multi(self) -> bool:
        """More than one token tile: an ug task is a (tile of a slot's picks,
        part) (``stage_ug``)."""
        return self.tokens > TILE

    @property
    def slot_tiles(self) -> int:
        """A slot's pick tiles at most (a token picks an expert once)."""
        return len(token_tiles(self.tokens))

    @property
    def max_tiles(self) -> int:
        """The pick tiles of U at most: a slot's partial tile each, then the
        whole tiles, and never more than a tile a pick."""
        return min(self.picks, self.max_u + self.picks // TILE)

    @property
    def tile_first0(self) -> int:
        """Past one token tile: each slot's first pick tile, then their count."""
        return self.queue0 + 1

    @property
    def tile_slot0(self) -> int:
        """Past one token tile: each pick tile's slot of U."""
        return self.tile_first0 + self.max_u + 1

    @property
    def queue0(self) -> int:
        """The ug task this CTA took from the queue (``take_ug_task``)."""
        return self.mask0 + MASK_WORDS * self.experts

    @property
    def expert_slot0(self) -> int:
        """The route table's per-expert slots (a slot where the expert is in U)."""
        return 3 * self.picks + 1 + self.max_u

    @property
    def order0(self) -> int:
        """The picks (t topk + k) in slot order, a slot's ascending."""
        return self.expert_slot0 + self.experts

    @property
    def first0(self) -> int:
        """Each slot's first position in the pick order, then the pick count."""
        return self.order0 + self.picks

    @property
    def mask0(self) -> int:
        """Each expert's tokens that picked it, a bit a token (``MASK_WORDS``)."""
        return self.first0 + self.max_u + 1

    @property
    def tile_picks(self) -> int:
        """A slot's picks at most: a token picks an expert once."""
        return min(self.tokens, 16)

    def route_w(self, tab, s, t, k):
        return fx.ptr_load(tab + (t * 2 * self.topk + self.topk + k)).bitcast(
            fx.Float32
        )

    def n_union(self, tab, s):
        return fx.ptr_load(tab + 2 * s * self.topk)

    def union_expert(self, tab, s, u):
        return fx.ptr_load(tab + (2 * s * self.topk + 1 + u))

    def slot_first(self, tab, u):
        return fx.ptr_load(tab + (self.first0 + u))

    def pick_at(self, tab, i):
        return fx.ptr_load(tab + (self.order0 + i))

    def tile_first(self, tab, u):
        return fx.ptr_load(tab + (self.tile_first0 + u))

    def tile_slot(self, tab, i):
        return fx.ptr_load(tab + (self.tile_slot0 + i))


def route_shape(key: MoeBuild) -> RouteShape:
    return RouteShape(key.experts, key.topk, key.tokens, key.tp, key.ug_groups)
