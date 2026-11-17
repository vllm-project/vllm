# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch geometry of the specialized-decode kernels as a function of the SM
count (and, for cluster kernels, of the occupancy the CUDA runtime reports).

Pure Python (no torch, no CUDA), so every choice can be checked on CPU for any
SM count. The stages call these functions with the live
`_arch.DeviceSpec.sm_count` and the probed occupancy; nothing here is a
substitute for the build-time residency checks in `_arch`, which still fail
closed.

Which launch dimensions depend on the SM count:

=================  ==================================================  =========
stage              grid                                                SM-bound?
=================  ==================================================  =========
front              4 x (heads + 1) CTAs, all spinning                  residency
sparse_decode      rows x Hkv clusters of 16 or 8 CTAs                 width, waves
o_proj             (6144 / 128) x 2 + 16 CTAs (shape-derived)          no
allreduce_norm     rows clusters of `cl` CTAs, all spinning          residency
moe                one CTA per SM: routed = SMs - shared CTAs          split
page_topk          rows x column splits CTAs, no cross-CTA wait        no
index_score        upstream kernel, (batch, 256) CTAs, no waits        no
=================  ==================================================  =========
"""

from __future__ import annotations

from collections.abc import Mapping

# Every grid below is sized from ``multiProcessorCount`` of the live device (it differs
# between parts of one arch and shrinks under MIG or green contexts); nothing here
# assumes a particular SM count.

# --------------------------------------------------------------------------- moe:
# shared-expert CTAs beside the routed CTAs (grid = one CTA per SM)
# --------------------------------------------------------------------------------------
MOE_VALID_N_SHARED = (24, 48)  # what the shared tiling admits at I = 768 per rank
MOE_MIN_ROUTED_CTAS = 64  # the fused router takes at most 2 experts per routed CTA
MOE_MAX_ROWS = 8

_H, _I, _E, _TOPK = 6144, 768, 128, 4
# Bytes one routed CTA group streams per distinct expert (NVFP4 FC1 + FC2 values and
# e4m3 scales per 16) and the shared CTAs stream per call (MXFP8 gate_up + down values
# and UE8M0 scales per 32).
MOE_ROUTED_EXPERT_BYTES = (2 * _I * _H + _H * _I) // 2 + (2 * _I * _H + _H * _I) // 16
MOE_SHARED_EXPERT_BYTES = (2 * _I * _H + _H * _I) + (2 * _I * _H + _H * _I) // 32
# Relative cost of one routed byte vs one shared byte on a CTA (router GEMV,
# FC1 -> FC2 hand-off and split units are not free). On B200 and GB300 this
# value gives 48 shared CTAs up to 2 rows and 24 above, the split the engine is
# validated with there (engine A/B against the stock MoE, its float64
# reference, and the decode-step measurements). A per-device sweep of the
# split is a follow-up; measured choices go into MOE_N_SHARED_OVERRIDES.
MOE_ROUTED_COST = 0.75

# Measured overrides of the model, keyed by live SM count: rows 1..8 -> shared
# CTAs. Empty until an A/B on hardware beats the model for some SM count; the
# model's choice is used for every SM count not listed.
MOE_N_SHARED_OVERRIDES: dict[int, tuple[int, ...]] = {}


def expected_distinct_experts(rows: int, experts: int = _E, topk: int = _TOPK) -> float:
    """E[distinct experts] of `rows` tokens choosing `topk` of `experts`
    uniformly (the routed weight bytes a call streams scale with this)."""
    return experts * (1.0 - (1.0 - topk / experts) ** rows)


def moe_step_cost(
    rows: int, sm_count: int, n_shared: int, routed_cost: float = MOE_ROUTED_COST
) -> float:
    """Bytes-per-CTA model of one engine call: the slower of the routed CTAs
    (the call's distinct experts over SMs - shared CTAs) and the shared CTAs."""
    routed_ctas = sm_count - n_shared
    if routed_ctas <= 0:
        return float("inf")
    routed = (
        routed_cost
        * expected_distinct_experts(rows)
        * MOE_ROUTED_EXPERT_BYTES
        / routed_ctas
    )
    return max(routed, MOE_SHARED_EXPERT_BYTES / n_shared)


def moe_n_shared_model(
    rows: int, sm_count: int, routed_cost: float = MOE_ROUTED_COST
) -> int:
    """The valid shared-CTA count with the lowest modelled cost (ties: more
    shared CTAs) that leaves at least `MOE_MIN_ROUTED_CTAS` routed CTAs."""
    ok = [s for s in MOE_VALID_N_SHARED if sm_count - s >= MOE_MIN_ROUTED_CTAS]
    if not ok:
        raise ValueError(
            f"moe: {sm_count} SMs cannot hold {MOE_MIN_ROUTED_CTAS} routed CTAs beside "
            f"{min(MOE_VALID_N_SHARED)} shared CTAs"
        )
    return min(ok, key=lambda s: (moe_step_cost(rows, sm_count, s, routed_cost), -s))


def moe_n_shared(rows: int, sm_count: int) -> int:
    """Shared-expert CTAs of the engine for `rows` tokens on `sm_count`
    (live) SMs: the model, unless a measured override exists for that count."""
    if not 1 <= rows <= MOE_MAX_ROWS:
        raise ValueError(f"moe: rows {rows} outside 1..{MOE_MAX_ROWS}")
    tab = MOE_N_SHARED_OVERRIDES.get(sm_count)
    nsh = tab[rows - 1] if tab is not None else moe_n_shared_model(rows, sm_count)
    if sm_count - nsh < MOE_MIN_ROUTED_CTAS:
        raise ValueError(
            f"moe: {sm_count} SMs leave {sm_count - nsh} routed CTAs beside {nsh} "
            "shared"
        )
    return nsh


def moe_routed_ctas(rows: int, sm_count: int, n_shared: int | None = None) -> int:
    """Routed CTAs of the engine: every SM not running a shared CTA."""
    nsh = moe_n_shared(rows, sm_count) if n_shared is None else n_shared
    return sm_count - nsh


def moe_fc2_blocks_per_cta(routed_ctas: int) -> int:
    """FC2 32-row output blocks per routed CTA, ceil(6144 / 32 / routed); the
    engine's shared-memory budget was sized for at most 2."""
    return -(-(_H // 32) // routed_ctas)


# --------------------------------------------------------------------------------------
# sparse_decode: cluster width per (rows, KV heads)
# --------------------------------------------------------------------------------------
SPARSE_CLUSTER_SIZES = (16, 8)  # instantiated widths, widest first
# From this many (row, KV head) clusters on, use 8-CTA clusters even when 16 would fit
# (the wave count matters more there than the shorter per-CTA page loop of 16). Applies
# to the one-wave rows only.
SPARSE_WIDE_CLUSTERS = 32


def sparse_waves(
    rows: int, num_kv_heads: int, cl: int, max_active_clusters: Mapping[int, int]
) -> float:
    """Waves of `rows` x `num_kv_heads` clusters of width `cl` (inf if
    that width is not schedulable)."""
    n = max_active_clusters.get(cl, 0)
    return float("inf") if n < 1 else -(-(rows * num_kv_heads) // n)


def sparse_cluster_size(
    rows: int,
    num_kv_heads: int,
    max_active_clusters: Mapping[int, int] | None = None,
    *,
    one_wave: bool = True,
) -> int:
    """Cluster width of the sparse decode for `rows` x `num_kv_heads`
    clusters. The preferred width is 16 below `SPARSE_WIDE_CLUSTERS`
    clusters, else 8. With `max_active_clusters` (width -> clusters the
    device holds at once, `cudaOccupancyMaxActiveClusters`):

    - `one_wave`: the first width, from the preferred one down, whose grid
      is co-resident; `ValueError` if none is (fail closed);
    - otherwise the grid only has to be schedulable: the width with the
      fewest waves, the wider one on a tie (more CTAs per row, shorter
      per-CTA page loop); `ValueError` if no width is schedulable."""
    need = rows * num_kv_heads
    if need <= 0:
        raise ValueError("sparse_decode: rows and KV heads must be positive")
    pref = 16 if need < SPARSE_WIDE_CLUSTERS else 8
    if max_active_clusters is None:
        return pref
    if one_wave:
        for cl in (c for c in SPARSE_CLUSTER_SIZES if c <= pref):
            if max_active_clusters.get(cl, 0) >= need:
                return cl
        raise ValueError(
            f"sparse_decode: {need} clusters do not fit in one wave at any width "
            f"(device holds {dict(max_active_clusters)})"
        )
    best = min(
        SPARSE_CLUSTER_SIZES,
        key=lambda c: (sparse_waves(rows, num_kv_heads, c, max_active_clusters), -c),
    )
    if sparse_waves(rows, num_kv_heads, best, max_active_clusters) == float("inf"):
        raise ValueError(
            "sparse_decode: no cluster width is schedulable "
            f"({dict(max_active_clusters)})"
        )
    return best


__all__ = [
    "MOE_N_SHARED_OVERRIDES",
    "MOE_VALID_N_SHARED",
    "expected_distinct_experts",
    "moe_fc2_blocks_per_cta",
    "moe_n_shared",
    "moe_n_shared_model",
    "moe_routed_ctas",
    "moe_step_cost",
    "sparse_cluster_size",
    "sparse_waves",
]
