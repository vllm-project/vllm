# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer's SM100 NVFP4-weight / BF16-activation grouped MoE GEMMs.

The FlashInfer kernel decodes E2M1 * E4M3 weight tiles to BF16 and runs BF16
``tcgen05`` MMAs with fp32 accumulation and the per-expert fp32 global scale
applied to the accumulator. Its SwiGLU epilogue keeps gate/up in fp32 and
rounds ``silu(g) * u`` once (sigmoid via ``ex2.approx`` + ``rcp.approx``), the
rounding rule every Flint MoE path follows.
"""

from functools import cache

import torch


# ((mma_m, route_tile, mma_k), cluster_mn, raster_along_m) for GEMM1 / GEMM2,
# tuned on GB200 for E=256, H=2048, I=512, top-6.
def _tactic(num_tokens: int):
    if num_tokens >= 4096:
        return (((256, 128, 256), (2, 1), True), ((128, 128, 256), (2, 1), True))
    if num_tokens >= 2048:
        return (((256, 128, 256), (2, 1), True), ((256, 128, 256), (2, 1), True))
    if num_tokens >= 512:
        return (((128, 32, 256), (2, 1), True), ((128, 32, 256), (2, 1), True))
    return (((128, 8, 256), (2, 1), True), ((128, 8, 256), (2, 1), True))


def prepare_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ModelOpt NVFP4 expert weights -> FlashInfer W4A16 layouts.

    GEMM1 rows go from [gate; up] to [up; gate] interleaved in 64-row groups;
    block scales go to the 6D MMA layout. Returns (w1, w1_sf, w2, w2_sf).
    """

    def interleave(x: torch.Tensor) -> torch.Tensor:
        e, rows, cols = x.shape
        x = torch.cat([x[:, rows // 2 :], x[:, : rows // 2]], 1)
        x = x.view(e, 2, rows // 128, 64, cols).transpose(1, 2)
        return x.contiguous().view(e, rows, cols)

    def mma_scales(s: torch.Tensor) -> torch.Tensor:
        e, rows, groups = s.shape
        assert rows % 128 == 0 and groups % 4 == 0
        s = s.view(torch.uint8).view(e, rows // 128, 4, 32, groups // 4, 4)
        s = s.permute(0, 1, 4, 3, 2, 5).contiguous().view(torch.float8_e4m3fn)
        # Physical (E, m_tiles, k_tiles, 32, 4, 4) as logical
        # (32, 4, m_tiles, 4, k_tiles, E).
        return s.permute(3, 4, 1, 5, 2, 0)

    w1 = interleave(w13)
    w1_sf = mma_scales(interleave(w13_scale.view(torch.uint8)))
    return w1, w1_sf, w2.contiguous(), mma_scales(w2_scale)


def grouped_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w1_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2: torch.Tensor,
    w2_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Routed expert outputs in permuted order (unweighted, BF16).

    Returns (y [route_slots, H], expanded_idx_to_permuted_idx [M, TOPK]); the
    caller does the weighted top-k reduction.
    """
    from flashinfer.fused_moe.cute_dsl.blackwell import moe_w4a16 as lmod
    from flashinfer.fused_moe.cute_dsl.moe_utils import (
        allocate_moe_sort_buffers,
        get_max_num_permuted_tokens,
        moe_permute,
        moe_sort,
    )
    from flashinfer.tllm_enums import (
        DEFAULT_SWIGLU_ALPHA,
        DEFAULT_SWIGLU_BETA,
        DEFAULT_SWIGLU_LIMIT,
        ActivationType,
    )

    num_tokens, hidden = x.shape
    top_k = topk_ids.shape[1]
    num_experts = w1.shape[0]
    inter = w2.shape[2] * 2
    gemm1_tactic, gemm2_tactic = _tactic(num_tokens)
    route_tile = gemm1_tactic[0][1]
    route_slots = get_max_num_permuted_tokens(
        num_tokens, top_k, num_experts, route_tile
    )
    sort_bufs = allocate_moe_sort_buffers(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_experts,
        tile_tokens_dim=route_tile,
        device=x.device,
    )
    (
        tile_to_expert,
        tile_to_mn_limit,
        expanded_to_permuted,
        permuted_to_expanded,
        _,
        num_tiles,
    ) = moe_sort(
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=0,
        num_local_experts=num_experts,
        tile_tokens_dim=route_tile,
        enable_pdl=True,
        **sort_bufs,
    )
    num_route_tiles = route_slots // route_tile
    tile_to_expert = tile_to_expert[:num_route_tiles]
    tile_to_mn_limit = tile_to_mn_limit[:num_route_tiles]
    permuted_to_expanded = permuted_to_expanded[:route_slots]
    hidden_ws = torch.empty(route_slots, hidden, dtype=x.dtype, device=x.device)
    inter_ws = torch.empty(route_slots, inter, dtype=x.dtype, device=x.device)
    moe_permute(
        input=x,
        permuted_output=hidden_ws,
        tile_idx_to_mn_limit=tile_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_to_expanded,
        num_non_exiting_tiles=num_tiles,
        max_num_permuted_tokens=route_slots,
        top_k=top_k,
        tile_size=route_tile,
        enable_pdl=True,
    )
    common = dict(
        tile_idx_to_expert_idx=tile_to_expert,
        tile_idx_to_mn_limit=tile_to_mn_limit,
        num_non_exiting_tiles=num_tiles,
        num_local_experts=num_experts,
        swiglu_alpha=DEFAULT_SWIGLU_ALPHA,
        swiglu_beta=DEFAULT_SWIGLU_BETA,
        swiglu_limit=DEFAULT_SWIGLU_LIMIT,
        situ_beta=None,
        situ_linear_beta=None,
        use_fused_finalize=False,
        permuted_idx_to_expanded_idx=None,
        token_final_scales=None,
        enable_pdl=True,
    )
    lmod._run_grouped_gemm(
        weight=w1,
        weight_sf=w1_sf,
        activations=hidden_ws,
        alpha=w1_alpha,
        output=inter_ws,
        activation_type=ActivationType.Swiglu,
        tactic=gemm1_tactic,
        **common,
    )
    # GEMM2 writes over the permuted activations, which are dead by now.
    lmod._run_grouped_gemm(
        weight=w2,
        weight_sf=w2_sf,
        activations=inter_ws,
        alpha=w2_alpha,
        output=hidden_ws,
        activation_type=None,
        tactic=gemm2_tactic,
        **common,
    )
    return hidden_ws, expanded_to_permuted


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _compile_key(num_tokens: int, top_k: int, num_experts: int, gemm1_rows: int):
    """The parts of the kernel cache keys that vary with num_tokens (tactics
    and whether GEMM1 uses the CLC scheduler, as in FlashInfer's launcher)."""
    from flashinfer.cute_dsl.utils import get_max_active_clusters
    from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_permuted_tokens

    gemm1, gemm2 = _tactic(num_tokens)
    (mma_m, route_tile, _), cluster, _ = gemm1
    slots = get_max_num_permuted_tokens(num_tokens, top_k, num_experts, route_tile)
    cta_m = mma_m // (2 if mma_m == 256 else 1)
    clusters = _cdiv(_cdiv(gemm1_rows, cta_m), cluster[0]) * _cdiv(
        _cdiv(slots, route_tile), cluster[1]
    )
    clc = clusters > get_max_active_clusters(cluster[0] * cluster[1])
    return gemm1, gemm2, clc


@cache
def _variants(top_k, num_experts, gemm1_rows, min_tokens, max_tokens) -> dict:
    """Smallest token count of each kernel cache key in [min, max]."""
    reps: dict = {}
    for n in range(min_tokens, max_tokens + 1):
        reps.setdefault(_compile_key(n, top_k, num_experts, gemm1_rows), n)
    return reps


_WARMED: set[tuple] = set()


def warmup(weights, top_k: int, min_tokens: int, max_tokens: int) -> None:
    """Compile every grouped-GEMM kernel variant used for min..max tokens.

    CuTe kernels compile on first use, which must not happen inside CUDA graph
    capture; the compile caches are process-wide, so each variant runs once.
    """
    num_experts, gemm1_rows = weights.w1.shape[:2]
    hidden = weights.w2.shape[1]
    reps = _variants(top_k, num_experts, gemm1_rows, min_tokens, max_tokens)
    for key, n in reps.items():
        if key in _WARMED:
            continue
        x = torch.zeros(n, hidden, dtype=torch.bfloat16, device=weights.w1.device)
        ids = torch.randint(0, num_experts, (n, top_k), device=x.device)
        tw = torch.full((n, top_k), 1.0 / top_k, device=x.device)
        grouped_moe(
            x,
            ids.int(),
            tw,
            weights.w1,
            weights.w1_sf,
            weights.w1_alpha,
            weights.w2,
            weights.w2_sf,
            weights.w2_alpha,
        )
        _WARMED.add(key)
    torch.accelerator.synchronize()
