# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark MXFP4 MoE on the native ``[T, top_k]`` routes, one output per route.

The post-expert RMSNorm is per route, so the experts must return the unreduced
``[T, top_k, H]`` outputs. The expanded path gets them by repeating the input
``top_k`` times and running AITER on ``T * top_k`` top-1 routes. This runs
AITER on the native routes instead: ``_metadata_transform`` substitutes the
expanded call's tuned config, so both paths launch the same stage1/stage2
tiles, and ``_stage2_override`` makes stage2 write each route's output to its
own ``[T, top_k, H]`` row instead of reducing them. Under EP, routes to
experts this rank does not own are left unwritten.

Tuned AITER rows for this path are looked up under the expanded call's key:
``get_padded_M(T * top_k)`` tokens with topk 1 (2 under EP, for the masked
slot the expanded call appends).

The hooks and stage2 wrappers used here are AITER internals; with an AITER
that lacks any of them, every batch keeps the expanded path.
"""

import functools
import inspect
from typing import Any

import torch


def _fm():
    import aiter.fused_moe as fm

    return fm


@functools.cache
def _aiter_has_per_route_hooks() -> bool:
    """Whether the installed AITER has every internal this module calls."""
    try:
        from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2

        fm = _fm()
        impl = inspect.signature(fm._fused_moe_impl).parameters
        v1 = inspect.signature(fm._get_flydsl_moe_kernels().flydsl_moe_stage2)
        v2 = inspect.signature(mxfp4_moe_gemm2)
        for name in (
            "_flydsl_stage2_wrapper",
            "_flydsl_v2_stage2_wrapper",
            "_mxfp4_scale_u8",
            "parse_flydsl_v2_gemm2_kernel",
        ):
            getattr(fm, name)
    except (ImportError, AttributeError):
        return False
    return (
        "_metadata_transform" in impl
        and "_stage2_override" in impl
        and "return_per_slot" in v1.parameters
        and "epilog" in v2.parameters
    )


@functools.cache
def _expanded_metadata(
    num_tokens_key: int,
    top_k: int,
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    w_dtype: torch.dtype,
    is_ep: bool,
    is_shuffled: bool,
):
    """The metadata AITER selects for the expanded call of the batch sizes
    that share ``num_tokens_key = get_padded_M(T * top_k)``."""
    from aiter import ActivationType, QuantType

    fm = _fm()
    num_experts, model_dim, inter_dim = fm.get_inter_dim(w1_shape, w2_shape)
    q_dtype_a = fm.resolve_activation_dtype(
        QuantType.per_1x32,
        w_dtype,
        activation=ActivationType.Silu,
        M=num_tokens_key,
    )
    # Mirrors the lookup in aiter.fused_moe._fused_moe_impl for vLLM's call.
    return fm.get_2stage_cfgs(
        num_tokens_key,
        model_dim,
        inter_dim,
        num_experts,
        1 + int(is_ep),
        torch.bfloat16,
        q_dtype_a,
        w_dtype,
        QuantType.per_1x32,
        inter_dim != w1_shape[1],
        ActivationType.Silu,
        False,
        0,
        0,
        is_shuffled,
        fm.GateMode.SEPARATED.value,
        is_ep=is_ep,
        swiglu_limit=0.0,
        opus_weights_shuffled=is_shuffled,
        input_dtype=torch.bfloat16,
    )


def _per_slot_stage2_kind(metadata) -> str | None:
    """``"v1"``/``"v2"`` for the FlyDSL stage2 families that can write one row
    per route with the tuned tiles, else ``None``."""
    fm = _fm()
    if (
        metadata.run_1stage
        or metadata.full_impl is not None
        or metadata.stage2 is None
        or metadata.output_aux
        # Split-K stage1 reads the bf16 input rather than the quantized one.
        or metadata.ksplit > 1
    ):
        return None
    func = getattr(metadata.stage2, "func", None)
    kernel_name = metadata.stage2.keywords.get("kernelName", "")
    if func is fm._flydsl_v2_stage2_wrapper:
        cfg = fm.parse_flydsl_v2_gemm2_kernel(kernel_name)
        fp4 = cfg is not None and cfg["a_dtype"] == cfg["b_dtype"] == "fp4"
        return "v2" if fp4 else None
    if func is fm._flydsl_stage2_wrapper:
        params = fm._get_flydsl_moe_kernels().get_flydsl_kernel_params(kernel_name)
        fp4 = params is not None and params["a_dtype"] == params["b_dtype"] == "fp4"
        return "v1" if fp4 else None
    return None


def supports_route_native(
    num_tokens: int,
    top_k: int,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    is_ep: bool,
) -> bool:
    """Whether the expanded call's kernels for ``num_tokens`` can run on the
    native routes. A config lookup only; nothing is compiled or launched."""
    return (
        _aiter_has_per_route_hooks()
        and _per_slot_stage2_kind(_metadata(num_tokens, top_k, w1, w2, is_ep))
        is not None
    )


def _metadata(
    num_tokens: int,
    top_k: int,
    w1: torch.Tensor,
    w2: torch.Tensor,
    is_ep: bool,
):
    return _expanded_metadata(
        _fm().get_padded_M(num_tokens * top_k),
        top_k,
        tuple(w1.shape),
        tuple(w2.shape),
        w1.dtype,
        is_ep,
        getattr(w1, "is_shuffled", False) or getattr(w2, "is_shuffled", False),
    )


def _v1_per_slot_stage2(
    route_out: torch.Tensor,
    stage2_args: tuple,
    stage2_kwargs: dict[str, Any],
    kernelName: str,
    inter_dim_pad: int = 0,
    model_dim_pad: int = 0,
    **_kwargs,
) -> None:
    fm = _fm()
    inter_states, _, w2, sorted_ids, sorted_expert_ids, num_valid_ids, _, topk = (
        stage2_args
    )
    moe_kernels = fm._get_flydsl_moe_kernels()
    params = moe_kernels.get_flydsl_kernel_params(kernelName)
    moe_kernels.flydsl_moe_stage2(
        inter_states=inter_states,
        w2=w2,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=route_out,
        topk=topk,
        tile_m=params["tile_m"],
        tile_n=params["tile_n"],
        tile_k=params["tile_k"],
        a_dtype=params["a_dtype"],
        b_dtype=params["b_dtype"],
        out_dtype=params["out_dtype"],
        w2_scale=stage2_kwargs["w2_scale"],
        a2_scale=stage2_kwargs["a2_scale"],
        sorted_weights=stage2_kwargs["sorted_weights"],
        sort_block_m=params.get("sort_block_m", 0),
        waves_per_eu=params.get("waves_per_eu", None),
        use_async_copy=params.get("use_async_copy", False),
        cu_num_mul=params.get("cu_num_mul", 1),
        b_nt=params.get("b_nt", 0),
        persist=params.get("persist", None),
        inter_dim_pad=inter_dim_pad,
        model_dim_pad=model_dim_pad,
        xcd_swizzle=params.get("xcd_swizzle", 0),
        return_per_slot=True,
    )


def _v2_per_slot_stage2(
    route_out: torch.Tensor,
    stage2_args: tuple,
    stage2_kwargs: dict[str, Any],
    kernelName: str,
    model_dim: int,
    inter_dim: int,
    num_experts: int,
    **_kwargs,
) -> None:
    from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2

    fm = _fm()
    inter_states, _, w2, sorted_ids, sorted_expert_ids, num_valid_ids, _, topk = (
        stage2_args
    )
    cfg = fm.parse_flydsl_v2_gemm2_kernel(kernelName)
    block_m = stage2_kwargs.get("block_m")
    mxfp4_moe_gemm2(
        inter_sorted_quant=fm._mxfp4_scale_u8(inter_states),
        inter_sorted_shuffled_scale=fm._mxfp4_scale_u8(stage2_kwargs["a2_scale"]),
        w2_u8=fm._mxfp4_scale_u8(w2),
        w2_scale_u8=fm._mxfp4_scale_u8(stage2_kwargs["w2_scale"]),
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=num_valid_ids,
        sorted_token_ids=sorted_ids,
        sorted_weights=stage2_kwargs["sorted_weights"],
        out=route_out,
        M_logical=route_out.shape[0],
        max_sorted=inter_states.shape[0],
        NE=num_experts,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        topk=topk,
        BM=cfg["tile_m"],
        BN=cfg["tile_n"],
        BK=cfg["tile_k"],
        use_nt=cfg["use_nt"],
        a_dtype=cfg["a_dtype"],
        b_dtype=cfg["b_dtype"],
        # The "reduce" epilog writes one row per (token, slot); the reduction
        # that would follow it is skipped.
        epilog="reduce",
        SBM=cfg["sort_block_m"] or (int(block_m) if block_m else cfg["tile_m"]),
        persist=cfg["persist"],
        g2_bf16_lds=cfg["bf16_lds"],
        g2_spart=cfg["spart"],
        out_dtype="bf16",
        is_ep=stage2_kwargs.get("expert_mask") is not None,
    )


def _per_slot_stage2(
    *,
    ordinary_stage2,
    stage2_args: tuple,
    stage2_kwargs: dict[str, Any],
    kind: str,
    route_out: torch.Tensor,
) -> torch.Tensor:
    fn = _v1_per_slot_stage2 if kind == "v1" else _v2_per_slot_stage2
    fn(route_out, stage2_args, stage2_kwargs, **ordinary_stage2.keywords)
    return route_out


def route_native_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    unit_weights: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    route_out: torch.Tensor,
    *,
    expert_mask: torch.Tensor | None = None,
    moe_sorting_dispatch_policy: int = 0,
) -> torch.Tensor:
    """Write the unweighted MXFP4 SiLU expert output of every route to
    ``route_out[T, top_k, H]``.

    Args:
        hidden_states: ``[T, H]`` bf16 expert input, one row per token.
        w1: AITER-shuffled MXFP4 gate/up weights.
        w2: AITER-shuffled MXFP4 down weights.
        topk_ids: ``[T, top_k]`` int32 global expert ids.
        unit_weights: ``[T, top_k]`` fp32 ones.
        w1_scale: ``w1`` block scales.
        w2_scale: ``w2`` block scales.
        route_out: ``[T, top_k, H]`` bf16 destination.
        expert_mask: EP only, AITER's ``[global_E + 1]`` ownership mask. Rows
            of routes whose expert is masked out are not written.
        moe_sorting_dispatch_policy: AITER sorting dispatch policy.

    """
    from aiter import ActivationType, QuantType

    num_tokens, top_k = topk_ids.shape
    metadata = _metadata(num_tokens, top_k, w1, w2, expert_mask is not None)
    kind = _per_slot_stage2_kind(metadata)
    if kind is None:
        raise NotImplementedError(
            f"No per-route AITER stage2 for {num_tokens} tokens: {metadata.stage2}"
        )
    assert route_out.shape == (num_tokens, top_k, w2.shape[1])
    assert route_out.is_contiguous() and route_out.dtype == torch.bfloat16
    _fm()._fused_moe_impl(
        hidden_states,
        w1,
        w2,
        unit_weights,
        topk_ids,
        expert_mask=expert_mask,
        activation=ActivationType.Silu.value,
        quant_type=QuantType.per_1x32.value,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        moe_sorting_dispatch_policy=moe_sorting_dispatch_policy,
        dtype=torch.bfloat16,
        swiglu_limit=0.0,
        _metadata_transform=lambda _: metadata,
        _stage2_override=functools.partial(
            _per_slot_stage2, kind=kind, route_out=route_out
        ),
    )
    return route_out
