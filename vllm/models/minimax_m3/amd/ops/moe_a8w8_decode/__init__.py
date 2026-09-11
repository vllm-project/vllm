# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL decode MoE for MiniMax-M3 MXFP8 weights on gfx950 (bf16 x, M <= 256).

Inline-sort routing at M <= 16,
``sort_decode`` above, gate/up GEMM + swiglu-OAI, down GEMM with atomic
accumulation, with the two GEMMs reading fp8 e4m3 weights (``gemm1.py``,
``gemm2.py``). The activations stay bf16 (a16w8): decode is weight-bandwidth
bound, so the MXFP8 activation quant of the aiter a8w8 path buys nothing there
and its two quant passes are dropped.

Weights are the tensors ``ModelOptMxFp8FusedMoE`` stores on the layer for the
AITER_MXFP8 backend (``shuffle_mxfp8_moe_weights``: gate/up-interleaved
``shuffle_weight`` + ``shuffle_scale``). The install hook wraps that method's
``apply`` and routes every ``M <= 256`` call to these kernels (the layer's
unfused shared experts, which vLLM keeps separate for ModelOpt MXFP8, stay with
the runner); everything else stays on aiter.
"""

import torch

from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.decode import (
    MAX_DECODE_TOKENS,
    supports_batch,
    supports_shapes,
)
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.hook import (
    maybe_run_shared_experts,
)

# Sort row block: 16 (one MFMA tile per block) below BM32_MIN_TOKENS, 32 from
# there (two tiles per unpacked W fragment: an expert with more than 16 rows,
# always the shared one, streams its weights half as often; inline sort reaches
# 32 tokens). MI355X chain sweep of 09-09: 32 loses at every M <= 256 (most
# experts have < 16 rows, so the second tile is padding work and the larger A
# staging halves the workgroups per CU), so it is off; kept for the lab.
BM32_MIN_TOKENS = MAX_DECODE_TOKENS + 1
# Wide-first layout (sorted mode): the shared expert, which every token routes
# to, is sorted first in gemm1.WIDE_BM-row blocks and run with the wide bodies
# of gemm1 / gemm2, so its weights stream once per WIDE_BM rows instead of once
# per 16. Chain sweep (MI355X, 09-09): -2 us at 48, -4 at 64, -7 at 96, -10 at
# 128, -18 at 256; nothing at 32.
WIDE_MIN_TOKENS = 40
_workspaces: dict = {}


def block_m_for(n_tokens: int) -> int:
    return 32 if n_tokens >= BM32_MIN_TOKENS else 16


def wide_for(n_tokens: int) -> bool:
    return n_tokens >= WIDE_MIN_TOKENS and n_tokens > block_m_for(n_tokens)


def _intermediate_workspace(
    device: torch.device, topk: int, intermediate_size: int, num_experts: int
) -> torch.Tensor:
    """gemm1 output, bf16 ``[rows, I]``: by pair (``pair * BM + row``) on the
    sort-free path, by sorted row on the sorted one; sized for the largest of
    the layouts of either block size at ``MAX_DECODE_TOKENS`` and allocated once
    per (device, topk, I, E) so HIP-graph capture records no allocation."""
    from vllm.models.minimax_m3.amd.ops.moe_a8w8_decode.gemm1 import WIDE_BM
    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.sort_decode import (
        max_sorted_rows,
        wide_layout_rows,
    )

    key = (
        device.index if device.index is not None else -1,
        topk,
        intermediate_size,
        num_experts,
    )
    ws = _workspaces.get(key)
    if ws is None:
        rows = max(
            [32 * topk * 32]
            + [
                max_sorted_rows(MAX_DECODE_TOKENS, num_experts, topk, bm)
                for bm in (16, 32)
            ]
            + [
                sum(wide_layout_rows(MAX_DECODE_TOKENS, num_experts, topk, bm, WIDE_BM))
                for bm in (16, 32)
            ]
        )
        ws = torch.empty((rows, intermediate_size), dtype=torch.bfloat16, device=device)
        _workspaces[key] = ws
    return ws


logger = init_logger(__name__)


def a16w8_decode_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    swiglu_alpha: float,
    swiglu_limit: float,
    fused_shared_expert: bool = True,
) -> torch.Tensor:
    """One MoE layer for ``M <= 256`` tokens on the FlyDSL a16w8 kernels.

    ``w13``/``w2`` (fp8) and their e8m0 scales are the layer tensors after
    ``shuffle_mxfp8_moe_weights``; only their data pointers are used.
    ``topk_ids`` / ``topk_weights`` are ``[M, topk]``. ``fused_shared_expert``
    says that the last expert is the fused shared one, routed by every token:
    the wide-first sort layout (``WIDE_MIN_TOKENS``) budgets its blocks from
    ``M`` and is only used then. Returns ``[M, hidden_size]`` bf16.
    """
    from vllm.models.minimax_m3.amd.ops.moe_a8w8_decode.host import (
        a16w8_gemm1,
        a16w8_gemm2,
    )
    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.sort_decode import (
        moe_sort_decode,
    )

    n_tokens = x.shape[0]
    assert n_tokens <= MAX_DECODE_TOKENS, n_tokens
    assert x.shape[1] == hidden_size and x.dtype == torch.bfloat16 and x.is_contiguous()
    topk = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()

    inter = _intermediate_workspace(x.device, topk, intermediate_size, num_experts)
    out = torch.empty((n_tokens, hidden_size), dtype=torch.bfloat16, device=x.device)
    bm = block_m_for(n_tokens)
    inline = n_tokens <= bm
    wide = fused_shared_expert and wide_for(n_tokens)
    if inline:
        sorted_ids = sorted_w = sorted_eids = num_valid = None
    else:
        from vllm.models.minimax_m3.amd.ops.moe_a8w8_decode.gemm1 import WIDE_BM

        sorted_ids, sorted_w, sorted_eids, num_valid = moe_sort_decode(
            topk_ids,
            topk_weights,
            num_experts,
            hidden_size,
            bm,
            out,
            wide_first=(num_experts - 1, WIDE_BM) if wide else None,
        )
    a16w8_gemm1(
        x_bf16=x,
        w1_fp8=w13,
        w1_scale_u8=w13_scale,
        inter_sorted_bf16=inter,
        n_tokens=n_tokens,
        NE=num_experts,
        D_HIDDEN=hidden_size,
        D_INTER=intermediate_size,
        topk=topk,
        alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        inline_sort=inline,
        topk_ids=topk_ids,
        zero_out=out,
        sorted_expert_ids=sorted_eids,
        num_valid_ids=num_valid,
        sorted_token_ids=sorted_ids,
        BM=bm,
        wide=wide,
    )
    a16w8_gemm2(
        inter_sorted_bf16=inter,
        w2_fp8=w2,
        w2_scale_u8=w2_scale,
        out_bf16=out,
        n_tokens=n_tokens,
        NE=num_experts,
        D_HIDDEN=hidden_size,
        D_INTER=intermediate_size,
        inline_sort=inline,
        topk=topk,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        sorted_expert_ids=sorted_eids,
        num_valid_ids=num_valid,
        sorted_token_ids=sorted_ids,
        sorted_weights=sorted_w,
        BM=bm,
        wide=wide,
    )
    return out


def is_mxfp8_aiter_layer(layer) -> bool:
    """True for a RoutedExperts layer on the ModelOpt MXFP8 method with the aiter
    (FlyDSL a8w8) backend, i.e. weights in the layout these kernels read."""
    qm = getattr(layer, "quant_method", None)
    backend = getattr(qm, "mxfp8_backend", None)
    return type(qm).__name__ == "ModelOptMxFp8FusedMoE" and (
        getattr(backend, "value", None) == "AITER_MXFP8"
    )


def _unsupported_reason(layer) -> str | None:
    """Static checks on a RoutedExperts layer; None when the fast path applies."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return "requires gfx950"
    if not is_mxfp8_aiter_layer(layer):
        qm = getattr(layer, "quant_method", None)
        return (
            f"quant method {type(qm).__name__} (backend "
            f"{getattr(qm, 'mxfp8_backend', None)}) "
            "is not ModelOpt MXFP8 on AITER_MXFP8"
        )
    if layer.activation != MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        return f"activation is {layer.activation}"
    if layer.swiglu_alpha is None or layer.swiglu_limit is None:
        return "swiglu_alpha / swiglu_limit not set"
    if layer.swiglu_beta not in (None, 1.0):
        return f"swiglu_beta {layer.swiglu_beta} != 1"
    if layer.apply_router_weight_on_input:
        return "apply_router_weight_on_input"
    if layer.expert_map is not None or layer.moe_config.use_ep:
        return "expert parallelism"
    if layer.moe_config.has_bias:
        return "expert bias"
    hidden = layer.moe_config.hidden_dim
    inter = layer.moe_config.intermediate_size_per_partition
    if not supports_shapes(hidden, inter):
        return f"shapes hidden={hidden} intermediate={inter} not tiled by the kernels"
    if layer.moe_config.hidden_dim_unpadded not in (None, hidden):
        return "padded hidden size"
    if layer.moe_config.intermediate_size_per_partition_unpadded not in (None, inter):
        return "padded intermediate size"
    if layer.w13_weight.dtype != torch.float8_e4m3fn:
        return f"w13 dtype {layer.w13_weight.dtype}"
    return None


def _fused_shared_expert(experts) -> bool:
    """True when the router appends the model's shared expert to every
    token's top-k as the last expert (aiter's fused shared experts), the
    routing the wide-first sort layout is built for. vLLM keeps the shared
    expert a separate module for ModelOpt MXFP8, so this is False there."""
    router = getattr(experts, "router", None)
    return getattr(router, "num_fused_shared_experts", 0) > 0


def install_decode_fast_path(experts, prefix: str = "") -> bool:
    """Route ``M <= 256`` calls of a MiniMax-M3 MXFP8 MoE layer to the FlyDSL
    a16w8 kernels. Returns True
    when installed."""
    layer = getattr(experts, "routed_experts", experts)
    try:
        reason = _unsupported_reason(layer)
    except Exception as exc:  # a layer/config shape this gate does not know
        reason = f"{type(exc).__name__}: {exc}"
    if reason is not None:
        logger.info_once(
            "M3 FlyDSL a16w8 decode MoE not used for %s: %s",
            prefix or "experts",
            reason,
        )
        return False
    qm = layer.quant_method
    if getattr(qm, "_m3_decode_fast_path", False):
        return True

    orig_apply = qm.apply
    hidden = layer.moe_config.hidden_dim
    inter = layer.moe_config.intermediate_size_per_partition
    alpha = float(layer.swiglu_alpha)
    limit = float(layer.swiglu_limit)
    fused_shared = _fused_shared_expert(experts)

    def apply(
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
        **kwargs,
    ):
        if kwargs or not supports_batch(x):
            return orig_apply(
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
                **kwargs,
            )
        maybe_run_shared_experts(shared_experts, shared_experts_input)
        return a16w8_decode_moe(
            x,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            hidden_size=hidden,
            intermediate_size=inter,
            num_experts=layer.w13_weight.shape[0],
            swiglu_alpha=alpha,
            swiglu_limit=limit,
            fused_shared_expert=fused_shared,
        )

    qm.apply = apply
    qm._m3_decode_fast_path = True
    logger.info_once(
        "M3 FlyDSL a16w8 decode MoE installed (MXFP8, M <= %d, shared expert %s)",
        MAX_DECODE_TOKENS,
        "fused" if fused_shared else "separate",
    )
    logger.debug("M3 FlyDSL a16w8 decode MoE installed for %s", prefix or "experts")
    return True


__all__ = [
    "BM32_MIN_TOKENS",
    "MAX_DECODE_TOKENS",
    "WIDE_MIN_TOKENS",
    "a16w8_decode_moe",
    "block_m_for",
    "install_decode_fast_path",
    "is_mxfp8_aiter_layer",
    "supports_batch",
    "supports_shapes",
]
