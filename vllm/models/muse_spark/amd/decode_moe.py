# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark decode-size MoE layer on native routes (ROCm gfx950, EP AITER MXFP4).

The AITER MoE path repeats each token's pre-expert-normed row once per route and
runs the experts on `T * top_k` rows. Here a call of at most `MAX_TOKENS` tokens
runs the whole layer on the layer's own weights:

* router logits and pooler, then one prep kernel: router split-K reduce, sigmoid
  top-k routing and the pre-expert RMSNorm, one row per token;
* the experts on the `[T, top_k]` routes: moe_sorting, the stage-1 MXFP4 quant of
  the T rows into sorted layout, AITER's FlyDSL stages with a per-slot stage 2;
* the fused post-expert finalize, masking the routes of other ranks' experts;
* the layer's reduction (all-reduce, or reduce-scatter on a token-sharded step)
  and the unpooler.

Anything else runs the layer as before; the paths share every tensor at the
layer boundary. Every rank takes the same path: the conditions read only the
layer's configuration and the step's token count.
"""

import functools
import inspect
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.models.muse_spark.common.layers import MuseSparkRMSNorm
from vllm.models.muse_spark.common.ops.fused_finalize import (
    fused_post_expert_finalize,
)

from .muse_moe_runner import MuseSparkAiterExperts, MuseSparkQuarkOCPMXFusedMoE
from .ops.decode_moe import decode_moe_prep, supports_decode_moe_prep

if TYPE_CHECKING:
    from .model import MuseSparkMoE, TokenShard

logger = init_logger(__name__)

# Largest MoE call (tokens) that takes this path.
MAX_TOKENS = 2048


class StageKernels(NamedTuple):
    """The AITER FlyDSL expert stages of one `[T, top_k]` routes shape."""

    stage1: str
    stage2: dict[str, Any]
    block_m: int


@functools.cache
def _aiter_kernels_available() -> bool:
    try:
        from aiter.fused_moe import (  # noqa: F401
            _flydsl_stage1_wrapper,
            get_2stage_cfgs,
            moe_sorting,
        )
        from aiter.ops.flydsl import moe_kernels  # noqa: F401
        from aiter.ops.quant import (  # noqa: F401
            fused_dynamic_mx_quant_moe_sort_hip,
        )
    except ImportError:
        logger.warning_once(
            "Muse Spark decode MoE needs a newer AITER; using the per-route path."
        )
        return False
    return True


class MuseSparkDecodeMoE:
    """One MuseSparkMoE layer's decode path (`create`: None when the layer's
    configuration does not support it)."""

    @classmethod
    def create(cls, moe: "MuseSparkMoE") -> "MuseSparkDecodeMoE | None":
        from .model import _supports_router_bf16x3_device as on_gfx950

        routed = moe.experts.routed_experts
        parallel = routed.moe_config.moe_parallel_config
        pre_norm, post_norm = moe.pre_expert_norm, moe.post_expert_norm
        if not (
            on_gfx950()
            and not moe.is_sequence_parallel
            and moe.pooler is not None
            and moe.unpooler is not None
            and pre_norm is not None
            and pre_norm.variance_size_override is None
            and isinstance(post_norm, MuseSparkRMSNorm)
            and post_norm.variance_size_override is None
            and moe.moe_router.router_score_normalization == "sigmoid"
            and isinstance(routed.quant_method, MuseSparkQuarkOCPMXFusedMoE)
            and parallel.use_ep
            and parallel.tp_size == 1
            and supports_decode_moe_prep(moe.num_experts, moe.top_k)
            and _aiter_kernels_available()
        ):
            return None
        logger.info_once(
            "Muse Spark decode MoE on native routes enabled up to %d tokens.",
            MAX_TOKENS,
        )
        return cls()

    def kernels(self, moe: "MuseSparkMoE", num_tokens: int) -> StageKernels | None:
        """The expert stages of a call of `num_tokens` tokens (the whole step's,
        on a token-sharded step), or None when the call takes the per-route
        path."""
        if not 0 < num_tokens <= MAX_TOKENS:
            return None
        routed = moe.experts.routed_experts
        kernel = routed.quant_method.moe_kernel
        if kernel is None or not isinstance(
            kernel.fused_experts, MuseSparkAiterExperts
        ):
            return None
        local, inter2 = routed.w13_weight.shape[:2]
        return _stage_kernels(
            num_tokens, moe.top_k, local, routed.moe_config.hidden_dim, inter2 // 2
        )

    def __call__(
        self,
        moe: "MuseSparkMoE",
        hidden_states: torch.Tensor,
        kernels: StageKernels,
        shard: "TokenShard | None" = None,
    ) -> torch.Tensor:
        router, pooler, unpooler = moe.moe_router, moe.pooler, moe.unpooler
        norm, post_norm = moe.pre_expert_norm, moe.post_expert_norm
        assert pooler is not None and unpooler is not None
        assert norm is not None and post_norm is not None
        if shard is not None:
            router_logits = shard.gather(moe._router_logits(hidden_states))
            pooled = shard.gather(pooler(hidden_states)[0])
        else:
            router_logits = moe._router_logits(hidden_states, reduce_splits=False)
            pooled = pooler(hidden_states)[0]
        topk_weights, topk_ids, normed = decode_moe_prep(
            router_logits.contiguous(),
            router.expert_bias,
            router.top_k,
            router.route_scale,
            router.router_output_multiplier,
            router.router_score_division_eps,
            pooled.contiguous(),
            norm.weight if norm.has_weight else None,
            norm.variance_epsilon,
            norm.fused_gain_center,
            norm.applies_weight_before_norm and norm.has_weight,
        )
        if router.capture_fn is not None:
            router.capture_fn(topk_ids)

        routed = moe.experts.routed_experts
        expert_mask, local_index = routed.expert_map, routed._expert_map
        assert expert_mask is not None and local_index is not None
        out = native_routes_moe(
            normed,
            topk_weights,
            topk_ids,
            routed.w13_weight,
            routed.w2_weight,
            routed.quant_method.moe_kernel.fused_experts.quant_config,
            expert_mask,
            local_index,
            kernels,
            post_norm,
        )
        if shard is not None:
            out = shard.reduce_scatter(out)
        else:
            out = tensor_model_parallel_all_reduce(out)
        return unpooler(out)[0]


def _stage_kernels(
    num_tokens: int, top_k: int, local_experts: int, hidden: int, inter: int
) -> StageKernels | None:
    """The expert stages AITER picks for `[T, top_k]` routes when this path
    can launch them, else None."""
    from aiter.fused_moe import get_padded_M

    return _stage_kernels_for(
        get_padded_M(num_tokens), top_k, local_experts, hidden, inter
    )


@functools.cache
def _stage_kernels_for(
    padded_tokens: int, top_k: int, local_experts: int, hidden: int, inter: int
) -> StageKernels | None:
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import get_2stage_cfgs
    from aiter.ops.flydsl.moe_common import GateMode

    try:
        metadata = get_2stage_cfgs(
            token=padded_tokens,
            model_dim=hidden,
            inter_dim=inter,
            expert=local_experts,
            topk=top_k,
            dtype=torch.bfloat16,
            q_dtype_a=dtypes.fp4x2,
            q_dtype_w=dtypes.fp4x2,
            q_type=QuantType.per_1x32,
            use_g1u1=True,
            activation=ActivationType.Silu,
            doweight_stage1=False,
            hidden_pad=0,
            intermediate_pad=0,
            is_shuffled=True,
            gate_mode=GateMode.SEPARATED,
            is_ep=True,
            opus_weights_shuffled=True,
            # Stage 1 reads rows quantized here, with their scales: rules out
            # configs that quantize their own bf16 input.
            has_activation_scales=True,
            input_dtype=dtypes.fp4x2,
        )
    except TypeError:
        metadata = None
    kernels = None if metadata is None else _flydsl_stage_kernels(metadata)
    if kernels is None:
        logger.warning_once(
            "Muse Spark decode MoE cannot launch AITER's expert stages for "
            "[%d, %d] routes; those calls use the per-route path.",
            padded_tokens,
            top_k,
        )
    return kernels


def _flydsl_stage_kernels(metadata: Any) -> StageKernels | None:
    """The stages of AITER's MoE `metadata` when both are FlyDSL MXFP4 kernels
    with a bf16 stage-1 output and the installed AITER takes the arguments
    `_native_routes_experts` passes, else None."""
    from aiter.fused_moe import _flydsl_stage1_wrapper
    from aiter.ops.flydsl import moe_kernels as fly

    kn1 = getattr(metadata.stage1, "keywords", {}).get("kernelName", "")
    kn2 = getattr(metadata.stage2, "keywords", {}).get("kernelName", "")
    # A fused-quant stage 1 returns quantized rows and scales, not bf16 rows.
    if (
        getattr(metadata, "fuse_quant", "")
        or not kn1.startswith("flydsl_moe1_afp4_wfp4_bf16")
        or not kn2.startswith("flydsl_moe2_afp4_wfp4_bf16")
    ):
        return None
    p = fly.get_flydsl_kernel_params(kn2)
    stage2 = dict(
        tile_m=p["tile_m"],
        tile_n=p["tile_n"],
        tile_k=p["tile_k"],
        a_dtype=p["a_dtype"],
        b_dtype=p["b_dtype"],
        out_dtype=p["out_dtype"],
        mode=p.get("mode", "atomic"),
        sort_block_m=p.get("sort_block_m", 0),
        waves_per_eu=p.get("waves_per_eu", None),
        use_async_copy=p.get("use_async_copy", False),
        cu_num_mul=p.get("cu_num_mul", 1),
        b_nt=p.get("b_nt", 0),
        persist=p.get("persist", None),
        xcd_swizzle=p.get("xcd_swizzle", 0),
        return_per_slot=True,
    )
    try:
        inspect.signature(_flydsl_stage1_wrapper).bind_partial(
            kernelName=kn1, w1_scale=None, a1_scale=None
        )
        inspect.signature(fly.flydsl_moe_stage2).bind_partial(
            w2_scale=None, a2_scale=None, **stage2
        )
    except TypeError:
        return None
    return StageKernels(kn1, stage2, int(metadata.block_m))


def native_routes_moe(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    expert_mask: torch.Tensor,
    local_index: torch.Tensor,
    kernels: StageKernels,
    post_expert_norm: MuseSparkRMSNorm,
) -> torch.Tensor:
    """This rank's experts' finalized MoE output `[T, H]` for the T normed bf16
    rows (unreduced across ranks; `local_index`: the `[E]` global-to-local
    expert map, -1 for other ranks' experts)."""
    routes = _native_routes_experts(
        hidden_states,
        topk_ids,
        w1,
        w2,
        quant_config.w1_scale,
        quant_config.w2_scale,
        expert_mask,
        kernels,
    )
    norm = post_expert_norm
    out = torch.empty(
        (routes.shape[0], routes.shape[-1]), dtype=routes.dtype, device=routes.device
    )
    fused_post_expert_finalize(
        routes,
        norm.weight if norm.has_weight else None,
        topk_weights,
        norm.variance_epsilon,
        norm.fused_gain_center,
        norm.applies_weight_before_norm,
        out,
        topk_ids=topk_ids,
        expert_map=local_index,
    )
    return out


def _native_routes_experts(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    expert_mask: torch.Tensor,
    kernels: StageKernels,
) -> torch.Tensor:
    """Unweighted per-(token, slot) MXFP4 expert outputs `[T, top_k, H]` of the T
    bf16 rows `hidden_states`.

    The FlyDSL stage kernels and quant rule of AITER's fused_moe on the routes'
    repeated rows, on the `[T, top_k]` routes: one moe_sorting; the stage-1 quant
    writes each token's row once per local route in sorted layout; stage 1 writes
    rows only for the local routes; stage 2 writes per-slot rows
    (`return_per_slot`). Rows of other ranks' routes are left unwritten.
    """
    from aiter.fused_moe import _flydsl_stage1_wrapper, moe_sorting
    from aiter.ops.flydsl import moe_kernels as fly
    from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort_hip

    num_tokens, top_k = topk_ids.shape
    hidden = hidden_states.shape[1]
    inter = w1.shape[1] // 2
    block_m = kernels.block_m
    e8m0, fp4x2 = torch.float8_e8m0fnu, torch.float4_e2m1fn_x2
    device = hidden_states.device

    def _sorted_scales(sorted_ids: torch.Tensor, cols: int) -> torch.Tensor:
        rows = (sorted_ids.shape[0] + 31) // 32 * 32
        return torch.empty((rows, (cols // 32 + 7) // 8 * 8), dtype=e8m0, device=device)

    unused_weights = torch.empty(
        (num_tokens, top_k), dtype=torch.float32, device=device
    )
    sorted_ids, _, sorted_eids, num_valid, _ = moe_sorting(
        topk_ids,
        unused_weights,
        expert_mask.numel(),
        hidden,
        torch.bfloat16,
        block_m,
        expert_mask,
        None,
        rocm_aiter_ops.get_moe_dispatch_policy(),
        accumulate=False,
    )[:5]
    a1 = torch.empty((num_tokens, hidden // 2), dtype=fp4x2, device=device)
    a1_scale = _sorted_scales(sorted_ids, hidden)
    fused_dynamic_mx_quant_moe_sort_hip(
        a1,
        a1_scale,
        hidden_states,
        sorted_ids,
        num_valid,
        num_tokens,
        block_m,
        num_tokens * top_k,
    )
    a2_bf16 = _flydsl_stage1_wrapper(
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_eids,
        num_valid,
        None,
        top_k,
        kernelName=kernels.stage1,
        w1_scale=w1_scale.view(e8m0),
        a1_scale=a1_scale,
    )
    a2 = torch.empty((num_tokens * top_k, inter // 2), dtype=fp4x2, device=device)
    a2_scale = _sorted_scales(sorted_ids, inter)
    fused_dynamic_mx_quant_moe_sort_hip(
        a2,
        a2_scale,
        a2_bf16.view(num_tokens * top_k, inter),
        sorted_ids,
        num_valid,
        num_tokens,
        block_m,
        num_tokens * top_k,
    )
    del a2_bf16
    out = fly.flydsl_moe_stage2(
        a2.view(num_tokens, top_k, -1),
        w2,
        sorted_ids,
        sorted_eids,
        num_valid,
        None,
        top_k,
        w2_scale=w2_scale.view(e8m0),
        a2_scale=a2_scale,
        **kernels.stage2,
    )
    return out.view(num_tokens, top_k, hidden)
