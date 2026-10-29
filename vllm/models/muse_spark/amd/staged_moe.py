# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark MXFP4 MoE through AITER's two-stage API.

The post-expert RMSNorm is per route, so the experts must return the unreduced
``[T, top_k, H]`` outputs. The stock path gets them by repeating the input
``top_k`` times and running AITER on ``T * top_k`` top-1 routes. With AITER's
``fused_moe_stage1`` / ``fused_moe_stage2`` this module instead

* runs the experts on the native ``[T, top_k]`` routes, one input row per
  token, with stage 2 writing every route to its own row (``per_slot``). The
  kernels are the ones AITER selects for the repeated-input call
  (``config_shape``), so the route outputs are unchanged. Under EP, routes to
  experts this rank does not own are left unwritten.
* at TP > 1, for large eager batches, also runs stage 2 column-sharded: it
  all-gathers stage 2's MXFP4 input and E8M0 scales along the intermediate dim
  (each rank's slice is whole 32-element quant groups, so the gathered tensors
  are exactly the full-width quantization) and multiplies them by a second
  copy of ``w2`` that holds this rank's ``H / tp`` output rows. That gives
  every route's exact output on those columns, so the post-expert norm needs
  only a ``[T, top_k]`` sum-of-squares all-reduce and an all-gather of the
  finalized columns (see ``ops/column_sharded_finalize.py``) instead of a
  reduce-scatter of ``[T, top_k, H]`` partial sums. The second ``w2`` is as
  large as the rank's regular ``w2`` shard.
"""

import inspect
from dataclasses import dataclass
from functools import cache

import torch

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

_MX_GROUP = 32
# AITER's E8M0 activation-scale layout stores each 32-row x 8-group tile as
# one contiguous 256-byte block, tiles of a 32-row band in group order.
_SCALE_TILE_ROWS = 32
_SCALE_TILE_GROUPS = 8
# Per-rank slices of H and of the intermediate dim must be multiples of this:
# whole scale tiles, and whole tiles of AITER's weight and scale shuffles.
_SHARD_ALIGN = 256


@cache
def aiter_has_staged_moe() -> bool:
    """Whether the installed AITER has the two-stage MoE API."""
    try:
        import aiter.fused_moe as fm
    except ImportError:
        return False
    stage1 = getattr(fm, "fused_moe_stage1", None)
    return (
        stage1 is not None
        and hasattr(fm, "fused_moe_stage2")
        and hasattr(fm, "supports_stage2_overrides")
        and "config_shape" in inspect.signature(stage1).parameters
    )


def _config_shape(num_tokens: int, top_k: int, is_ep: bool) -> tuple[int, int]:
    """``(tokens, topk)`` of the repeated-input call, whose tuned AITER rows
    the native-route call runs: ``T * top_k`` top-1 routes, plus under EP the
    always-masked slot AITER's EP convention appends to every row."""
    return num_tokens * top_k, 1 + int(is_ep)


@cache
def _stage2_overridable(
    config_shape: tuple[int, int],
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    w_dtype: torch.dtype,
    is_ep: bool,
    is_shuffled: bool,
) -> bool:
    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType

    num_tokens, topk = config_shape
    num_experts, model_dim, inter_dim = fm.get_inter_dim(w1_shape, w2_shape)
    padded = fm.get_padded_M(num_tokens)
    q_dtype_a = fm.resolve_activation_dtype(
        QuantType.per_1x32, w_dtype, activation=ActivationType.Silu, M=padded
    )
    # The lookup fused_moe_stage1 does for the calls below.
    metadata = fm.get_2stage_cfgs(
        padded,
        model_dim,
        inter_dim,
        num_experts,
        topk,
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
        is_ep=is_ep,
        swiglu_limit=0.0,
        opus_weights_shuffled=is_shuffled,
        input_dtype=torch.bfloat16,
    )
    return fm.supports_stage2_overrides(metadata)


def staged_moe_supported(
    num_tokens: int,
    top_k: int,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    is_ep: bool,
) -> bool:
    """Whether :func:`staged_fused_moe` can run a batch of ``num_tokens``. A
    config lookup only; nothing is compiled or launched."""
    return aiter_has_staged_moe() and _stage2_overridable(
        _config_shape(num_tokens, top_k, is_ep),
        tuple(w1.shape),
        tuple(w2.shape),
        w1.dtype,
        is_ep,
        getattr(w1, "is_shuffled", False) and getattr(w2, "is_shuffled", False),
    )


def column_sharded_down_supported(moe_config: FusedMoEConfig) -> bool:
    """Whether this MXFP4 layer builds the column-sharded down projection."""
    parallel_config = moe_config.moe_parallel_config
    tp_size = parallel_config.tp_size
    return (
        envs.VLLM_ROCM_MUSE_MOE_COLUMN_SHARDED_DOWN
        and tp_size > 1
        and not parallel_config.use_ep
        and moe_config.activation == MoEActivation.SILU
        and not moe_config.has_bias
        and moe_config.swiglu_limit is None
        and moe_config.hidden_dim_unpadded in (None, moe_config.hidden_dim)
        and moe_config.intermediate_size_per_partition_unpadded
        in (None, moe_config.intermediate_size_per_partition)
        and not moe_config.intermediate_pad
        and moe_config.hidden_dim % (tp_size * _SHARD_ALIGN) == 0
        and moe_config.intermediate_size_per_partition % _SHARD_ALIGN == 0
        and aiter_has_staged_moe()
    )


def _all_to_all_rows(x: torch.Tensor) -> torch.Tensor:
    """``[E, H, c]`` intermediate slice of every row -> ``[E, H / tp, tp * c]``
    full intermediate dim of this rank's rows (bytes)."""
    x = x.view(torch.uint8)
    tp_size = get_tensor_model_parallel_world_size()
    num_experts, rows, cols = x.shape
    rows_per_rank = rows // tp_size
    send = (
        x.view(num_experts, tp_size, rows_per_rank, cols).transpose(0, 1).contiguous()
    )
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(recv, send, group=get_tp_group().device_group)
    # recv[r] is rank r's intermediate slice of this rank's rows.
    return recv.permute(1, 2, 0, 3).reshape(num_experts, rows_per_rank, -1)


@dataclass
class ColumnShardedDownWeights:
    """This rank's output rows of ``w2`` in AITER's stage2 layout."""

    w2: torch.Tensor
    """``[E, H / tp, I / 2]`` FP4, shuffled."""
    w2_scale: torch.Tensor
    """``[E, H / tp, I / 32]`` E8M0, shuffled."""

    @classmethod
    def from_checkpoint(
        cls, w2: torch.Tensor, w2_scale: torch.Tensor
    ) -> "ColumnShardedDownWeights":
        """Re-shard this rank's raw (unshuffled) ``w2[E, H, I / tp]`` and its
        scales to output rows with one all-to-all each, then shuffle them the
        way the AITER MXFP4 backend shuffles the regular ``w2``."""
        from aiter import dtypes
        from aiter.utility.fp4_utils import e8m0_shuffle

        w2, w2_scale = _all_to_all_rows(w2), _all_to_all_rows(w2_scale)
        num_experts, rows, _ = w2_scale.shape
        w2_scale = e8m0_shuffle(w2_scale.reshape(num_experts * rows, -1)).view(
            num_experts, rows, -1
        )
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4_dtype is not None:
            w2 = w2.view(fp4_dtype)
        (w2,) = rocm_aiter_ops.shuffle_weights(w2)
        w2.is_shuffled = True
        return cls(w2=w2, w2_scale=w2_scale.view(dtypes.fp8_e8m0))


@dataclass
class ColumnShardedRoutes:
    """Per-route expert outputs on this rank's ``H / tp`` output columns,
    handed from the experts to the finalize."""

    routes: torch.Tensor
    """``[T, top_k, H / tp]``, unweighted."""


def all_gather_columns(
    x: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """All-gather TP shards ``[n, c]`` along columns into ``[n, tp * c]``
    (written into ``out`` when given): AITER's all-gather in row chunks that
    fit its IPC buffer, else RCCL. Eager only."""
    comm = getattr(get_tp_group().device_communicator, "aiter_ar_comm", None)
    if comm is not None and not comm.disabled and x.is_contiguous():
        ca = comm.aiter_ca
        row_bytes = x.shape[1] * x.element_size()
        rows = min(x.shape[0], ca.max_size // (ca.world_size * 2) // row_bytes)
        # AITER needs every chunk, including the last partial one, 16B-aligned.
        if row_bytes % 16 == 0 and rows > 0 and ca.should_custom_ag(x[:rows]):
            if out is None:
                out = x.new_empty(x.shape[0], x.shape[1] * ca.world_size)
            for start in range(0, x.shape[0], rows):
                end = min(start + rows, x.shape[0])
                ca.all_gather_unreg(x[start:end], out=out[start:end], dim=-1)
            return out
    gathered = tensor_model_parallel_all_gather(x, dim=-1)
    return gathered if out is None else out.copy_(gathered)


def _all_gather_byte_columns(x: torch.Tensor) -> torch.Tensor:
    """:func:`all_gather_columns` of a 1-byte ``[n, c]`` tensor, moved as bf16
    pairs (``c`` must be even)."""
    gathered = all_gather_columns(x.view(torch.uint8).view(torch.bfloat16))
    return gathered.view(torch.uint8).view(x.dtype)


def _gather_scales(a2_scale: torch.Tensor) -> torch.Tensor:
    """The full-width E8M0 stage2 input scales from each rank's slice: a
    32-row band's tiles are one contiguous run per rank, and the ranks' runs
    side by side are the full-width band."""
    rows, groups = a2_scale.shape
    gathered = _all_gather_byte_columns(
        a2_scale.reshape(rows // _SCALE_TILE_ROWS, groups * _SCALE_TILE_ROWS)
    )
    return gathered.view(rows, -1)


def _gatherable(a2: torch.Tensor, a2_scale: torch.Tensor | None) -> bool:
    """Whether the stage2 input is MXFP4 in the layout the gathers assume."""
    if a2_scale is None or a2.element_size() != 1 or a2_scale.element_size() != 1:
        return False
    if not (a2.is_contiguous() and a2_scale.is_contiguous()):
        return False
    groups = a2.shape[-1] * 2 // _MX_GROUP
    return (
        a2.dim() >= 2
        and a2_scale.dim() == 2
        and a2_scale.shape[1] == groups
        and groups % _SCALE_TILE_GROUPS == 0
        and a2_scale.shape[0] % _SCALE_TILE_ROWS == 0
    )


def staged_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    route_out: torch.Tensor,
    *,
    expert_mask: torch.Tensor | None = None,
    moe_sorting_dispatch_policy: int = 0,
    column_weights: ColumnShardedDownWeights | None = None,
) -> torch.Tensor | ColumnShardedRoutes:
    """The MXFP4 SiLU expert output of every route, weighted by
    ``topk_weights``, in ``route_out[T, top_k, H]``; or, with
    ``column_weights``, on this rank's ``H / tp`` columns in the front of
    ``route_out``'s storage.

    Args:
        hidden_states: ``[T, H]`` bf16 expert input, one row per token.
        w1: AITER-shuffled MXFP4 gate/up expert weights.
        w2: AITER-shuffled MXFP4 down expert weights.
        topk_ids: ``[T, top_k]`` int32 global expert ids.
        topk_weights: ``[T, top_k]`` fp32 route weights.
        w1_scale: ``w1``'s E8M0 block scales.
        w2_scale: ``w2``'s E8M0 block scales.
        route_out: ``[T, top_k, H]`` bf16 destination.
        expert_mask: EP only, AITER's ``[global_E + 1]`` ownership mask.
        moe_sorting_dispatch_policy: AITER sorting dispatch policy.
        column_weights: TP only, this rank's output rows of ``w2``. Every TP
            rank must pass them for the same batch: the gathers inside are
            collectives. The decision to use them depends only on shapes and
            AITER's tuned configs, so it is the same on every rank.

    """
    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType

    num_tokens, top_k = topk_ids.shape
    inputs = fm.fused_moe_stage1(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        expert_mask=expert_mask,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        moe_sorting_dispatch_policy=moe_sorting_dispatch_policy,
        dtype=torch.bfloat16,
        swiglu_limit=0.0,
        config_shape=_config_shape(num_tokens, top_k, expert_mask is not None),
    )
    if column_weights is not None and _gatherable(inputs.a2, inputs.a2_scale):
        a2 = inputs.a2
        gathered = _all_gather_byte_columns(a2.reshape(-1, a2.shape[-1]))
        cols = column_weights.w2.shape[1]
        routes = route_out.view(-1)[: num_tokens * top_k * cols]
        routes = routes.view(num_tokens, top_k, cols)
        fm.fused_moe_stage2(
            inputs,
            out=routes,
            per_slot=True,
            w2=column_weights.w2,
            w2_scale=column_weights.w2_scale,
            a2=gathered.view(*a2.shape[:-1], -1),
            a2_scale=_gather_scales(inputs.a2_scale),
        )
        return ColumnShardedRoutes(routes)
    return fm.fused_moe_stage2(inputs, out=route_out, per_slot=True)
