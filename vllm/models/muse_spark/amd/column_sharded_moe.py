# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Column-sharded MXFP4 down projection for MuseSpark prefill at TP > 1.

Under TP the experts' intermediate dim is sharded, so each rank's down
projection (AITER stage2) yields a partial sum of every route over all ``H``
output columns, and the per-route post-expert norm needs those summed first:
a reduce-scatter of ``[T, top_k, H]`` per layer. Instead, this path

1. all-gathers stage2's MXFP4 input and its E8M0 scales along the
   intermediate dim. Each rank's slice is whole 32-element quant groups, so
   the gathered tensors are exactly the full-width quantization;
2. runs the same stage2 kernel on a second copy of ``w2`` that holds this
   rank's ``H / tp`` output rows over the full intermediate dim, which gives
   the exact per-route outputs on those columns.

The post-expert norm then needs only a ``[T, top_k]`` sum-of-squares
all-reduce (see ``ops/column_sharded_finalize.py``) and an all-gather of the
finalized columns. The second ``w2`` costs ``1 / tp`` of the routed expert
weights per rank in the MXFP4 layers.
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
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

logger = init_logger(__name__)

_MX_GROUP = 32
# AITER's E8M0 activation-scale layout stores each 32-row x 8-group tile as
# one contiguous 256-byte block, tiles of a 32-row band in group order.
_SCALE_TILE_ROWS = 32
_SCALE_TILE_GROUPS = 8
# Per-rank slices of H and of the intermediate dim must be multiples of this:
# whole scale tiles, and whole tiles of AITER's weight and scale shuffles.
_SHARD_ALIGN = 256


@cache
def _aiter_supports_stage2_override() -> bool:
    try:
        from aiter.fused_moe import _fused_moe_impl
    except ImportError:
        return False
    return "_stage2_override" in inspect.signature(_fused_moe_impl).parameters


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
        and _aiter_supports_stage2_override()
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


class _ColumnShardedStage2:
    """AITER ``_stage2_override``: runs the selected stage2 on the gathered
    input and this rank's ``w2`` rows, writing ``out``. A stage2 kernel or
    input layout it does not cover runs unchanged into AITER's own output;
    ``ran`` tells the two apart. The decision depends only on shapes and
    AITER's tuned configs, so it is the same on every TP rank."""

    def __init__(self, weights: ColumnShardedDownWeights, out: torch.Tensor) -> None:
        self.weights = weights
        self.out = out
        self.ran = False

    def __call__(self, *, ordinary_stage2, stage2_args, stage2_kwargs):
        import aiter.fused_moe as fm

        a2, w1, _, sorted_ids, sorted_expert_ids, num_valid_ids, moe_out, topk = (
            stage2_args
        )
        a2_scale = stage2_kwargs.get("a2_scale")
        func = getattr(ordinary_stage2, "func", ordinary_stage2)
        if func not in (
            fm._flydsl_stage2_wrapper,
            fm._flydsl_v2_stage2_wrapper,
        ) or not _gatherable(a2, a2_scale):
            ordinary_stage2(*stage2_args, **stage2_kwargs)
            return moe_out
        if not fm.stage2_uses_route_reduce(ordinary_stage2):
            # Atomic stage2 accumulates into its output.
            self.out.zero_()
        weights = self.weights
        a2 = _all_gather_byte_columns(a2.reshape(-1, a2.shape[-1])).view(
            *a2.shape[:-1], -1
        )
        # The v2 kernel sizes the GEMM from model_dim / inter_dim, v1 from the
        # shapes of w2 and out.
        kwargs = dict(
            stage2_kwargs,
            w2_scale=weights.w2_scale,
            a2_scale=_gather_scales(a2_scale),
            model_dim=weights.w2.shape[1],
            inter_dim=a2.shape[-1] * 2,
        )
        ordinary_stage2(
            a2,
            w1,
            weights.w2,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            self.out,
            topk,
            **kwargs,
        )
        self.ran = True
        return self.out


def column_sharded_fused_experts(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    weights: ColumnShardedDownWeights,
    out: torch.Tensor,
    moe_sorting_dispatch_policy: int = 0,
) -> torch.Tensor:
    """AITER MXFP4 SiLU MoE on ``hidden_states[M, H]`` whose stage2 runs on
    ``weights`` (this rank's output rows of ``w2``; the regular intermediate-
    sharded ``w1`` / ``w2`` still size stage1 and select AITER's config).

    Returns ``out[M, H / tp]`` with the exact outputs on this rank's columns,
    or, if AITER's selected kernels are not covered (including one-stage
    configs), AITER's regular ``[M, H]`` partial sums. Every TP rank must call
    this with the same shapes: the gathers inside stage2 are collectives.
    """
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import _fused_moe_impl

    assert out.is_contiguous() and out.shape == (
        hidden_states.shape[0],
        weights.w2.shape[1],
    )
    stage2 = _ColumnShardedStage2(weights, out)
    args = (hidden_states, w1, w2, topk_weights, topk_ids)
    kwargs = dict(
        activation=ActivationType.Silu.value,
        quant_type=QuantType.per_1x32.value,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        moe_sorting_dispatch_policy=moe_sorting_dispatch_policy,
        dtype=out.dtype,
        swiglu_limit=0.0,
    )
    try:
        result = _fused_moe_impl(*args, **kwargs, _stage2_override=stage2)
    except RuntimeError as e:
        # A one-stage config has no stage2 to override. AITER rejects it before
        # running anything, identically on every rank, so no gather is pending.
        if "two-stage" not in str(e):
            raise
        result = _fused_moe_impl(*args, **kwargs)
    if not stage2.ran:
        logger.warning_once(
            "MuseSpark column-sharded MoE down projection: AITER selected a "
            "stage2 it does not cover; such batches keep the reduce-scatter."
        )
    return result
