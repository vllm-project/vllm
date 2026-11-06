# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The o_proj GEMM and the attention boundary (o_proj all-reduce,
post_attn_norm, gated FP32 residual, pre_ffn_norm) of the Muse Spark
specialized decode.

- ``OProjection``: this rank's row-parallel o_proj partial,
  [b, 128 * 64 / tp] x [8192, 128 * 64 / tp]^T -> BF16 [b, 8192]: on cuBLAS at
  TP4 and TP16; at TP8 (``PACKAGE_GEMM_TPS``) on the package's small-M GEMM
  (the unpooler's kernel, cutedsl/gemm/unpooler.py), whose epilogue also
  publishes the partial into the attention boundary's mailbox (zero-copy).
- ``AttentionBoundary`` (an ``AttentionBoundaryProvider``):
  - TP4 and TP8: the all-reduce fused with the dual norm
    (all_reduce_dual_norm.py), one mailbox shared by every layer, summing in
    the order of the all-reduce the stock o_proj reaches, which the executor
    reads once at build from the backend that all-reduce resolved to
    (``BuildContext.all_reduce_order``; TP4, one node: the MNNVL one-shot by
    default, the node-local one-shot when set; TP8, across trays: the MNNVL
    one-shot and two-shot);
  - TP16: the all-reduce op the executor is given
    (``RoutedExpertsOps.all_reduce``: the TP communicator's MNNVL all-reduce,
    the one the stock o_proj reaches), then the standalone dual norm.
  Both are bit for bit the stock all-reduce followed by the stock dual norm.
"""

import functools
import weakref
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

from vllm.distributed import get_tp_group
from vllm.logger import init_logger

from ..common.dual_norm import DualNorm, DualNormConstants, boundary_constants
from ..common.host import assert_not_capturing
from ..common.mailbox import LamportMailbox
from ..common.policy import ALL_REDUCE_TPS, FUSED_TPS, OPROJ, PACKAGE_GEMM_TPS
from ..gemm.unpooler import Unpooler
from .all_reduce_dual_norm import AllReduceDualNorm, H

# Rows the o_proj GEMM warms up for at build: the executor's BATCHES.
ROWS = (8,)

AllReduce = Callable[[torch.Tensor], torch.Tensor]

logger = init_logger(__name__)


class OProjection:
    """x BF16 [b, K] (this rank's attention output, K = 128 * 64 / tp) ->
    this rank's o_proj partial BF16 [b, 8192], before the all-reduce.

    Without a mailbox: torch.mm into ``out`` (cuBLAS, as the stock o_proj),
    bit for bit.

    With ``mailbox`` (the fused attention boundary's, width 8192): the package
    GEMM (``Unpooler`` with the ``OPROJ`` launch shape), which also multicasts
    the partial into this rank's slot of the mailbox. Not bitwise to cuBLAS
    (summation order); the boundary bound with ``published`` then runs its
    consumer alone.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        rows: Sequence[int] = ROWS,
        mailbox: LamportMailbox | None = None,
    ) -> None:
        assert_not_capturing("OProjection")
        if weight.dtype != torch.bfloat16 or weight.dim() != 2:
            raise ValueError("the o_proj weight must be a BF16 [N, K] matrix")
        if weight.shape[0] != H or not weight.is_contiguous():
            raise ValueError(f"the o_proj weight must be a contiguous [{H}, K]")
        n, k = weight.shape
        self.weight = weight
        self.gemm: Unpooler | None = None
        if mailbox is not None:
            self.gemm = Unpooler(weight, mailbox, shape=OPROJ, rows=rows)
            return
        # cuBLAS's handle and workspace for this stream, before capture (the
        # stock o_proj may run another backend eagerly).
        for m in rows:
            torch.mm(weight.new_zeros(m, k), weight.t(), out=weight.new_empty(m, n))

    @classmethod
    def build(
        cls, ctx: Any, layer: nn.Module, *, rows: Sequence[int] = ROWS
    ) -> "OProjection":
        """From a stock decoder layer's o_proj; with the model's fused boundary
        mailbox where it publishes into it (``producer_mailbox``)."""
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        o_proj = layer.self_attn.o_proj
        if not isinstance(o_proj.quant_method, UnquantizedLinearMethod):
            raise RuntimeError("the specialized decode needs an unquantized o_proj")
        if o_proj.bias is not None:
            raise RuntimeError("the specialized decode needs a bias-free o_proj")
        mailbox = producer_mailbox(ctx)
        if mailbox is not None:
            logger.info_once(
                "Muse Spark o_proj at TP%d: package GEMM %s, partial "
                "published zero-copy into the attention boundary mailbox",
                ctx.tp,
                tuple(OPROJ[:3]),
            )
        return cls(o_proj.weight, rows, mailbox)

    def run(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The partial, in ``out`` (and in the mailbox with the package GEMM:
        the attention boundary bound with ``published`` must run next)."""
        if self.gemm is not None:
            return self.gemm.run(x, out)
        return torch.mm(x, self.weight.t(), out=out)


# The fused boundaries' mailbox of each model, shared by its boundaries.
_FUSED: "weakref.WeakKeyDictionary[nn.Module, AllReduceDualNorm]" = (
    weakref.WeakKeyDictionary()
)


def fused_all_reduce(ctx: Any) -> AllReduceDualNorm:
    """The model's mailbox, created on first use (collective over the TP group:
    every rank builds the same layers in the same order), summing in the order
    of the stock TP all-reduce (``ctx.all_reduce_order``)."""
    shared = _FUSED.get(ctx.backbone)
    if shared is None:
        tp = get_tp_group()
        if ctx.group is not tp and ctx.group is not tp.device_group:
            raise ValueError("the attention boundary takes the TP group")
        sum_order = ctx.all_reduce_order
        if ctx.tp not in FUSED_TPS or sum_order is None:
            raise ValueError(f"TP{ctx.tp} has no fused all-reduce order")
        shared = AllReduceDualNorm(
            tp.device_group, tp.cpu_group, device=ctx.device, sum_order=sum_order
        )
        _FUSED[ctx.backbone] = shared
        logger.info_once(
            "Muse Spark attention boundary at TP%d: all-reduce fused, %s sum "
            "(the stock all-reduce's order)",
            ctx.tp,
            sum_order,
        )
    return shared


def producer_mailbox(ctx: Any) -> LamportMailbox | None:
    """The fused boundary's mailbox where the o_proj GEMM publishes its
    partial into it (``PACKAGE_GEMM_TPS``), else None: the one place that
    decides, for the o_proj (which GEMM) and the attention boundary (whether
    it publishes)."""
    return fused_all_reduce(ctx).mailbox if ctx.tp in PACKAGE_GEMM_TPS else None


@functools.cache
def standalone_dual_norm(constants: DualNormConstants) -> DualNorm:
    """The dual norm's standalone kernel, compiled once per constant set."""
    return DualNorm(constants)


class AttentionBoundary:
    """The attention boundary (an ``AttentionBoundaryProvider``): ``partial``
    BF16 [b, H] -> all-reduce, post_attn_norm, gated add into ``stream`` (FP32
    [b, H], in place), pre_ffn_norm into ``out`` (BF16 [b, H])."""

    def __init__(
        self,
        op: Any,
        all_reduce: AllReduce | None,
        weights: tuple[
            torch.Tensor | None, torch.Tensor | float, torch.Tensor, torch.Tensor
        ],
    ) -> None:
        self.op, self.all_reduce = op, all_reduce
        self.post_weight, self.alpha, self.beta, self.next_weight = weights

    @classmethod
    def unsupported(cls, vllm_config) -> str | None:
        tp = vllm_config.parallel_config.tensor_parallel_size
        if tp not in FUSED_TPS and tp not in ALL_REDUCE_TPS:
            return f"an attention boundary at TP{tp}"
        return None

    @classmethod
    def build(
        cls,
        ctx: Any,
        layer: nn.Module | None,
        *,
        all_reduce: AllReduce | None = None,
    ) -> "AttentionBoundary":
        """``layer`` is a stock decoder layer; ``all_reduce`` the all-reduce op
        (``RoutedExpertsOps.all_reduce``), given at TP16 only."""
        assert layer is not None
        post, connection, pre = layer.post_attn_norm, layer.add_attn, layer.pre_ffn_norm
        if any(n.variance_size_override is not None for n in (post, pre)):
            raise ValueError("the dual norm reduces over the full hidden size")
        if post.hidden_size != H or pre.hidden_size != H:
            raise ValueError(f"the attention boundary needs hidden size {H}")
        constants = boundary_constants(post, connection, pre)
        weights = (
            post.weight if post.has_weight else None,
            connection.alpha,
            connection.beta,
            pre.weight,
        )
        if ctx.tp in FUSED_TPS:
            if all_reduce is not None:
                raise ValueError(f"TP{ctx.tp} fuses its all-reduce")
            op: Any = fused_all_reduce(ctx).bind(
                constants, published=producer_mailbox(ctx) is not None
            )
        elif ctx.tp in ALL_REDUCE_TPS:
            if all_reduce is None:
                raise ValueError(f"TP{ctx.tp} needs the all-reduce op")
            op = standalone_dual_norm(constants)
        else:
            raise ValueError(f"no attention boundary at TP{ctx.tp}")
        return cls(op, all_reduce, weights)

    def prepare_kv(self) -> None:
        pass

    def run(
        self, partial: torch.Tensor, stream: torch.Tensor, out: torch.Tensor
    ) -> None:
        if self.all_reduce is not None:
            partial = self.all_reduce(partial)
        self.op.run(
            partial,
            stream,
            self.post_weight,
            self.alpha,
            self.beta,
            self.next_weight,
            out,
        )
