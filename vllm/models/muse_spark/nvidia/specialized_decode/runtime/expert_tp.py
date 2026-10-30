# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert tensor parallelism for the routed NVFP4 experts, built from the
stock expert-parallel weights (pure torch and torch.distributed).

The stock layer under TP x EP with EP equal to TP holds ``E / tp`` whole experts
per rank, placed linearly (rank ``r`` owns experts ``[r Er, (r + 1) Er)``), in
the trtllm-gen layout the stock conversion leaves. Expert tensor parallelism
gives every rank all ``E`` experts at ``1 / tp`` of the intermediate width
``I``: rank ``r`` holds intermediate slice ``[r I', (r + 1) I')``, ``I' = I /
tp``, as the stock TP loader would cut it before that conversion.

The cut is free. A TP slice of the converted layout is, byte for byte, the
converted layout of the TP slice:

- w13 weight and w13 block scales ([E, 2I, H/2] and [E, 2I, H/16]): the gated
  interleave puts slice ``r``'s ``2 I'`` rows on the contiguous rows
  ``[2 r I', 2 (r + 1) I')`` and the 32-row shuffle stays inside them; the
  scales' 128x4 interleave keeps whole 128-row tiles contiguous.
- w2 weight ([E, H, I/2]): columns ``[r I'/2, (r + 1) I'/2)`` of every row.
- w2 block scales ([E, H, I/16], interleaved as [H/128, I/64, 512] bytes per
  expert): the k-tiles ``[r I'/64, (r + 1) I'/64)`` of every 128-row m-tile,
  which is already the slice's own interleave.

This needs ``I' % 64 == 0`` and ``H % 128 == 0``. The per-expert scalars
(g1_alphas, g1_scale_c, g2_alphas) do not depend on the slice: their global
input scales are one value for all experts, the same on every rank.

``reshard_nvfp4`` builds a rank's TP experts with one NCCL all-to-all per
tensor kind over the TP group (source ``s`` sends slice ``d`` of its experts to
rank ``d``; the receive buffer is the final tensor, experts in global order) and
three all-gathers of the scalars. It runs once per layer at build, before
capture, and keeps the stock tensors: the stock layer still serves every step
the executor does not take. ``probe_rel_l2`` compares one layer's TP route rows,
summed across ranks, with its EP rows.

Every check runs on every rank and is agreed across the group before any rank
raises, so a refusal never leaves the other ranks waiting in a collective.
"""

from typing import NamedTuple

import torch
import torch.distributed as dist

# Rows of one m-tile and bytes of one (m-tile, k-tile) block of the 128x4
# block-scale interleave.
SF_TILE_ROWS = 128
SF_TILE_BYTES = 512
# Block scales cover 16 FP4 values; a k-tile holds 4 of them (64 values).
SF_VEC = 16
SF_TILE_K = 4 * SF_VEC


class StockNvfp4(NamedTuple):
    """One rank's stock NVFP4 experts of one layer in the converted layout,
    as the stock layer holds them (``first_expert`` = its linear placement)."""

    w13: torch.Tensor  # uint8 [Er, 2I, H/2]
    w13_sf: torch.Tensor  # e4m3 [Er, 2I, H/16], interleaved
    w2: torch.Tensor  # uint8 [Er, H, I/2]
    w2_sf: torch.Tensor  # e4m3 [Er, H, I/16], interleaved
    g1_alphas: torch.Tensor  # fp32 [Er]
    g1_scale_c: torch.Tensor  # fp32 [Er]
    g2_alphas: torch.Tensor  # fp32 [Er]
    a1_gscale: torch.Tensor  # fp32 [1], the experts' input global scale
    a2_gscale: torch.Tensor  # fp32 [1], the intermediate's global scale
    first_expert: int


class TpExperts(NamedTuple):
    """One rank's TP experts of one layer: all ``num_experts`` experts at
    intermediate slice ``rank`` of ``tp``, in the converted layout (the block
    scales as bytes; view them as e4m3 for the op)."""

    w13: torch.Tensor  # uint8 [E, 2I', H/2]
    w13_sf: torch.Tensor  # uint8 [E, 2I', H/16]
    w2: torch.Tensor  # uint8 [E, H, I'/2]
    w2_sf: torch.Tensor  # uint8 [E, H, I'/16]
    g1_alphas: torch.Tensor  # fp32 [E]
    g1_scale_c: torch.Tensor  # fp32 [E]
    g2_alphas: torch.Tensor  # fp32 [E]
    intermediate: int  # I'
    num_experts: int


class Shape(NamedTuple):
    experts: int  # Er, experts per rank
    hidden: int  # H
    intermediate: int  # I
    tp: int

    @property
    def slice(self) -> int:
        return self.intermediate // self.tp


def shape_of(stock: StockNvfp4, tp: int) -> Shape:
    """The shape the stock w13 implies (checked by ``unsupported``)."""
    if stock.w13.dim() != 3:
        raise ValueError("the w13 weight must be [experts, rows, columns]")
    er, rows, half_h = stock.w13.shape
    return Shape(er, 2 * half_h, rows // 2, tp)


def unsupported(stock: StockNvfp4, tp: int) -> str | None:
    """Why ``stock`` cannot be cut ``tp`` ways, or None."""
    shape = shape_of(stock, tp)
    er, h, inter = shape.experts, shape.hidden, shape.intermediate
    expected = {
        "w13": (stock.w13, torch.uint8, (er, 2 * inter, h // 2)),
        "w13_sf": (stock.w13_sf, torch.float8_e4m3fn, (er, 2 * inter, h // SF_VEC)),
        "w2": (stock.w2, torch.uint8, (er, h, inter // 2)),
        "w2_sf": (stock.w2_sf, torch.float8_e4m3fn, (er, h, inter // SF_VEC)),
        "g1_alphas": (stock.g1_alphas, torch.float32, (er,)),
        "g1_scale_c": (stock.g1_scale_c, torch.float32, (er,)),
        "g2_alphas": (stock.g2_alphas, torch.float32, (er,)),
        "a1_gscale": (stock.a1_gscale, torch.float32, (1,)),
        "a2_gscale": (stock.a2_gscale, torch.float32, (1,)),
    }
    for name, (tensor, dtype, shape_) in expected.items():
        if tensor.dtype != dtype or tuple(tensor.shape) != shape_:
            return (
                f"{name} is {tensor.dtype} {tuple(tensor.shape)}, expected "
                f"{dtype} {shape_}"
            )
        if not tensor.is_contiguous():
            return f"{name} is not contiguous"
    if tp < 2 or inter % tp:
        return f"intermediate {inter} does not split {tp} ways"
    if shape.slice % SF_TILE_K or h % SF_TILE_ROWS:
        return (
            f"intermediate slice {shape.slice} and hidden {h} must be multiples "
            f"of {SF_TILE_K} and {SF_TILE_ROWS}"
        )
    return None


def slice_views(stock: StockNvfp4, shape: Shape) -> tuple[torch.Tensor, ...]:
    """(w13, w13_sf, w2, w2_sf) as byte views ``[tp, Er, ...]``: entry ``d``
    is intermediate slice ``d`` of this rank's experts, in the slice's own
    converted layout (no copy; not contiguous). ``shape`` is
    ``shape_of(stock, tp)`` for a ``tp`` that ``unsupported`` accepts."""
    er, h, s, tp = shape.experts, shape.hidden, shape.slice, shape.tp
    w13 = stock.w13.view(er, tp, 2 * s, h // 2).transpose(0, 1)
    w13_sf = (
        stock.w13_sf.view(torch.uint8).view(er, tp, 2 * s, h // SF_VEC).transpose(0, 1)
    )
    w2 = stock.w2.view(er, h, tp, s // 2).permute(2, 0, 1, 3)
    w2_sf = (
        stock.w2_sf.view(torch.uint8)
        .view(er, h // SF_TILE_ROWS, tp, s // SF_TILE_K, SF_TILE_BYTES)
        .permute(2, 0, 1, 3, 4)
    )
    return w13, w13_sf, w2, w2_sf


def final_shapes(shape: Shape, num_experts: int) -> tuple[tuple[int, ...], ...]:
    """The TP tensors' shapes: (w13, w13_sf, w2, w2_sf)."""
    h, s = shape.hidden, shape.slice
    return (
        (num_experts, 2 * s, h // 2),
        (num_experts, 2 * s, h // SF_VEC),
        (num_experts, h, s // 2),
        (num_experts, h, s // SF_VEC),
    )


def agree(group: dist.ProcessGroup, reason: str | None, device) -> str | None:
    """None when no rank of ``group`` has a reason; else this rank's reason, or
    that another rank refused. Collective."""
    flag = torch.tensor([0 if reason is None else 1], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    if int(flag.item()) == 0:
        return None
    return reason or "another rank refused"


def check_global_scales(stock: StockNvfp4, group: dist.ProcessGroup) -> str | None:
    """None when the experts' input and intermediate global scales are the same
    value, bit for bit, on every rank (the TP scalars need it); collective."""
    world = dist.get_world_size(group)
    local = torch.cat([stock.a1_gscale, stock.a2_gscale]).view(torch.int32)
    gathered = torch.empty((world, 2), dtype=torch.int32, device=local.device)
    dist.all_gather_into_tensor(gathered, local.contiguous(), group=group)
    if not bool((gathered == gathered[0]).all()):
        return "the experts' global input scales differ across ranks"
    return None


def reshard_nvfp4(
    stock: StockNvfp4, group: dist.ProcessGroup, num_experts: int
) -> TpExperts:
    """This rank's TP experts of one layer, from every rank's ``stock`` (EP over
    ``group`` with linear placement). Collective over ``group`` (NCCL); run it
    on every rank, before capture, outside any stream other than the current."""
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = stock.w13.device
    shape = shape_of(stock, world)
    reason = unsupported(stock, world)
    if reason is None:
        if shape.experts * world != num_experts:
            reason = (
                f"{world} ranks x {shape.experts} experts is not {num_experts} experts"
            )
        elif stock.first_expert != rank * shape.experts:
            reason = (
                f"rank {rank} holds experts from {stock.first_expert}, not "
                f"{rank * shape.experts} (linear placement)"
            )
    reason = agree(group, reason, device)
    if reason is None:
        reason = agree(group, check_global_scales(stock, group), device)
    if reason is not None:
        raise RuntimeError(f"Expert TP cannot reshard the stock experts: {reason}")

    out = []
    for view, final in zip(slice_views(stock, shape), final_shapes(shape, num_experts)):
        # One kind at a time: the pack buffer is the only transient copy.
        packed = view.contiguous()
        received = torch.empty(final, dtype=torch.uint8, device=device)
        dist.all_to_all_single(received, packed, group=group)
        del packed
        out.append(received)
    scalars = []
    for local in (stock.g1_alphas, stock.g1_scale_c, stock.g2_alphas):
        gathered = torch.empty((num_experts,), dtype=torch.float32, device=device)
        dist.all_gather_into_tensor(gathered, local, group=group)
        scalars.append(gathered)
    w13, w13_sf, w2, w2_sf = out
    g1_alphas, g1_scale_c, g2_alphas = scalars
    return TpExperts(
        w13=w13,
        w13_sf=w13_sf,
        w2=w2,
        w2_sf=w2_sf,
        g1_alphas=g1_alphas,
        g1_scale_c=g1_scale_c,
        g2_alphas=g2_alphas,
        intermediate=shape.slice,
        num_experts=num_experts,
    )


def route_rows(
    rows: torch.Tensor, permuted_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The GEMM2 row of every route (t, k), in route order (rows of routes not
    on this rank read as zero), and which routes are on this rank."""
    idx = permuted_idx.reshape(-1)
    local = idx >= 0
    gathered = rows.index_select(0, idx.clamp_min(0).to(torch.long))
    return gathered.masked_fill(~local.unsqueeze(-1), 0), local


def probe_rel_l2(
    ep: tuple[torch.Tensor, torch.Tensor],
    tp: tuple[torch.Tensor, torch.Tensor],
    group: dist.ProcessGroup,
) -> float:
    """Rel L2 over all routes of one layer's EP route rows against its TP route
    rows summed across ranks (FP32 sum, one BF16 rounding, as the TP finalize
    sums them). ``ep`` and ``tp`` are (gemm2 rows, permute map) of the two ops
    on the same input and routing; the TP map must cover every route (else
    every rank raises). Collective; every rank returns the same value."""
    ep_rows, ep_local = route_rows(*ep)
    tp_rows, tp_local = route_rows(*tp)
    reason = None if bool(tp_local.all()) else "the TP experts left a route unserved"
    reason = agree(group, reason, tp_rows.device)
    if reason is not None:
        raise RuntimeError(reason)
    summed = tp_rows.float()
    dist.all_reduce(summed, group=group)
    # The norms in FP64: BF16 rows of large magnitude must not overflow them.
    summed = summed.to(torch.bfloat16).double()[ep_local]
    want = ep_rows.double()[ep_local]
    sums = torch.stack([(summed - want).square().sum(), want.square().sum()])
    dist.all_reduce(sums, group=group)
    return float((sums[0] / sums[1]).sqrt())
