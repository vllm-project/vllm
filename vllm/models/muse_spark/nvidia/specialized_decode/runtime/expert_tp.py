# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert tensor parallelism for the routed experts (NVFP4, and the BF16 edge
layers), built from the stock expert-parallel weights (pure torch and
torch.distributed).

The stock layer under TP x EP with EP equal to TP holds `E / tp` whole experts
per rank, placed linearly (rank `r` owns experts `[r Er, (r + 1) Er)`), in
the trtllm-gen layout the stock conversion leaves. Expert tensor parallelism
gives every rank all `E` experts at `1 / tp` of the intermediate width
`I`: rank `r` holds intermediate slice `[r I', (r + 1) I')`, `I' = I /
tp`, as the stock TP loader would cut it before that conversion.

The cut is free. A TP slice of the converted layout is, byte for byte, the
converted layout of the TP slice:

- w13 weight and w13 block scales ([E, 2I, H/2] and [E, 2I, H/16]): the gated
  interleave puts slice `r`'s `2 I'` rows on the contiguous rows
  `[2 r I', 2 (r + 1) I')` and the 32-row shuffle stays inside them; the
  scales' 128x4 interleave keeps whole 128-row tiles contiguous.
- w2 weight ([E, H, I/2]): columns `[r I'/2, (r + 1) I'/2)` of every row.
- w2 block scales ([E, H, I/16], interleaved as [H/128, I/64, 512] bytes per
  expert): the k-tiles `[r I'/64, (r + 1) I'/64)` of every 128-row m-tile,
  which is already the slice's own interleave.

This needs `I' % 64 == 0` and `H % 128 == 0`. The per-expert scalars
(g1_alphas, g1_scale_c, g2_alphas) do not depend on the slice: their global
input scales are one value for all experts, the same on every rank.

`reshard_nvfp4` builds a rank's TP experts with one NCCL all-to-all per
tensor kind over the TP group (source `s` sends slice `d` of its experts to
rank `d`; the receive buffer is the final tensor, experts in global order) and
three all-gathers of the scalars. It runs once per layer at build, before
capture, and keeps the stock tensors: the stock layer still serves every step
the executor does not take. `probe_rel_l2` compares one layer's TP route rows,
summed across ranks, with its EP rows.

The BF16 edge layers' experts (`reshard_bf16`) are cut the same way from the
stock block-major-K layout ([E, K/64, rows, 64]): w2 slices on whole 64-wide
K blocks, and w13 slices by gathering rows, because the stock conversion's
gated row permutation depends on the intermediate width. `bf16_w13_rows` reads
that permutation from the conversion itself.

Every check runs on every rank and is agreed across the group before any rank
raises, so a refusal never leaves the other ranks waiting in a collective.
"""

import functools
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
    as the stock layer holds them (`first_expert` = its linear placement)."""

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
    """One rank's TP experts of one layer: all `num_experts` experts at
    intermediate slice `rank` of `tp`, in the converted layout (the block
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


class TpBf16Experts(NamedTuple):
    """One rank's BF16 TP experts of one layer in the stock block-major-K
    layout, all `num_experts` experts at intermediate slice `rank` of `tp`."""

    w13: torch.Tensor  # bf16 [E, H/64, 2I', 64]
    w2: torch.Tensor  # bf16 [E, I'/64, H, 64]
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
    """The shape the stock w13 implies (checked by `unsupported`)."""
    if stock.w13.dim() != 3:
        raise ValueError("the w13 weight must be [experts, rows, columns]")
    er, rows, half_h = stock.w13.shape
    return Shape(er, 2 * half_h, rows // 2, tp)


def unsupported(stock: StockNvfp4, tp: int) -> str | None:
    """Why `stock` cannot be cut `tp` ways, or None."""
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
    """(w13, w13_sf, w2, w2_sf) as byte views `[tp, Er, ...]`: entry `d`
    is intermediate slice `d` of this rank's experts, in the slice's own
    converted layout (no copy; not contiguous). `shape` is
    `shape_of(stock, tp)` for a `tp` that `unsupported` accepts."""
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
    """None when no rank of `group` has a reason; else this rank's reason, or
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
    """This rank's TP experts of one layer, from every rank's `stock` (EP over
    `group` with linear placement). Collective over `group` (NCCL); run it
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


# The stock block-major-K layout: bf16 K blocks of 128 bytes.
BLOCK_K = 64


@functools.cache
def bf16_w13_rows(intermediate: int, tp: int, device) -> torch.Tensor:
    """For each rank `r`, the rows `j` in `[0, 2 I')` of the stock-converted
    w13 (`[E, H/64, 2I, 64]`) that make rank `r`'s own converted slice: the
    converted w13 of the slice (gate rows `[r I', (r + 1) I')` then up rows
    `[I + r I', I + (r + 1) I')`) is the full converted w13 indexed by
    `rows[r]` along its rows. int64 `[tp, 2I']`, on the CPU; cached, so not
    to be modified.

    The stock conversion only permutes rows (whole 64-wide K blocks), by a
    permutation that depends on the row count, so each is read by converting
    a tagged tensor: row `i` carries `i` in its first element."""
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        convert_moe_weights_to_flashinfer_trtllm_block_layout as convert,
    )

    def converted_source_rows(rows: int, device) -> torch.Tensor:
        tagged = torch.zeros((1, rows, BLOCK_K), dtype=torch.bfloat16, device=device)
        tagged.view(torch.int16)[0, :, 0] = torch.arange(
            rows, dtype=torch.int16, device=device
        )
        w2 = torch.zeros((1, BLOCK_K, BLOCK_K), dtype=torch.bfloat16, device=device)
        convert({}, tagged, w2, is_gated_act_gemm=True)
        return tagged.view(torch.int16)[0, :, 0].to(torch.int64).cpu()

    s = intermediate // tp
    full_source = converted_source_rows(2 * intermediate, device)
    full_dest = torch.empty_like(full_source)
    full_dest[full_source] = torch.arange(2 * intermediate)
    slice_source = converted_source_rows(2 * s, device)
    r = torch.arange(tp).unsqueeze(1)
    original = (slice_source // s) * intermediate + r * s + slice_source % s
    return full_dest[original]


def reshard_bf16(
    w13: torch.Tensor,
    w2: torch.Tensor,
    group: dist.ProcessGroup,
    num_experts: int,
    intermediate: int,
) -> TpBf16Experts:
    """This rank's BF16 TP experts of one layer, from every rank's stock
    experts (`w13` `[Er, H/64, 2I, 64]`, `w2` `[Er, I/64, H, 64]`, the
    block-major-K views of the converted weights, EP over `group` with linear
    placement). Collective over `group` (NCCL); run it on every rank, before
    capture. The experts are bf16 and the shapes checked on every rank before
    any raises."""
    world = dist.get_world_size(group)
    device = w13.device
    er, k_blocks = w13.shape[:2]
    s = intermediate // world
    reason = None
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        reason = "the experts are not bf16"
    elif intermediate % (world * 2 * BLOCK_K):
        reason = f"intermediate {intermediate} does not split {world} ways in 128s"
    elif er * world != num_experts:
        reason = f"{world} ranks x {er} experts is not {num_experts} experts"
    elif tuple(w13.shape) != (er, k_blocks, 2 * intermediate, BLOCK_K) or tuple(
        w2.shape
    ) != (er, intermediate // BLOCK_K, k_blocks * BLOCK_K, BLOCK_K):
        reason = f"unexpected shapes {tuple(w13.shape)}, {tuple(w2.shape)}"
    elif not (w13.is_contiguous() and w2.is_contiguous()):
        reason = "the experts are not contiguous"
    reason = agree(group, reason, device)
    if reason is not None:
        raise RuntimeError(f"Expert TP cannot reshard the BF16 experts: {reason}")

    rows = bf16_w13_rows(intermediate, world, device).to(device)
    packed = torch.empty(
        (world, er, k_blocks, 2 * s, BLOCK_K), dtype=w13.dtype, device=device
    )
    for d in range(world):
        torch.index_select(w13, 2, rows[d], out=packed[d])
    w13_tp = torch.empty(
        (num_experts, k_blocks, 2 * s, BLOCK_K), dtype=w13.dtype, device=device
    )
    dist.all_to_all_single(w13_tp, packed, group=group)
    del packed
    packed = (
        w2.view(er, world, s // BLOCK_K, w2.shape[2], BLOCK_K)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )
    w2_tp = torch.empty(
        (num_experts, s // BLOCK_K, w2.shape[2], BLOCK_K), dtype=w2.dtype, device=device
    )
    dist.all_to_all_single(w2_tp, packed, group=group)
    return TpBf16Experts(w13_tp, w2_tp, s, num_experts)


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
    sums them). `ep` and `tp` are (gemm2 rows, permute map) of the two ops
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
