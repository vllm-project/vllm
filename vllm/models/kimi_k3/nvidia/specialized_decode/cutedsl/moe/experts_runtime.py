# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent routed experts for B<=64 (B>8: grouped units of <=8 tokens per expert)
over one shared core and model-owned workspace. The entry wait completes the
predecessor (incl. workspace rearm/counters) before shared state is read; output rows
keep the caller's top16 order."""

import functools

import cuda.bindings.driver as cuda
import cutlass as c
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ..common.host import check_residency, fake, stream, tensor_maps, to_cute
from ..common.policy import batch_table
from .experts import (
    PLAN_BYTES,
    ExpertsGeometry,
    ExpertsKernel,
    ExpertsPlanKernel,
    fc2_pairs,
)


def tensor(value, align=16):
    return to_cute(value.reshape(-1), align)


def streamed(batch):
    """B<=8 on the incremental producer: ~1-1.5% at B4-8, neutral at B1-2, which keep
    the previous producer."""
    return bool(batch_table("4-8:1").get(batch, 0))


def stealing(batch):
    """B>8 experts balance the FC2(B) tail by claims."""
    return batch > 8 and bool(batch_table("9-64:1").get(batch, 0))


@functools.cache
def compile_experts(tp, experts, batch, grid, device, publish=False, *knobs):
    """Only code and geometry are cached; no owner tensors or descriptors. ``knobs``:
    (B<=8 stream, steal)."""
    defaults = (streamed(batch), False)
    stream, steal = knobs + defaults[len(knobs) :]
    capacity = 1 << (batch - 1).bit_length()
    moe = ExpertsKernel(tp, capacity, experts, grid, publish, stream, steal)
    flat = lambda n, dtype=c.Uint8, align=16: fake((n,), (1,), dtype, align)
    box = (c.Int64(0), c.Int32(0), c.Int32(0)) if publish else ()
    if publish == "twoshot":  # receive slots, owners' bases, parity x TP + rank
        box = (fake((3 * tp, 64, 3584 // tp)), fake((tp,), None, c.Int64), c.Int32(0))
    x, ids = fake((batch, 3584), (3584, 1)), flat(batch * 16, c.Int32)
    x = flat(PLAN_BYTES) if moe.grouped else x  # grouped: the MXFP8 plan
    scales = (flat(experts * n) for n in (moe.w13_scale_bytes, moe.w2_scale_bytes))
    out = flat(batch * 16 * (1 if publish else 3584), c.BFloat16)
    maps = flat(768 if moe.paired else 256, align=64)
    tensors = (x, ids, maps, *scales, flat(moe.workspace_bytes), out)
    scalars = (c.Int32(batch), *map(c.Float32, (100, 0.25, 0.04)), *box)
    with torch.accelerator.device_index(device):
        return cute.compile(moe, *tensors, *scalars, make_fake_stream())


@functools.cache
def compile_plan(batch, device):
    """Plan writer for callers without the routing kernel's copy role in front."""
    args = (fake((batch, 3584)), fake((PLAN_BYTES,), (1,), c.Uint8), make_fake_stream())
    with torch.accelerator.device_index(device):
        return cute.compile(ExpertsPlanKernel(batch), *args)


def scratch(tp, device):
    """Model-owned expert buffers: the workspace and the grouped plan (MXFP8 x, written
    by the routing kernel or ``compile_plan``)."""
    size, u8 = (ExpertsGeometry(tp).workspace_bytes, torch.uint8)
    workspace = torch.full((size,), 255, device=device, dtype=u8)
    plan = torch.empty(PLAN_BYTES, device=device, dtype=u8)
    return {"workspace": workspace, "plan": plan}


class PersistentExperts:
    def __init__(self, w13, s13, w2, s2, *, beta=4.0, linear_beta=25.0):
        experts, rows, _ = w13.shape
        self.tp, self.grid = (8, 136) if rows == 768 else (16, 152)
        self.device, self.experts = w13.device, experts
        self.weights = (w13, s13, w2, s2)
        # TP8: B3-8 publish on 128-136 CTAs, leaving SMs for the shared down, its push
        # and the one-wave tail; grouped B9-64 on 152. TP16 B3-8 on 136 CTAs so the
        # shared expert runs beside the experts (on every SM it ran before them).
        grids = "5-6:128,7:132,9-64:152" if self.tp == 8 else "3-8:136"
        self.grids = batch_table(grids)
        scalars = (beta * linear_beta, 1 / beta, 1 / linear_beta)
        self.scalars = tuple(map(c.Float32, scalars))
        # The experts kernel uses maps 0-1 (L2 promotion on w13); the TP8 B1 cluster
        # experts use maps 2-4 (w13, w2 two-chunk and one-chunk boxes, promoted); TP8
        # FC2 pairs (B4-64) use map 5 (w2 chunk 2 of 256 rows).
        w2_tile = (192, 3, 64) if self.tp == 8 else (128, 2, 32)
        specs = [(w13, rows, 1792, 28, 64, 2, True), (w2, 3584, *w2_tile, 2, False)]
        if self.tp == 8:
            specs += [specs[0]] + [(w2, 3584, 192, 3, 64, box, True) for box in (2, 1)]
        dtype = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
        swizzle = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
        l2 = cuda.CUtensorMapL2promotion
        l2 = (l2.CU_TENSOR_MAP_L2_PROMOTION_NONE, l2.CU_TENSOR_MAP_L2_PROMOTION_L2_256B)
        maps = [
            (weight.data_ptr(), dtype, (128, experts * n, chunks), (row_bytes, stride))
            + ((128, 128, box), swizzle, l2[promote])
            for weight, n, row_bytes, chunks, stride, box, promote in specs
        ]
        if self.tp == 8:
            shape = (128, experts * 3584, 3), (192, 64), (128, 256, 1)
            maps.append((w2.data_ptr(), dtype, *shape, swizzle, l2[False]))
        self.maps = tensor_maps(maps, self.device)
        scales = (tensor(s.view(torch.uint8)) for s in (s13, s2))
        self.views = (to_cute(self.maps[:256], 64), *scales)
        self.all_maps = to_cute(self.maps, 64)  # TP8 FC2 pairs: maps 0-5
        self.prepared = {}

    def prepare(self, batch, buffers, out, publish=(), planned=False, twoshot=False):
        """``publish``: (multicast, parity, rank) of the WaveMoeTail latent slot to
        publish (``twoshot``: the ``common.twoshot`` push box); ``planned``: B>8 calls
        find the plan written (by the routing kernel), else run writes it from x first."""
        mode = "twoshot" if twoshot else bool(publish)
        args: tuple = (batch, self.grid_for(batch), self.device.index, mode)
        args += (streamed(batch), stealing(batch))
        code = compile_experts(self.tp, self.experts, *args)
        # CTAs wait on other clusters' partials: the whole grid must be resident.
        check_residency(code, self.device.index, args[1], 384, "Kimi K3 MoE experts")
        ws, maps = tensor(buffers["workspace"]), self.views[0]
        plan: tuple | None = None
        capacity = 1 << (batch - 1).bit_length()
        if fc2_pairs(self.tp, capacity, batch > 8 or streamed(batch)):
            maps = self.all_maps
        if batch > 8:
            plan = (tensor(buffers["plan"]), compile_plan(batch, self.device.index))
            plan = plan[:1] if planned else plan
        self.prepared[batch] = (out, tensor(out), ws, code, publish, plan, maps)

    def grid_for(self, batch):
        return self.grids.get(batch, self.grid)

    def run(self, x, ids, weights=None):
        batch = x.shape[0]
        out, output, workspace, code, box, plan, maps = self.prepared[batch]
        output = tensor(weights) if box else output
        source = to_cute(x, 16)
        if plan is not None:
            if len(plan) > 1:
                plan[1](source, plan[0], stream(self.device))
            source = plan[0]
        args = (source, tensor(ids), maps, *self.views[1:], workspace, output)
        code(*args, c.Int32(batch), *self.scalars, *box, stream(self.device))
        return out
