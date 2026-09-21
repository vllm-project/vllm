# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Global expert pool tables and the per-layer step program.

One VRAM bank shared by every MoE layer: row ``r`` holds the expert whose
key is ``row_key[r]`` (key = layer * E + expert) or is free. ``hot_phys[key]``
is the row or -1; each layer's slice is the expert map its kernel reads.
``cold_phys[key]`` is the expert's host row (its expert id) while it is not
resident, so evictions never copy out (the pinned host source is the
backing store).

Each layer's decode step is one device program: distinct valid selections
are stamped with the step clock; each miss takes the row of the least
recently used resident expert of any layer (ties by row), copying in from
this layer's host rows; a miss that finds no victim, or any miss while the
gate is closed, is staged into the shared staging rows for this step only.
The step map is this layer's ``hot_phys`` slice with the staged experts
overlaid. Everything runs on the compute stream with fixed shapes and
addresses, so it can be captured in a CUDA graph; the host reads the
tables only at stats reports.

``step_reference`` (torch, synchronizing) defines the semantics; the CPU
device and the tests run it, and the Triton program must match it exactly.
Ported from the lab expert tier (global_pool.py) without the control file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TENSORS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_2",
    "w2_weight_scale_2",
)

PLAN_WIDTH = 16
ROW_USE_NEVER = 0x7FFFFFFFFFFFFFFF  # staging rows: never a victim


@dataclass
class GlobalTables:
    """Pool-wide device state; every tensor has a fixed address."""

    num_layers: int
    num_experts: int
    pool_rows: int
    hot_phys: Any  # [L*E] int32 key -> bank row / -1
    cold_phys: Any  # [L*E] int32 key -> RAM row (expert id) / -1
    row_key: Any  # [pool_rows + staging] int32 row -> key / -1
    row_use: Any  # [pool_rows + staging] int64 step of the row's last selection
    clock: Any  # [1] int64
    gate: Any  # [1] int32 promotions allowed
    error: Any  # [1] int32 sticky device error
    promote_limit: Any  # [1] int32 max promotions per layer call, 0 = unlimited
    promote_interval: Any  # [1] int32 promote only every N forwards
    forwards: Any  # [1] int32 forwards seen with the gate open (layer 0 count)
    promote_min_misses: Any  # [1] int32 misses a key needs before promotion
    protect_recent: Any  # [1] int32 forwards a used row stays unevictable
    miss_count: Any  # [L*E] int32 misses since the key was last resident
    staging_rows: Any  # [S] int32 shared staging rows (constant)

    @property
    def keys(self):
        return self.num_layers * self.num_experts

    def layer_slice(self, table, layer):
        start = layer * self.num_experts
        return table[start : start + self.num_experts]


@dataclass
class StepBuffers:
    """Per-layer fixed-address scratch the step program fills."""

    gather_src: Any  # [W] int32 RAM rows
    gather_dst: Any  # [W] int32 bank rows
    gather_count: Any  # [1] int32
    routes: Any  # [W] int32 physical row per ids lane this step, -1 padding
    staged_expert: Any  # [W] int32 (scratch for the map overlay)
    staged_row: Any  # [W] int32
    staged_count: Any  # [1] int32
    promoted_count: Any  # [1] int32
    step_map: Any  # [E] int32


def allocate_global_tables(device, num_experts, slots_per_layer, staging):
    """Layer l's first `slots_per_layer[l]` experts start resident, packed
    in layer order; the `staging` rows follow the pool and stay free."""
    import torch

    num_layers = len(slots_per_layer)
    if staging < 1 or any(not 0 < s < num_experts for s in slots_per_layer):
        raise ValueError("Global pool needs staging rows and partial layers")
    keys = num_layers * num_experts
    pool_rows = sum(slots_per_layer)
    hot_phys = torch.full((keys,), -1, dtype=torch.int32)
    cold_phys = torch.arange(num_experts, dtype=torch.int32).repeat(num_layers)
    row_key = torch.full((pool_rows + staging,), -1, dtype=torch.int32)
    offset = 0
    for layer, slots in enumerate(slots_per_layer):
        base = layer * num_experts
        hot_phys[base : base + slots] = torch.arange(
            offset, offset + slots, dtype=torch.int32
        )
        cold_phys[base : base + slots] = -1
        row_key[offset : offset + slots] = torch.arange(
            base, base + slots, dtype=torch.int32
        )
        offset += slots
    return GlobalTables(
        num_layers=num_layers,
        num_experts=num_experts,
        pool_rows=pool_rows,
        hot_phys=hot_phys.to(device),
        cold_phys=cold_phys.to(device),
        row_key=row_key.to(device),
        row_use=torch.cat(
            (
                torch.zeros(pool_rows, dtype=torch.int64),
                torch.full((staging,), ROW_USE_NEVER, dtype=torch.int64),
            )
        ).to(device),
        clock=torch.zeros(1, dtype=torch.int64, device=device),
        gate=torch.zeros(1, dtype=torch.int32, device=device),
        error=torch.zeros(1, dtype=torch.int32, device=device),
        promote_limit=torch.zeros(1, dtype=torch.int32, device=device),
        promote_interval=torch.ones(1, dtype=torch.int32, device=device),
        forwards=torch.zeros(1, dtype=torch.int32, device=device),
        promote_min_misses=torch.ones(1, dtype=torch.int32, device=device),
        protect_recent=torch.zeros(1, dtype=torch.int32, device=device),
        miss_count=torch.zeros(keys, dtype=torch.int32, device=device),
        staging_rows=torch.arange(
            pool_rows, pool_rows + staging, dtype=torch.int32, device=device
        ),
    )


def allocate_step_buffers(device, num_experts, width=PLAN_WIDTH):
    import torch

    def ints(n):
        return torch.zeros(n, dtype=torch.int32, device=device)

    return StepBuffers(
        gather_src=ints(width),
        gather_dst=ints(width),
        gather_count=ints(1),
        routes=torch.full((width,), -1, dtype=torch.int32, device=device),
        staged_expert=ints(width),
        staged_row=ints(width),
        staged_count=ints(1),
        promoted_count=ints(1),
        step_map=torch.full((num_experts,), -1, dtype=torch.int32, device=device),
    )


def set_gate(tables, enabled):
    tables.gate.fill_(1 if enabled else 0)


CONTROL_MAX = 2**31 - 1  # device scalars are int32
CONTROL_FIELDS = (
    "promote_limit",
    "promote_interval",
    "promote_min_misses",
    "protect_recent",
    "gate",
)


def validate_control(values):
    """Validate a control mapping; returns the normalized dict or raises."""
    out = {}
    for name, value in values.items():
        if name not in CONTROL_FIELDS:
            raise ValueError(f"Unknown pool control {name!r}")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Pool control {name} must be an integer")
        if not 0 <= value <= CONTROL_MAX:
            raise ValueError(f"Pool control {name} outside [0, {CONTROL_MAX}]")
        if name == "promote_limit" and value < 0:
            raise ValueError("promote_limit must be nonnegative")
        if name in ("promote_interval", "promote_min_misses") and value < 1:
            raise ValueError(f"{name} must be positive")
        if name == "protect_recent" and value < 0:
            raise ValueError("protect_recent must be nonnegative")
        if name == "gate" and value not in (0, 1):
            raise ValueError("gate must be 0 or 1")
        out[name] = value
    return out


def set_control(tables, **values):
    """Write placement controls (device scalars at fixed addresses).

    `promote_limit`: promotions per *layer call* (0 = unlimited); every
    layer reads the same scalar, so it is not a model-wide total.
    `promote_interval`: promote only on every N-th forward with the gate
    open, counted once per forward at layer 0 (a prefill or a multi-row
    verify forward counts once). `promote_min_misses`: a key is promoted
    only once it has missed this many times since it was last resident
    (1 = first miss). `protect_recent`: rows used within the last N
    forwards, the previous one included, are never victims (0 = none).
    `gate`: 0 freezes the placement. In every frozen or deferred case all
    misses are still computed from staging rows; only the placement changes.
    All fields are validated before any is written.
    """
    out = validate_control(values)
    for name, value in out.items():
        if name == "gate":
            set_gate(tables, bool(value))
        else:
            getattr(tables, name).fill_(value)
    return out


def read_control(tables):
    values = {name: int(getattr(tables, name)[0]) for name in CONTROL_FIELDS}
    values["forwards"] = int(tables.forwards[0])
    return values


def step_reference(tables, layer, ids, buffers):
    """Plan and flip one layer step on the host (torch, synchronizing).

    Returns (gathers, step_map) with gathers as (RAM row, bank row) pairs in
    copy order: promotions first, then staged-only misses. Semantics:

    - `ids` values outside [0, E) other than -1 set the sticky error and
      are skipped; -1 is padding.
    - Distinct valid selections in first-occurrence order. With the gate
      open the clock advances and every selected resident row is stamped.
    - Recency lives on pool rows (FreeToken's usage-per-slot): each miss,
      in that order, evicts the pool row with the smallest (row_use, row)
      among rows not stamped this step and takes it; without such a row
      it is staged only. Staging rows are never victims. With the gate
      closed every miss is staged only and recency is untouched.
    - `buffers.routes[i]` is the physical row of ids lane i (-1 for
      padding, invalid, or a duplicate of an earlier lane's expert is still
      resolved to that expert's row).
    """
    import torch

    E = tables.num_experts
    if not 0 <= layer < tables.num_layers:
        raise ValueError("Layer index outside the pool")
    hot = tables.hot_phys.tolist()
    cold = tables.cold_phys.tolist()
    row_key = tables.row_key.tolist()
    row_use = tables.row_use.tolist()
    staging_rows = tables.staging_rows.tolist()
    gate = bool(int(tables.gate[0]))
    raw = [int(v) for v in ids.reshape(-1).tolist()]
    if len(raw) > buffers.gather_src.shape[0] or len(raw) > len(staging_rows):
        raise ValueError("Step ids exceed the plan width or the staging rows")
    error = bool(int(tables.error[0]))
    selected: list[int] = []
    for value in raw:
        if value == -1:
            continue
        if not 0 <= value < E:
            error = True
            continue
        if value not in selected:
            selected.append(value)
    clock = int(tables.clock[0])
    forwards = int(tables.forwards[0])
    base = layer * E
    if gate:
        clock += 1
        if layer == 0:
            forwards += 1
        for e in selected:
            if hot[base + e] >= 0:
                row_use[hot[base + e]] = clock
    limit = int(tables.promote_limit[0])
    interval = int(tables.promote_interval[0])
    min_misses = int(tables.promote_min_misses[0])
    protect = int(tables.protect_recent[0]) * tables.num_layers
    miss_count = tables.miss_count.tolist()
    promote_ok = gate and (forwards - 1) % interval == 0
    gathers: list[tuple[int, int]] = []
    staged: list[tuple[int, int]] = []
    for e in selected:
        key = base + e
        if hot[key] >= 0:
            continue
        victim_row = -1
        if gate:
            miss_count[key] += 1
        if (
            promote_ok
            and (limit == 0 or len(gathers) < limit)
            and miss_count[key] >= min_misses
        ):
            best = None
            for r in range(tables.pool_rows):
                if (
                    row_key[r] >= 0
                    and row_use[r] < clock
                    and (protect == 0 or row_use[r] < clock - protect)
                ):
                    candidate = (row_use[r], r)
                    if best is None or candidate < best:
                        best = candidate
            if best is not None:
                victim_row = best[1]
        if victim_row < 0:
            staged.append((e, staging_rows[len(staged)]))
            continue
        victim = row_key[victim_row]
        hot[victim], cold[victim] = -1, victim % E
        hot[key], cold[key] = victim_row, -1
        row_key[victim_row] = key
        row_use[victim_row] = clock
        miss_count[key] = 0
        gathers.append((e, victim_row))
    step_map = hot[base : base + E]
    for e, row in staged:
        step_map[e] = row
    routes = [-1] * buffers.routes.shape[0]
    for i, value in enumerate(raw):
        if 0 <= value < E:
            routes[i] = step_map[value]
    device = tables.hot_phys.device

    def write(target, values, dtype):
        target.copy_(torch.tensor(values, dtype=dtype, device=device))

    write(tables.hot_phys, hot, torch.int32)
    write(tables.cold_phys, cold, torch.int32)
    write(tables.row_key, row_key, torch.int32)
    write(tables.row_use, row_use, torch.int64)
    tables.clock.fill_(clock)
    tables.forwards.fill_(forwards)
    write(tables.miss_count, miss_count, torch.int32)
    tables.error.fill_(1 if error else 0)
    pairs = gathers + [(e, row) for e, row in staged]
    buffers.gather_count.fill_(len(pairs))
    buffers.promoted_count.fill_(len(gathers))
    buffers.staged_count.fill_(len(staged))
    for i, (src, dst) in enumerate(pairs):
        buffers.gather_src[i], buffers.gather_dst[i] = src, dst
    for i, (e, row) in enumerate(staged):
        buffers.staged_expert[i], buffers.staged_row[i] = e, row
    write(buffers.step_map, step_map, torch.int32)
    write(buffers.routes, routes, torch.int32)
    return pairs, buffers.step_map


def step(tables, layer, ids, buffers):
    """Plan and flip one layer step: Triton on CUDA, the reference elsewhere."""
    if tables.hot_phys.device.type != "cuda":
        step_reference(tables, layer, ids, buffers)
        return
    flat = ids.reshape(-1)
    if not flat.is_contiguous():
        raise ValueError("Global step requires contiguous ids")
    width = buffers.gather_src.shape[0]
    if flat.numel() > width or flat.numel() > tables.staging_rows.shape[0]:
        raise ValueError("Step ids exceed the plan width or the staging rows")
    rows = tables.pool_rows
    _step_kernel()[(1,)](
        flat,
        flat.numel(),
        layer,
        tables.hot_phys,
        tables.cold_phys,
        tables.row_key,
        tables.row_use,
        tables.clock,
        tables.gate,
        tables.error,
        tables.promote_limit,
        tables.promote_interval,
        tables.forwards,
        tables.promote_min_misses,
        tables.protect_recent,
        tables.miss_count,
        tables.staging_rows,
        buffers.gather_src,
        buffers.gather_dst,
        buffers.gather_count,
        buffers.routes,
        buffers.staged_expert,
        buffers.staged_row,
        buffers.staged_count,
        buffers.promoted_count,
        buffers.step_map,
        tables.num_experts,
        rows,
        tables.num_layers,
        WIDTH=width,
        BLOCK_R=_next_power_of_two(rows),
        MAP_BLOCK=1024,
        num_warps=8,
    )


def check_global_tables(tables):
    """Consistency of the pool; raises on any violation.

    Every key is resident or has its RAM row, never both; resident keys and
    rows are a bijection; staging rows are never owned; no device error.
    """
    E = tables.num_experts
    hot = tables.hot_phys.tolist()
    cold = tables.cold_phys.tolist()
    row_key = tables.row_key.tolist()
    staging = set(tables.staging_rows.tolist())
    owners = {}
    for key, (h, c) in enumerate(zip(hot, cold)):
        if (h >= 0) == (c >= 0):
            raise AssertionError(f"Key {key} must be resident or backed, not both")
        if c >= 0 and c != key % E:
            raise AssertionError(f"Key {key} must be backed by its own RAM row")
        if h >= 0:
            if h in staging or not 0 <= h < tables.pool_rows:
                raise AssertionError(f"Key {key} owns a row outside the pool")
            if h in owners:
                raise AssertionError(f"Row {h} has two owners")
            owners[h] = key
    for row, key in enumerate(row_key):
        if owners.get(row, -1) != key:
            raise AssertionError(f"Row {row} owner table disagrees")
    if int(tables.error[0]):
        raise RuntimeError("Global pool recorded a device error")


def resident_per_layer(tables):
    hot = tables.hot_phys.view(tables.num_layers, tables.num_experts)
    return (hot >= 0).sum(dim=1).tolist()


_KERNELS: dict[str, Any] = {}


def _next_power_of_two(value):
    return 1 << max(int(value) - 1, 0).bit_length()


def _step_kernel():
    """One program: `step_reference` on the device."""
    if "step" in _KERNELS:
        return _KERNELS["step"]
    from vllm.triton_utils import tl, triton

    @triton.jit
    def global_pool_step(
        ids_ptr,
        n,
        layer,
        hot_phys_ptr,
        cold_phys_ptr,
        row_key_ptr,
        row_use_ptr,
        clock_ptr,
        gate_ptr,
        error_ptr,
        limit_ptr,
        interval_ptr,
        forwards_ptr,
        min_misses_ptr,
        protect_ptr,
        miss_count_ptr,
        staging_ptr,
        gather_src_ptr,
        gather_dst_ptr,
        gather_count_ptr,
        routes_ptr,
        staged_expert_ptr,
        staged_row_ptr,
        staged_count_ptr,
        promoted_count_ptr,
        step_map_ptr,
        num_experts,
        pool_rows,
        num_layers,
        WIDTH: tl.constexpr,
        BLOCK_R: tl.constexpr,
        MAP_BLOCK: tl.constexpr,
    ):
        never = 0x7FFFFFFFFFFFFFFF
        lane = tl.arange(0, WIDTH)
        present = lane < n
        raw = tl.load(ids_ptr + lane, mask=present, other=-1).to(tl.int64)
        valid = present & (raw >= 0) & (raw < num_experts)
        bad = present & (raw != -1) & (~valid)
        if tl.sum(bad.to(tl.int32), 0) > 0:
            tl.store(error_ptr, 1)
        safe = tl.where(valid, raw, 0)
        same = safe[:, None] == safe[None, :]
        earlier = lane[None, :] < lane[:, None]
        duplicate = tl.sum((same & earlier & valid[None, :]).to(tl.int32), 1) > 0
        distinct = valid & (duplicate == 0)
        # `layer` may arrive as a Python int (Triton specializes 0 and 1).
        base = tl.full((), 0, tl.int64) + layer * num_experts
        keys = base + safe
        resident = tl.load(hot_phys_ptr + keys, mask=distinct, other=-1).to(tl.int64)
        hit = distinct & (resident >= 0)
        gate = tl.load(gate_ptr) != 0
        clock = tl.load(clock_ptr)
        forwards = tl.load(forwards_ptr)
        if gate:
            clock = clock + 1
            tl.store(clock_ptr, clock)
            if layer == 0:
                forwards = forwards + 1
                tl.store(forwards_ptr, forwards)
            tl.store(row_use_ptr + tl.where(hit, resident, 0), clock, mask=hit)
        limit = tl.load(limit_ptr)
        interval = tl.load(interval_ptr)
        min_misses = tl.load(min_misses_ptr)
        protect = tl.load(protect_ptr).to(tl.int64) * num_layers
        promote_ok = gate & (((forwards - 1) % interval) == 0)
        tl.debug_barrier()
        # The pool-wide recency vector is only needed when a miss can be
        # promoted (FreeToken scans its cache inside the same condition);
        # on an all-hit layer, the common case, nothing below touches it.
        offs_r = tl.arange(0, BLOCK_R)
        in_pool = offs_r < pool_rows
        misses = tl.sum((distinct & (~hit)).to(tl.int32), 0)
        scan = promote_ok & (misses > 0)
        use = tl.full((BLOCK_R,), never, tl.int64)
        if scan:
            use = tl.load(row_use_ptr + offs_r, mask=in_pool, other=never)
            # Rows used within the last protect_recent forwards (the
            # previous one included) stay.
            if protect > 0:
                use = tl.where(use >= clock - protect, never, use)
            # Rows selected this step (hits) are masked in registers; extract
            # lane i's hit row (or -1): the other lanes contribute 0.
            hit_rows = tl.where(hit, resident, -1)
            for i in range(0, WIDTH):
                hit_row = tl.sum(tl.where(lane == i, hit_rows, 0), 0)
                use = tl.where(offs_r.to(tl.int64) == hit_row, never, use)
        promoted = 0
        staged = 0
        for i in range(0, WIDTH):
            is_miss = tl.sum(
                tl.where(lane == i, (distinct & (~hit)).to(tl.int32), 0), 0
            )
            if is_miss > 0:
                expert = tl.load(ids_ptr + i).to(tl.int64)
                key = base + expert
                victim_row = tl.full((), -1, tl.int64)
                misses_so_far = tl.load(miss_count_ptr + key)
                if gate:
                    misses_so_far = misses_so_far + 1
                    tl.store(miss_count_ptr + key, misses_so_far)
                if (
                    promote_ok
                    & ((limit == 0) | (promoted < limit))
                    & (misses_so_far >= min_misses)
                ):
                    best = tl.min(use, 0)
                    if best != never:
                        victim_row = tl.min(
                            tl.where(use == best, offs_r.to(tl.int64), never), 0
                        )
                if victim_row >= 0:
                    victim = tl.load(row_key_ptr + victim_row).to(tl.int64)
                    tl.store(hot_phys_ptr + victim, -1)
                    tl.store(
                        cold_phys_ptr + victim, (victim % num_experts).to(tl.int32)
                    )
                    tl.store(hot_phys_ptr + key, victim_row.to(tl.int32))
                    tl.store(cold_phys_ptr + key, -1)
                    tl.store(row_key_ptr + victim_row, key.to(tl.int32))
                    tl.store(row_use_ptr + victim_row, clock)
                    tl.store(miss_count_ptr + key, 0)
                    tl.store(gather_src_ptr + promoted, expert.to(tl.int32))
                    tl.store(gather_dst_ptr + promoted, victim_row.to(tl.int32))
                    use = tl.where(offs_r.to(tl.int64) == victim_row, never, use)
                    promoted += 1
                else:
                    tl.store(staged_expert_ptr + staged, expert.to(tl.int32))
                    tl.store(staged_row_ptr + staged, tl.load(staging_ptr + staged))
                    staged += 1
            # The scalar table stores above must be visible to the next
            # miss's loads of hot_phys / row_key.
            tl.debug_barrier()
        tl.store(promoted_count_ptr, promoted)
        tl.store(staged_count_ptr, staged)
        tl.store(gather_count_ptr, promoted + staged)
        tl.debug_barrier()
        for i in range(0, staged):
            tl.store(gather_src_ptr + promoted + i, tl.load(staged_expert_ptr + i))
            tl.store(gather_dst_ptr + promoted + i, tl.load(staged_row_ptr + i))
        for start in range(0, num_experts, MAP_BLOCK):
            offs = start + tl.arange(0, MAP_BLOCK)
            in_range = offs < num_experts
            rows = tl.load(hot_phys_ptr + base + offs, mask=in_range, other=-1)
            tl.store(step_map_ptr + offs, rows, mask=in_range)
        tl.debug_barrier()
        for i in range(0, staged):
            expert = tl.load(staged_expert_ptr + i)
            tl.store(step_map_ptr + expert, tl.load(staged_row_ptr + i))
        tl.debug_barrier()
        # Routes: the physical row of every ids lane through the step map.
        route = tl.load(step_map_ptr + safe, mask=valid, other=-1)
        tl.store(routes_ptr + lane, tl.where(valid, route, -1), mask=lane < WIDTH)

    _KERNELS["step"] = global_pool_step
    return global_pool_step
