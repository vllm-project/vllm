# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.8-27B dense mono grid and stage plans, on the CPU.

``tiling`` picks the grid from the device's compute units (256 on MI355X /
MI350X, 128 on MI350P) and derives each K1 / K2 stage's per-CTA work from it.
These tests need no GPU, FlyDSL or torch: the device query is replaced, and the
kernels' task placement is replayed here in plain integers from the same plan
(the formulas of ``dense_ffn`` / ``dense_gdn``), to check that every task runs
exactly once at every validated width, that the 256-CTA plan is the one the
kernels were written for, and that no GDN pass waits on a later one.
"""

from collections import Counter

import pytest

from vllm.models.qwen3_5.amd.mono import tiling as T

GRIDS = T.VALIDATED_GRIDS


# ---------------------------------------------------------------- the grid
@pytest.mark.parametrize("cus", [256, 128])
def test_grid_is_a_validated_cu_count(cus):
    assert T.resolve_grid(cus) == cus
    assert T.resolve_grid(cus, "") == cus


@pytest.mark.parametrize("cus", [304, 240, 192, 160, 80, 64, 32])
def test_unvalidated_cu_count_is_refused(cus):
    with pytest.raises(ValueError, match=f"no validated grid width for {cus}"):
        T.resolve_grid(cus)


def test_override_runs_a_narrower_validated_grid():
    assert T.resolve_grid(256, "128") == 128
    assert T.resolve_grid(256, 256) == 256
    # a GPU without a validated count of its own, run at a narrower width
    assert T.resolve_grid(304, "256") == 256


@pytest.mark.parametrize(
    "cus, override, match",
    [
        (128, "256", "wider than the GPU's 128"),
        (256, "192", "not a validated grid width"),
        (256, "x", "not an integer"),
    ],
)
def test_bad_override_is_refused(cus, override, match):
    with pytest.raises(ValueError, match=match):
        T.resolve_grid(cus, override)


@pytest.mark.parametrize(
    "cus, env, grid", [(256, None, 256), (128, None, 128), (256, "128", 128)]
)
def test_device_grid(monkeypatch, cus, env, grid):
    monkeypatch.setattr(T, "_compute_units", lambda: cus)
    if env is None:
        monkeypatch.delenv(T.GRID_ENV, raising=False)
    else:
        monkeypatch.setenv(T.GRID_ENV, env)
    assert T.device_grid() == grid


def test_device_grid_refuses_mi300x_count(monkeypatch):
    monkeypatch.setattr(T, "_compute_units", lambda: 304)
    monkeypatch.delenv(T.GRID_ENV, raising=False)
    with pytest.raises(ValueError, match="validated"):
        T.device_grid()


def test_a_grid_wider_than_the_gpu_is_refused_before_it_hangs(monkeypatch):
    monkeypatch.setattr(T, "_resident_cus", lambda: 128)
    monkeypatch.delenv(T.GRID_ENV, raising=False)
    T.check_resident(128)
    with pytest.raises(RuntimeError, match="deadlock.*grid=128"):
        T.check_resident(256)


# ---------------------------------------------------------------- 256: as written
def test_256_plan_is_the_mi355x_layout():
    """The numbers the kernels hard-coded before the grid was a parameter."""
    f1, f2 = T.FGeo(1, 256), T.FGeo(2, 256)
    assert (f1.g_full, f1.sl_cols, f1.bulk_blocks) == (4, 4, 512)
    assert (f2.g_full, f2.sl_cols) == (2, 2)
    # down: D_FULL = 256, D_SL = 64, D_SL_ROWS = 4, one pass of (whole, split)
    assert (f1.d_per, f1.d_full, f1.d_sl, f1.d_sl_rows) == (1, 256, 64, 4)
    assert f1.d_passes == 1 and f1.d_pass_tiles(0) == (0, f1.d_per)
    assert f1.o_passes == 1
    g1, g2 = T.Geo(1, 256), T.Geo(2, 256)
    assert (g1.qg_cta, g1.qg_pass, g1.qkvz_passes, g1.ba_ctas) == (4, 4, 1, 48)
    assert (g2.qg_cta, g2.qg_pass, g2.qkvz_passes, g2.ba_ctas) == (2, 2, 1, 24)
    for s in range(1, 9):
        assert T.n_tasks(g1, s) == 3 * s
        assert T.gdn_passes(g1, s) == ((0, 3 * s),)
    for s in range(1, 17):
        assert T.n_tasks(g2, s) == -(-3 * s // 2)
        assert T.gdn_passes(g2, s) == ((0, T.n_tasks(g2, s)),)


def test_128_plan():
    f1, f2 = T.FGeo(1, 128), T.FGeo(2, 128)
    assert (f1.g_full, f1.sl_cols, f1.bulk_blocks) == (8, 8, 512)
    assert (f2.g_full, f2.sl_cols) == (4, 4)
    assert (f1.d_per, f1.d_full, f1.d_sl, f1.d_sl_rows) == (2, 256, 64, 8)
    assert f1.d_passes == 2
    assert [f1.d_pass_tiles(p) for p in range(2)] == [(0, 1), (2, None)]
    assert f1.o_passes == 2
    g1, g2 = T.Geo(1, 128), T.Geo(2, 128)
    assert (g1.qg_cta, g1.qg_pass, g1.qkvz_passes) == (8, 4, 2)
    assert (g2.qg_cta, g2.qg_pass, g2.qkvz_passes) == (4, 4, 1)
    assert T.gdn_passes(g1, 4) == ((0, 24),)
    assert T.gdn_passes(g1, 5) == ((0, 18), (18, 12))
    assert T.gdn_passes(g1, 8) == ((0, 24), (24, 24))
    assert T.gdn_passes(g2, 8) == ((0, 24),)
    assert T.gdn_passes(g2, 9) == ((0, 15), (15, 12))
    assert T.gdn_passes(g2, 16) == ((0, 24), (24, 24))


def _plan_ok(grid):
    try:
        for tp in T.TP_SIZES:
            T.check_ffn(T.FGeo(tp, grid))
            T.check_gdn(T.Geo(tp, grid))
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("grid", GRIDS)
def test_validated_grids_pass_the_layout_checks(grid):
    assert _plan_ok(grid)


@pytest.mark.parametrize("grid", [64, 160, 192, 320])
def test_other_grids_fail_the_layout_checks(grid):
    """Widths without a stage plan fail the invariants too, so widening
    ``VALIDATED_GRIDS`` alone cannot let one through."""
    assert not _plan_ok(grid)


# ---------------------------------------------------------------- coverage
def _once(counter, universe):
    assert set(counter) == set(universe), "a task is never run"
    assert all(n == 1 for n in counter.values()), "a task runs twice"


@pytest.mark.parametrize("grid", GRIDS)
def test_owners_cover_160_tasks_once(grid):
    """o_proj / out_proj / K1 front: CTA b runs owners b + p grid < 160."""
    got = Counter(
        b + p * grid
        for b in range(grid)
        for p in range(T.owner_passes(grid))
        if b + p * grid < T.O_TASKS
    )
    _once(got, range(T.O_TASKS))


@pytest.mark.parametrize("tp", T.TP_SIZES)
@pytest.mark.parametrize("grid", GRIDS)
def test_gate_up_columns_once(grid, tp):
    """K2 gate_up: CTA b's whole groups g_full b + j, then columns sl_off ..
    sl_off + sl_cols of split group g_full grid + b / (16 / sl_cols)."""
    geo = T.FGeo(tp, grid)
    got = Counter()
    for b in range(grid):
        for j in range(geo.g_full):
            g = geo.g_full * b + j
            got.update((g, col) for col in range(T.ROWS))
        per = T.ROWS // geo.sl_cols
        g = geo.g_full * grid + b // per
        off = b % per * geo.sl_cols
        got.update((g, col) for col in range(off, off + geo.sl_cols))
    _once(got, [(g, col) for g in range(geo.ng) for col in range(T.ROWS)])
    # HQ: pairs of whole groups (j odd) are qdq blocks (g - 1) * 16 / 32
    blocks = Counter(
        (geo.g_full * b + j - 1) * T.ROWS // 32
        for b in range(grid)
        for j in range(1, geo.g_full, 2)
    )
    _once(blocks, range(geo.bulk_blocks))


@pytest.mark.parametrize("tp", T.TP_SIZES)
@pytest.mark.parametrize("grid", GRIDS)
def test_down_rows_once(grid, tp):
    """K2 down: pass p's tiles on the even / odd waves (``d_pass_tiles``),
    whole group i grid + b or the CTA's rows of the split group; the output
    threads write exactly those rows (``down_out`` / ``down_out_split``)."""
    geo = T.FGeo(tp, grid)
    loaded, written = Counter(), Counter()
    per = T.ROWS // geo.d_sl_rows
    for b in range(grid):
        for p in range(geo.d_passes):
            for i in geo.d_pass_tiles(p):
                if i is None:
                    continue
                if i < geo.d_per:
                    rows = range((i * grid + b) * T.ROWS, (i * grid + b + 1) * T.ROWS)
                else:
                    off = b % per * geo.d_sl_rows
                    base = (geo.d_full + b // per) * T.ROWS
                    rows = range(base + off, base + off + geo.d_sl_rows)
                loaded.update(rows)
                written.update(rows)
    _once(loaded, range(T.HIDDEN))
    _once(written, range(T.HIDDEN))


@pytest.mark.parametrize("tp", T.TP_SIZES)
@pytest.mark.parametrize("grid", GRIDS)
def test_qkvz_groups_once(grid, tp):
    """K1 in_proj_qkvz: pass p, wave pair q: group qg_cta b + p qg_pass + q."""
    geo = T.Geo(tp, grid)
    got = Counter(
        geo.qg_cta * b + p * geo.qg_pass + q
        for b in range(grid)
        for p in range(geo.qkvz_passes)
        for q in range(geo.qg_pass)
    )
    _once(got, range(geo.qkvz // T.ROWS))
    assert 2 * geo.qg_pass <= T.WAVES and geo.ba_ctas <= grid <= T.THREADS
    # a pass's outputs, THREADS a round, each (group, token, row) once
    for s in range(1, T.max_tokens(tp) + 1):
        n = geo.qg_pass * T.ROWS * s
        rounds = geo.qkvz_out_rounds(s)
        outs = Counter(
            u
            for i in range(rounds)
            for u in range(T.THREADS * i, T.THREADS * (i + 1))
            if u < n
        )
        _once(outs, range(n))
        assert rounds == 1 or grid < 256


def test_qkvz_output_rounds():
    """One round wherever #61573 ran (256 CTAs); two at TP2 on 128 CTAs above
    8 rows, where 4 groups x 16 rows x S outnumber the 512 threads."""
    for tp in T.TP_SIZES:
        geo = T.Geo(tp, 256)
        assert {geo.qkvz_out_rounds(s) for s in range(1, T.max_tokens(tp) + 1)} == {1}
    assert T.Geo(1, 128).qkvz_out_rounds(8) == 1
    assert [T.Geo(2, 128).qkvz_out_rounds(s) for s in (8, 9, 16)] == [1, 2, 2]


def _task_of(geo, s, b, k):
    """``dense_gdn.task_of`` in integers: (token, value head, chunk) or None
    for a dead task."""
    if geo.tpt % geo.grid == 0:
        tpc = geo.tpt // geo.grid
        w = (k % tpc) * geo.grid + b
        return k // tpc, w // T.NVCH, w % T.NVCH
    g = k * geo.grid + b
    if g >= geo.tpt * s:
        return None
    w = g % geo.tpt
    return g // geo.tpt, w // T.NVCH, w % T.NVCH


@pytest.mark.parametrize("tp", T.TP_SIZES)
@pytest.mark.parametrize("grid", GRIDS)
def test_gdn_tasks_once_and_passes_closed(grid, tp):
    """Every (token, head, chunk) runs once, no pass exceeds GDN_PASS_TASKS,
    and a pass's tokens are all of their tasks on every CTA: the hand-offs a
    task waits on (its key head's q / k, its value head's o, all of its token)
    are produced in the same pass, so no CTA waits on a later pass."""
    geo = T.Geo(tp, grid)
    for s in range(1, T.max_tokens(tp) + 1):
        passes = T.gdn_passes(geo, s)
        assert sum(n for _, n in passes) == T.n_tasks(geo, s)
        assert all(n <= T.GDN_PASS_TASKS for _, n in passes)
        got, pass_of_token = Counter(), {}
        for p, (k0, nk) in enumerate(passes):
            for b in range(grid):
                for k in range(k0, k0 + nk):
                    task = _task_of(geo, s, b, k)
                    if task is None:
                        continue
                    got[task] += 1
                    assert pass_of_token.setdefault(task[0], p) == p, (s, task)
        _once(
            got,
            [(t, h, c) for t in range(s) for h in range(geo.nv) for c in range(T.NVCH)],
        )


@pytest.mark.parametrize("grid", GRIDS)
def test_stage_plan_table(grid):
    rows = {r["stage"]: r for r in T.stage_plan(grid, 1, 8)}
    assert rows["in_proj_qkvz (16-row groups)"]["passes"] == (1 if grid == 256 else 2)
    assert rows["gdn (8 state rows a task)"]["per_cta"] == 6144 // grid
    assert rows["down (16-row groups)"]["passes"] == (1 if grid == 256 else 2)
    assert rows["o_proj + norm (32 columns a task)"]["passes"] == (
        1 if grid == 256 else 2
    )
