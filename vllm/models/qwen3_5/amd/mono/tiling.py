# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The grid a dense mono build launches, and how each stage's tasks fall on it.

Every K1 / K2 launch is one persistent CTA per compute unit, all of them
co-resident: a spin wait on another CTA progresses only while that CTA runs. So
the grid width is the GPU's compute-unit count. The kernels were laid out for
MI355X's 256; this module derives every stage's per-CTA work from the width, so
a 128-CU gfx950 (MI350P, 4 XCDs) runs the same stage bodies, the stages whose
tasks outnumber the CTAs in more passes a CTA:

    stage            tasks (TP1)        256 CTAs             128 CTAs
    o_proj / front   160 owners         CTAs 0..159          2 passes on 0..31
    in_proj_qkvz     1024 groups        4 a CTA, 1 pass      8 a CTA, 2 passes
    gdn              768 S              3 S a CTA, 1 pass    6 S, 2 passes at S > 4
    gate_up          1088 groups        4 + 4 columns        8 + 8 columns
    down             320 row groups     1 + 4 rows, 1 pass   2 + 8 rows, 2 passes

A task is computed the same way whichever CTA and pass runs it (same lanes, same
K chains, same reduction order), so the outputs do not depend on the width. At
the 256-CTA width every stage is one pass and traces as before.

Nothing depends on the XCD count: hand-offs are device scope (past the per-XCD
L2s) and no task is placed by XCD.

Pure Python (no FlyDSL, no torch at import): the runtime gate and the CPU tests
import it.
"""

import functools
import os
from typing import NamedTuple

THREADS = 512
WAVES = THREADS // 64

# The grid the kernels were written for: one CTA per CU of an MI355X / MI350X.
DEFAULT_GRID = 256
# The widths whose stage plans are implemented. 256: MI355X, MI350X; 128:
# MI350P (and a 128-CU partition of a 256-CU part).
VALIDATED_GRIDS = (256, 128)
# Override: run a validated width narrower than the GPU (the 128-CTA build on a
# 256-CU MI355X, say). It may not exceed the compute units.
GRID_ENV = "VLLM_ROCM_QWEN38_DENSE_MONO_GRID"

# Qwen3.8-27B (TP1 shapes)
HIDDEN = 5120
INTER = 17408
CORE = 6144  # o_proj / out_proj K
ROWS = 16  # rows of a weight group (an MFMA tile's N)
STEP = 128  # K an fp4 tile step covers
QUARTERS = 4  # a K2 tile's K chain cut into quarters, one a wave pair
MAX_TOKENS = 8
MAX_TOKENS_TP2 = 16
TP_SIZES = (1, 2)

# The o_proj / out_proj owners and K1's front: 32 hidden columns a task.
O_TASKS = HIDDEN // (2 * ROWS)  # 160

# GDN
HD = 128
NK = 16
NV = 48
BA_E = 8  # K eighths of an in_proj_ba group
VCH = 8  # state rows a GDN task
NVCH = HD // VCH  # 16 tasks a value head
# The most GDN tasks a CTA runs in one pass: the widest single-pass load of a
# validated 256-CTA build (TP1 at 8 rows, TP2 at 16). More tasks than this go in
# passes, so a pass's registers (the state rows in flight) and LDS stay within
# what that build uses.
GDN_PASS_TASKS = 24


def max_tokens(tp: int) -> int:
    """The widest step K1 / K2 build for at TP ``tp``."""
    return MAX_TOKENS_TP2 if tp == 2 else MAX_TOKENS


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def resolve_grid(cus: int, override: str | int | None = None) -> int:
    """The grid width for a gfx950 with ``cus`` compute units: ``cus`` when it
    is a validated width, else ``override`` (``GRID_ENV``) when that is a
    validated width no wider than ``cus``.

    Raises ValueError on any other count: a grid wider than the compute units
    deadlocks (a waiting CTA spins on one that never becomes resident), and a
    width without a validated stage plan is refused rather than guessed."""
    valid = ", ".join(str(g) for g in VALIDATED_GRIDS)
    if override is not None and str(override).strip() != "":
        try:
            grid = int(override)
        except ValueError:
            raise ValueError(f"{GRID_ENV}={override!r} is not an integer") from None
        if grid not in VALIDATED_GRIDS:
            raise ValueError(
                f"{GRID_ENV}={grid}: not a validated grid width (validated: {valid})"
            )
        if grid > cus:
            raise ValueError(
                f"{GRID_ENV}={grid}: wider than the GPU's {cus} compute units "
                "(every CTA must be resident)"
            )
        return grid
    if cus in VALIDATED_GRIDS:
        return cus
    raise ValueError(
        f"no validated grid width for {cus} compute units (validated: {valid}; "
        "256 is MI355X / MI350X, 128 is MI350P). Set "
        f"{GRID_ENV} to a validated width below {cus} to run that build anyway"
    )


def _compute_units() -> int:
    """The current device's compute-unit count (imports torch and the platform
    lazily; the tests replace this function)."""
    import torch

    from vllm.platforms import current_platform

    return current_platform.num_compute_units(torch.cuda.current_device())


def device_grid() -> int:
    """``resolve_grid`` for the current device and ``GRID_ENV``."""
    return resolve_grid(_compute_units(), os.environ.get(GRID_ENV))


@functools.cache
def _resident_cus() -> int:
    return _compute_units()


def check_resident(grid: int) -> None:
    """Refuse a build whose grid is wider than the device's compute units: its
    launch would hang rather than fail."""
    cus = _resident_cus()
    if grid > cus:
        try:
            hint = f"; build with grid={resolve_grid(cus, os.environ.get(GRID_ENV))}"
        except ValueError as exc:
            hint = f"; {exc}"
        raise RuntimeError(
            f"a {grid}-CTA dense mono build on a GPU with {cus} compute units "
            "would deadlock (every CTA must be resident)" + hint
        )


def owner_passes(grid: int) -> int:
    """Passes a CTA makes over the O_TASKS owner tasks (task ``bid + p grid``)."""
    return _cdiv(O_TASKS, grid)


# ---------------------------------------------------------------- K2 (FFN)
class FGeo(NamedTuple):
    """A TP rank's share of the FFN on a ``grid``-CTA launch.

    gate_up: half the intermediate at TP2 (gate_up's N, down's K) in groups of
    16 columns; a CTA takes ``g_full`` whole groups, then ``sl_cols`` columns of
    one of the groups left over. TP1: 1088 groups, 4 whole and 4 columns of the
    64 others at 256 CTAs, 8 and 8 at 128; TP2: 544, 2 and 2 / 4 and 4.

    down: the 320 hidden row groups; a CTA takes ``d_per`` whole ones, then
    ``d_sl_rows`` rows of one of the ``d_sl`` left over: 1 and 4 at 256, 2 and
    8 at 128. Its tiles (whole groups, then the split one) go two a pass, one
    on the even waves and one on the odd (``d_pass_tiles``)."""

    tp: int
    grid: int = DEFAULT_GRID

    @property
    def inter(self) -> int:
        return INTER // self.tp

    @property
    def ng(self) -> int:  # groups of 16 intermediate columns
        return self.inter // ROWS

    @property
    def g_full(self) -> int:  # whole groups a CTA
        return self.ng // self.grid

    @property
    def sl_cols(self) -> int:  # a CTA's columns of the split groups
        return self.ng % self.grid * ROWS // self.grid

    @property
    def blocks(self) -> int:  # qdq blocks of 32 columns a token
        return self.inter // 32

    @property
    def bulk_blocks(self) -> int:  # the whole groups' blocks (HQ)
        return self.g_full * self.grid * ROWS // 32

    @property
    def d_steps(self) -> int:
        return self.inter // STEP

    @property
    def d_q(self) -> int:  # a wave's K quarter of down
        return self.d_steps // QUARTERS

    @property
    def w2_dw(self) -> int:
        return self.inter // 8

    @property
    def w2s_dw(self) -> int:
        return self.inter // 32 // 4

    @property
    def hqrow(self) -> int:
        return self.inter // 8 + 4

    @property
    def hsrow(self) -> int:
        return self.inter // 32

    # down
    @property
    def d_per(self) -> int:  # whole row groups a CTA (group i grid + bid)
        return HIDDEN // ROWS // self.grid

    @property
    def d_full(self) -> int:  # the whole groups, all CTAs
        return self.d_per * self.grid

    @property
    def d_sl(self) -> int:  # split row groups
        return HIDDEN // ROWS - self.d_full

    @property
    def d_sl_rows(self) -> int:  # a CTA's rows of its split group
        return ROWS * self.d_sl // self.grid

    @property
    def d_tiles(self) -> int:  # tiles a CTA: d_per whole, then the split one
        return self.d_per + 1

    @property
    def d_passes(self) -> int:
        return _cdiv(self.d_tiles, 2)

    def d_pass_tiles(self, p: int) -> tuple[int | None, int | None]:
        """Pass p's tiles (even waves, odd waves): tile i < d_per is whole group
        i grid + bid, tile d_per the split group's rows, None none."""
        return tuple(i if i < self.d_tiles else None for i in (2 * p, 2 * p + 1))

    @property
    def o_passes(self) -> int:
        return owner_passes(self.grid)


def check_ffn(geo: FGeo) -> None:
    """The layout invariants K2's stage bodies rely on; ValueError if broken."""
    why = []
    if not geo.grid <= THREADS:
        why.append("one ready flag a thread")
    if not (geo.g_full >= 2 and geo.g_full % 2 == 0):
        why.append(f"g_full {geo.g_full}: pairs of whole groups make the qdq blocks")
    if geo.ng % geo.grid * ROWS % geo.grid != 0:
        why.append("the split groups' columns do not divide over the CTAs")
    if not (geo.sl_cols > 0 and geo.sl_cols % 2 == 0 and ROWS % geo.sl_cols == 0):
        why.append(f"sl_cols {geo.sl_cols}: column pairs, a whole share of a group")
    if geo.d_steps % QUARTERS != 0:
        why.append("down's K steps do not divide in quarters")
    if not (geo.d_per >= 1 and geo.d_sl > 0 and geo.d_sl * ROWS % geo.grid == 0):
        why.append(f"down: {geo.d_per} whole and {geo.d_sl} split row groups")
    if not (geo.d_sl_rows % 2 == 0 and geo.d_sl_rows > 0 and ROWS % geo.d_sl_rows == 0):
        why.append(f"d_sl_rows {geo.d_sl_rows}: row pairs, a whole share of a group")
    if why:
        raise ValueError(f"K2 at TP{geo.tp} on {geo.grid} CTAs: " + "; ".join(why))


# ---------------------------------------------------------------- K1 (GDN)
class Geo(NamedTuple):
    """A rank's GDN shapes at tensor parallel size ``tp`` on a ``grid``-CTA
    launch: 1 / tp of the key and value heads, each of in_proj_qkvz's q | k | v
    | z, in_proj_ba's b | a and the conv's q | k | v sliced contiguously (the
    stock column-parallel split)."""

    tp: int
    grid: int = DEFAULT_GRID

    @property
    def nk(self) -> int:
        return NK // self.tp

    @property
    def nv(self) -> int:
        return NV // self.tp

    @property
    def key(self) -> int:
        return self.nk * HD

    @property
    def val(self) -> int:
        return self.nv * HD

    @property
    def qkvz(self) -> int:
        return 2 * self.key + 2 * self.val

    @property
    def ba(self) -> int:
        return 2 * self.nv

    @property
    def conv(self) -> int:
        return 2 * self.key + self.val

    @property
    def qg_cta(self) -> int:
        """The CTA's in_proj_qkvz groups of 16 rows: 4 [2 at TP2] on 256 CTAs, 8
        [4] on 128."""
        return self.qkvz // ROWS // self.grid

    @property
    def qg_pass(self) -> int:
        """The in_proj_qkvz groups a pass: wave w group w / 2, K half w % 2."""
        return min(self.qg_cta, WAVES // 2)

    @property
    def qkvz_passes(self) -> int:
        return self.qg_cta // self.qg_pass

    @property
    def ba_ctas(self) -> int:
        """CTAs of in_proj_ba: its groups x K eighths (48 at TP1, 24 at TP2)."""
        return self.ba // ROWS * BA_E

    @property
    def tpt(self) -> int:
        """Tasks a token (768 at TP1, 384 at TP2)."""
        return self.nv * NVCH

    @property
    def o_passes(self) -> int:
        return owner_passes(self.grid)


def n_tasks(geo: Geo, s: int) -> int:
    """Tasks a CTA at S rows: task k of CTA b is ``k grid + b`` of the S x tpt
    (3 S at TP1 on 256 CTAs, 6 S on 128; at TP2 ceil(1.5 S) on 256, the last
    one dead on half the CTAs at odd S, and 3 S on 128)."""
    return _cdiv(geo.tpt * s, geo.grid)


def gdn_passes(geo: Geo, s: int) -> tuple[tuple[int, int], ...]:
    """The CTA's GDN tasks in passes, (first task, tasks) each.

    One pass when all of them fit ``GDN_PASS_TASKS``. Otherwise a pass is a run
    of whole tokens: each token is ``tpt / grid`` consecutive tasks on every CTA
    (``tpt`` a multiple of the grid), and every hand-off a task waits on (its key
    head's q / k, its value head's o) comes from a task of the same token, so all
    CTAs produce what a pass waits on in that pass. The tokens are split as
    evenly as the passes allow."""
    nt = n_tasks(geo, s)
    if nt <= GDN_PASS_TASKS:
        return ((0, nt),)
    if geo.tpt % geo.grid:
        raise ValueError(
            f"K1 at TP{geo.tp} on {geo.grid} CTAs: {nt} GDN tasks a CTA at {s} rows"
            f" need passes, and a token's {geo.tpt} tasks do not divide over the CTAs"
        )
    tpc = geo.tpt // geo.grid
    if tpc > GDN_PASS_TASKS:
        raise ValueError(f"K1: one token's {tpc} tasks a CTA exceed a pass")
    n = _cdiv(s, GDN_PASS_TASKS // tpc)
    out, t0 = [], 0
    for p in range(n):
        tokens = s // n + (p < s % n)
        out.append((t0 * tpc, tokens * tpc))
        t0 += tokens
    return tuple(out)


def check_gdn(geo: Geo) -> None:
    """The layout invariants K1's stage bodies rely on; ValueError if broken."""
    why = []
    if not geo.grid <= THREADS:
        why.append("one ready flag a thread")
    if geo.qg_cta * ROWS * geo.grid != geo.qkvz:
        why.append("in_proj_qkvz's groups do not divide over the CTAs")
    if not (geo.qg_cta >= 1 and geo.qg_cta % geo.qg_pass == 0):
        why.append(f"{geo.qg_cta} qkvz groups a CTA in passes of {geo.qg_pass}")
    if not (geo.ba % ROWS == 0 and geo.ba_ctas <= geo.grid and geo.nv % geo.nk == 0):
        why.append("in_proj_ba's groups x eighths need a CTA each")
    if why:
        raise ValueError(f"K1 at TP{geo.tp} on {geo.grid} CTAs: " + "; ".join(why))
    for s in range(1, max_tokens(geo.tp) + 1):
        gdn_passes(geo, s)


for _tp in TP_SIZES:
    for _grid in VALIDATED_GRIDS:
        check_ffn(FGeo(_tp, _grid))
        check_gdn(Geo(_tp, _grid))


# ---------------------------------------------------------------- stage table
def stage_plan(grid: int, tp: int, s: int, oproj: bool = True) -> list[dict]:
    """Each K1 / K2 stage's task count and per-CTA passes at this width, TP and
    row count: what the design notes tabulate and the tests pin. ``oproj``: K2
    folds o_proj (TP1's default)."""
    f, g = FGeo(tp, grid), Geo(tp, grid)
    passes = gdn_passes(g, s)
    rows = []
    if tp == 1:
        rows.append(
            dict(kernel="K1", stage="front (norm, 32 columns a task)",
                 tasks=O_TASKS, passes=g.o_passes)
        )  # fmt: skip
    rows += [
        dict(kernel="K1", stage="in_proj_qkvz (16-row groups)",
             tasks=g.qkvz // ROWS, per_cta=g.qg_cta, passes=g.qkvz_passes),
        dict(kernel="K1", stage="in_proj_ba (group x K eighth)",
             tasks=g.ba_ctas, passes=1),
        dict(kernel="K1", stage="gdn (8 state rows a task)", tasks=g.tpt * s,
             per_cta=n_tasks(g, s), passes=len(passes),
             pass_tasks=tuple(n for _, n in passes)),
    ]  # fmt: skip
    if tp == 2:
        rows.append(
            dict(kernel="K1", stage="out_proj (32 columns a task)",
                 tasks=O_TASKS, passes=g.o_passes)
        )  # fmt: skip
    if tp == 1 and oproj:
        rows.append(
            dict(kernel="K2", stage="o_proj + norm (32 columns a task)",
                 tasks=O_TASKS, passes=f.o_passes)
        )  # fmt: skip
    rows += [
        dict(kernel="K2", stage="gate_up (16-column groups)", tasks=f.ng,
             per_cta=f.g_full, split_cols=f.sl_cols, passes=f.g_full + 1),
        dict(kernel="K2", stage="down (16-row groups)", tasks=HIDDEN // ROWS,
             per_cta=f.d_per, split_rows=f.d_sl_rows, passes=f.d_passes),
    ]  # fmt: skip
    return rows
