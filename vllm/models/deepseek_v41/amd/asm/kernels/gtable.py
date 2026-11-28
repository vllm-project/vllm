# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""Group-table work decode shared by the MoE stage kernels.

The prep (moe_prep.py, or the host in tests) writes a compact group table:
int32 [NG, 20] = e, 0, 0, 0, picks[16] (p = token*6 + k, -1 = none), one
record (80 B) per 16-pick group, expert-major, and counts[0] = NG. A stage
kernel with UPG units per group (18 row blocks for stage 1, 20 chunks for
stage 2) decodes unit u as group u // UPG, sub-index u % UPG. The grid is
fixed at `grid` CTAs (a power of two, default GRID). split "ceil" (stage 1):
CTA b takes units [b*per, b*per+per), per = ceil(n/grid), so up to ~10% of the
CTAs idle at the tail (memory-bound: idle CTAs cost nothing). "table" (stage
2): [counts[4 + b], counts[5 + b]), written by moe_prep (cost-balanced, UPG
20).

SGPRs used: s18 = n units, s19 = scratch, s50 = u_end, s52 = u_begin, s90 = g0
(first group of this CTA, its record at LDS 0), s91 = /UPG magic;
scratch s76..s81, s92.
"""

GRID = 256
REC = 80
MAGIC = {
    18: (0xE38E38E4, 4),
    20: (0xCCCCCCCD, 4),
}  # x / d = mulhi(x, m) >> sh (x < 2^20)


def range_and_copy(
    UPG: int,
    counts: int,
    groups: int,
    tv: list[int],
    end_label: str,
    grid: int = GRID,
    split: str = "ceil",
) -> list[str]:
    """Counts / groups: SGPR pairs (first index) holding the pointers.
    Counts are read with a vector sc1 load (the table is written earlier in the
    same chain, possibly by another XCD, and this XCD's L2 may still hold the
    previous launch's line). split "table" needs a third temp VGPR tv[2]."""
    assert split in ("ceil", "table"), split
    m, sh = MAGIC[UPG]
    lg = grid.bit_length() - 1
    assert grid == 1 << lg
    load = [
        f"  v_mov_b32 v{tv[0]}, 0",
        f"  global_load_dword v{tv[1]}, v{tv[0]}, s[{counts}:{counts + 1}] sc1",
        "  s_waitcnt vmcnt(0)",
        f"  v_readfirstlane_b32 s18, v{tv[1]}",
    ]
    if split == "table":
        assert UPG == 20 and len(tv) > 2
        rng = [
            "  s_lshl_b32 s80, s2, 2",
            f"  v_mov_b32 v{tv[0]}, s80",  # s19 may be counts' high dword
            f"  global_load_dword v{tv[1]}, v{tv[0]}, s[{counts}:{counts + 1}] offset:16 sc1",
            f"  global_load_dword v{tv[2]}, v{tv[0]}, s[{counts}:{counts + 1}] offset:20 sc1",
            "  s_waitcnt vmcnt(0)",
            f"  v_readfirstlane_b32 s52, v{tv[1]}",
            f"  v_readfirstlane_b32 s50, v{tv[2]}",
        ]
        load = []
    else:
        rng = [
            f"  s_mul_i32 s18, s18, {UPG}",  # units
            f"  s_add_u32 s19, s18, {grid - 1}",
            f"  s_lshr_b32 s19, s19, {lg}",  # per = ceil(n / grid)
            "  s_mul_i32 s52, s2, s19",
            "  s_add_u32 s50, s52, s19",
            "  s_min_u32 s50, s50, s18",
        ]
    o = [
        *load,
        f"  s_mov_b32 s91, {m:#x}",
        "  s_waitcnt lgkmcnt(0)",
        *rng,
        "  s_cmp_ge_u32 s52, s50",
        f"  s_cbranch_scc1 {end_label}",
        "  s_mul_hi_u32 s90, s52, s91",
        f"  s_lshr_b32 s90, s90, {sh}",  # g0
        "  s_sub_u32 s92, s50, 1",
        "  s_mul_hi_u32 s92, s92, s91",
        f"  s_lshr_b32 s92, s92, {sh}",  # g1
        "  s_sub_u32 s92, s92, s90",
        "  s_add_u32 s92, s92, 1",
        f"  s_mul_i32 s77, s92, {REC}",  # bytes to copy
        f"  s_mul_i32 s76, s90, {REC}",
        f"  s_add_u32 s78, s{groups}, s76",
        f"  s_addc_u32 s79, s{groups + 1}, 0",
        f"  v_lshlrev_b32 v{tv[0]}, 2, v0",
        f"{end_label}_copy:",
        f"  v_cmp_gt_u32 vcc, s77, v{tv[0]}",
        "  s_and_saveexec_b64 s[80:81], vcc",
        f"  s_cbranch_execz {end_label}_copied",
        f"  global_load_dword v{tv[1]}, v{tv[0]}, s[78:79]",
        "  s_waitcnt vmcnt(0)",
        f"  ds_write_b32 v{tv[0]}, v{tv[1]}",
        f"  v_add_u32 v{tv[0]}, 1024, v{tv[0]}",
        "  s_mov_b64 exec, s[80:81]",
        f"  s_branch {end_label}_copy",
        f"{end_label}_copied:",
        "  s_mov_b64 exec, -1",
        "  s_waitcnt lgkmcnt(0)",
        "  s_barrier",
    ]
    return o


def decode(UPG: int, u_sgpr: int, g: int, sub: int) -> list[str]:
    """s{g} = u // UPG (the gid), s{sub} = u % UPG, s76 = LDS offset of the record."""
    _, sh = MAGIC[UPG]
    return [
        f"  s_mul_hi_u32 s{g}, s{u_sgpr}, s91",
        f"  s_lshr_b32 s{g}, s{g}, {sh}",
        f"  s_mul_i32 s{sub}, s{g}, {UPG}",
        f"  s_sub_u32 s{sub}, s{u_sgpr}, s{sub}",
        f"  s_sub_u32 s76, s{g}, s90",
        f"  s_mul_i32 s76, s76, {REC}",
    ]
