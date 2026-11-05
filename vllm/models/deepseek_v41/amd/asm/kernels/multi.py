# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""Several kernels as one launch, side by side: CTAs first ..
first + n - 1 run body ``tag`` with s2 = CTA - first, s[0:1] at that body's
block of the combined kernargs (laid out exactly as the body's own KArgs,
names prefixed ``tag.``), v0 = thread id. Bodies are the kernels' code
unchanged except for their labels (``.Ltag_*``); ``s_endpgm`` stays (a CTA
runs one body). The launch gets the bodies' largest VGPR / AGPR / SGPR / LDS
counts and block size, so co-residency is the largest body's.

    k = multi.source("ffn_prep_sdn", [("prep", prep_k, 1), ("sdn", sdn_k, 160)])
    Kernel(k["name"], k["src"], k["ka"])(k["grid"], k["block"], **{"prep.x": .., "sdn.w": ..})
"""

import regex as re

from vllm.models.deepseek_v41.amd.asm.kernels.kmod import KArgs, kernel_source


def _use(src: str, key: str) -> int:
    return int(re.search(rf"\.amdhsa_{key} (\d+)", src).group(1))


def source(name: str, parts: list[tuple[str, dict, int]]):
    """One launch of ``parts``: (tag, a kernel generator's dict with name / src
    / ka, CTA count) each."""
    args, bases, c = [], {}, []
    for tag, k, _ in parts:
        args += [(f"{tag}.{n}", kind) for n, kind in k["ka"].args]
    ka = KArgs(*args)
    vg = ag = sg = lds = 0
    block = 0
    first = 0
    for i, (tag, k, n) in enumerate(parts):
        base = ka.off[f"{tag}.{k['ka'].args[0][0]}"]
        assert all(
            ka.off[f"{tag}.{a}"] - base == k["ka"].off[a] for a, _ in k["ka"].args
        ), tag
        bases[tag] = (first, n, base)
        first += n
    grid = first
    # dispatch: the last part takes the rest
    for tag, k, n in parts[:-1]:
        f, _, _ = bases[tag]
        c.append(f"  s_cmp_lt_u32 s2, {f + n}")
        c.append(f"  s_cbranch_scc1 .L{tag}_enter")
    c.append(f"  s_branch .L{parts[-1][0]}_enter")
    for tag, k, n in parts:
        src, kname = k["src"], k["name"]
        f, _, base = bases[tag]
        body = src.split(f"\n{kname}:\n", 1)[1].split("\n.Lfunc_end:", 1)[0]
        c.append(f".L{tag}_enter:")
        if f:
            c.append(f"  s_sub_u32 s2, s2, {f}")
        if base:
            c += [f"  s_add_u32 s0, s0, {base}", "  s_addc_u32 s1, s1, 0"]
        c += [
            re.sub(r"\.L(\w+)", lambda mt, tag=tag: f".L{tag}_{mt.group(1)}", ln)
            for ln in body.split("\n")
        ]
        acc = _use(src, "accum_offset")
        vg, ag = max(vg, acc), max(ag, _use(src, "next_free_vgpr") - acc)
        sg, lds = (
            max(sg, _use(src, "next_free_sgpr")),
            max(lds, _use(src, "group_segment_fixed_size")),
        )
        block = max(block, k.get("block", 256))
    src = kernel_source(
        name,
        c,
        ka,
        lds=lds,
        vgprs=vg,
        agprs=(ag + 3) // 4 * 4,
        sgprs=min(max(sg, 16), 102),
        block=block,
    )
    return dict(name=name, src=src, ka=ka, grid=grid, block=block, vgprs=vg, agprs=ag)
