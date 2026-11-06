# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1c ``mla_attn``'s launch geometry, without the kernel's FlyDSL imports: the
runtime's config checks (``runtime.m1c_serves``) size M1c's KV splits on any
platform."""

from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import BLOCKS

NH = 12  # local heads at TP8 (``mla.NH``)
XCDS = 8
TILE = 16  # tokens a wave's tile
RTILES = {4: 1, 8: 2}  # row tiles a CTA by qlen (3 head groups either way)


def splits(qlen: int, rt: int = 0) -> int:
    """KV splits of the launch (every request's together): the head groups'
    CTAs an XCD hold a whole number of splits."""
    return XCDS * geometry(qlen, rt)[2]


def geometry(qlen: int, rt: int = 0):
    """(heads a CTA, head groups, splits an XCD); a CTA ``rt`` row tiles of 16
    (``RTILES`` by default)."""
    ht = TILE * (rt or RTILES[qlen]) // qlen
    hg = NH // ht
    return ht, hg, BLOCKS // XCDS // hg


def ctas(qlen: int, rt: int = 0) -> int:
    """The working CTAs (0 .. ctas): a done flag each when fused (``cta``'s
    ``done``), at (split, head group) ``split * hg + group``."""
    _, hg, spx = geometry(qlen, rt)
    return XCDS * spx * hg
