# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attention input projection of a decode step, one engine per prepared row count.

GB300, TP8 shard of the K3.1 KDA (6288 x 7168) and DSA (3808 x 7168) projections,
back-to-back launches over weights larger than L2, us:

  rows   KDA  split-K / warp / cuBLAS   DSA  split-K / warp / cuBLAS   (SOL KDA / DSA)
  1           14.54 / 13.41 / 16.89          10.93 /  8.60 / 12.23      11.02 / 6.68
  5           14.86 / 13.82 / 16.58          11.13 /  8.51 / 11.70      11.04 / 6.69
  10          15.30 / 14.64 / 16.89           9.82 /  8.91 / 11.54      11.05 / 6.70
  32          15.46 / 17.61 / 17.42          10.11 / 10.14 / 12.05      11.13 / 6.76
  64          16.58 /   -   / 16.18          11.95 /   -   / 11.75      11.23 / 6.85
  80          27.24 /   -   / 17.81          19.43 /   -   / 11.76      11.28 / 6.89
  160         30.72 /   -   / 21.81          27.72 /   -   / 12.67      11.55 / 7.10
  320         55.38 /   -   / 27.59          39.50 /   -   / 16.02      12.07 / 7.53

so up to 16 rows the warp split-K (many small co-resident CTAs keep the weight stream
at HBM speed), up to 64 the cluster split-K and past it cuBLAS (the split-K tiles stop
at 64 columns and re-read the weight per column tile)."""

import torch

from .projection import PreparedProjection
from .warp_splitk import (
    _CTA_RESERVED_SMEM,
    _SM_SMEM_BYTES,
    PreparedWarpSplitK,
    WarpSplitKTactic,
    _smem_bytes,
    check_tactic,
)

WARP_ROWS = 16  # warp split-K up to here
SPLIT_K_ROWS = 64  # cluster split-K up to here; cuBLAS past it


def warp_tactic(m, n, k, sms):
    """Warp split-K tactic of ``m`` rows of an [n, k] weight: 32-row output tiles
    when they fill at most two waves, the smallest token tile holding ``m``, 256-wide
    K tiles and the deepest ring that keeps every wave's CTAs resident on an SM."""
    output = 32 if n % 32 == 0 and n // 32 <= 2 * sms else 16
    token = 8 if m <= 8 else 16 if m <= 16 else 32
    k_tile = 256 if k % 256 == 0 and k // 256 != 2 else 128
    ctas = (n // output) * -(-m // token)
    resident = -(-ctas // sms)
    best = None
    for stages in range(2, min(16, k // k_tile) + 1):
        t = WarpSplitKTactic(output, token, k_tile, stages, 2)
        if resident * (_smem_bytes(t) + _CTA_RESERVED_SMEM) <= _SM_SMEM_BYTES:
            best = t
    if best is None:
        best = WarpSplitKTactic(output, token, k_tile, 2, 2)
    check_tactic(best, n, k)
    return best


def warp_eligible(n, k):
    return n % 16 == 0 and k % 128 == 0 and k // 128 <= 64


class DecodeProjection:
    """``out[M, N] = x[M, K] @ weight[N, K]^T`` for each prepared M: warp split-K up
    to ``WARP_ROWS``, the cluster split-K ``policies`` (M -> tactic) up to
    ``SPLIT_K_ROWS``, cuBLAS past it. ``skinny``: an optional engine for the rows it
    prepares (``run(x, out)`` with ``configs``)."""

    def __init__(self, weight, batches, policies, skinny=None):
        n, k = weight.shape
        sms = torch.cuda.get_device_properties(weight.device).multi_processor_count
        self.weight, self.engines = weight, {}
        warp = {}
        packed = weight.stride() == (k, 1) and weight.data_ptr() % 32 == 0
        if warp_eligible(n, k) and packed:
            warp = {b: warp_tactic(b, n, k, sms) for b in batches if b <= WARP_ROWS}
        split = {
            b: policies[b]
            for b in batches
            if b not in warp and b <= SPLIT_K_ROWS and b in policies
        }
        self.warp = PreparedWarpSplitK(weight, warp) if warp else None
        self.split = PreparedProjection(weight, split).prepare() if split else None
        self.kinds = {}
        for b in batches:
            if skinny is not None and b in skinny.configs:
                self.kinds[b], self.engines[b] = "skinny", skinny.run
            elif b in warp:
                self.kinds[b], self.engines[b] = "warp", self.warp.run
            elif b in split:
                self.kinds[b], self.engines[b] = "split_k", self.split.run
            else:
                self.kinds[b], self.engines[b] = "cublas", self.cublas

    def cublas(self, x, out):
        return torch.mm(x, self.weight.t(), out=out)

    def engine(self, rows):
        return self.kinds[rows]

    def run(self, x, out):
        rows = x.shape[0]
        if self.kinds[rows] == "warp" and not _warp_operands(x, out):
            # The warp split-K takes packed row-major, 32 B aligned operands only.
            return self.cublas(x, out)
        return self.engines[rows](x, out)


def _warp_operands(x, out):
    k, n = x.shape[1], out.shape[1]
    return (
        x.stride() == (k, 1)
        and out.stride() == (n, 1)
        and x.data_ptr() % 32 == 0
        and out.data_ptr() % 32 == 0
    )
