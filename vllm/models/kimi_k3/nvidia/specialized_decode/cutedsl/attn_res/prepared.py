# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared seven-CTA AttnRes consumers for ordinary or Lamport mailbox input."""

from functools import cache
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ..common.host import (
    architecture,
    dynamic_fake,
    dynamic_view,
    fake,
    stream,
    to_cute,
)
from ..common.policy import batch_table
from .kernel import ClusterAttnRes

# Rows whose AttnRes reads its saved-block statistics (logits, softmax maximum and
# exponent sum, FP32 mixture) precomputed by the previous layer's fused o-projection
# while that kernel waits for its input (``oproj.consumer`` ahead group), per
# (TP size, DCP size). runtime.decode applies it to KDA layers that follow a KDA
# layer and read one of the ATTN_STATS_BLOCKS saved-block counts (those whose
# o-projection with the statistics keeps the stock kernel's register allocation);
# other pairs, rows and layers keep the in-kernel statistics. Chosen by an
# end-to-end A/B at c = 1 (TP16 + DCP16, DSpark k = 7: 8 rows) and a one-GPU
# kernel-chain benchmark at 5 and 8 rows; other (TP, DCP) pairs were not measured.
# The outputs are bit-identical either way.
ATTN_STATS = {(16, 16): "5-8:1"}
ATTN_STATS_BLOCKS = {(16, 16): "2-5:1,7-8:1"}


def statistics_rows(tp, dcp):
    """The rows (ATTN_STATS) that take the precomputed statistics at (tp, dcp)."""
    table = batch_table(ATTN_STATS.get((tp, dcp), ""))
    return frozenset(b for b, on in table.items() if on)


def statistics_blocks(tp, dcp):
    """The saved-block counts (ATTN_STATS_BLOCKS) that may take them at (tp, dcp)."""
    table = batch_table(ATTN_STATS_BLOCKS.get((tp, dcp), ""))
    return frozenset(n for n, on in table.items() if on)


@cache
def compile_consumer(nb, delta, write, final, slots, cap, arch, aux=False, stats=0):
    """`stats`: the precomputed variant over [stats, 7168] / [stats, 2] FP32 rows."""
    row, weight, eps = dynamic_fake(2), fake((7168,), (1,)), cutlass.Float32(1e-6)
    source = fake((slots, cap, 7168), (cap * 7168, 7168, 1)) if slots else row
    kernel = ClusterAttnRes(
        nb,
        slots=slots,
        has_delta=delta,
        write=write,
        final=final,
        precomputed=stats > 0,
    )
    args: tuple = (source, row, dynamic_fake(3), weight, weight, weight, row, eps, eps)
    if stats:
        args += (fake((stats, 7168), None, torch.float32),)
        args += (fake((stats, 2), None, torch.float32),)
    args += (row,) * aux  # the updated prefix row, also stored to an aux stream
    return cute.compile(kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}")


class PreparedAttnRes:
    """Ordinary-input code plus the Lamport variant of its bound mailbox ring."""

    def __init__(self, weights, key, eps):
        self.device = weights[0].device
        self.views = tuple(to_cute(w.reshape(-1)) for w in weights)
        self.key, self.eps = key, tuple(map(cutlass.Float32, eps))
        self.code = self._compile(0, 0)
        self.ring: Any = None
        self.aux: Any = None
        self.stats: Any = None

    def _compile(self, slots, cap, aux=False, stats=0):
        arch = architecture(self.device)
        return compile_consumer(*self.key, slots, cap, arch, aux, stats)

    @classmethod
    def for_target(cls, target, *, final=False, has_delta=True):
        if final:
            norm, query = target.output_attn_res_norm, target.output_attn_res_proj
            output, nb, write = target.norm, target.num_attn_res_blocks, -1
        else:
            norm = target.self_attention_res_norm
            query, output = target.self_attention_res_proj, target.input_layernorm
            nb = target.prev_valid_blocks
            write = target.block_write_idx if target.is_block_write_layer else -1
        weights = (norm.weight, query.weight, output.weight)
        eps = (norm.variance_epsilon, output.variance_epsilon)
        return cls(weights, (nb, has_delta, write, final), eps)

    def bind_ring(self, mailbox):
        """Lamport input: sum the [slots, capacity, 7168] mailbox rows, then re-arm."""
        shape = tuple(mailbox.shape[:2])
        self.ring = self._compile(*shape), to_cute(mailbox), shape

    def bind_aux(self):
        """Variants that also store the updated prefix rows (a stock aux stream)."""
        ring = None
        if self.ring is not None:
            slots, cap = self.ring[2]
            ring = self._compile(slots, cap, True)
        self.aux = self._compile(0, 0, True), ring

    def bind_statistics(self, mixture, scal, rows):
        """`rows` run the precomputed variants of the bound codes (ordinary, ring,
        aux), reading the statistics from `mixture` [cap, 7168] and `scal`
        [cap, 2] FP32, which the previous kernel that computes them fills."""
        cap = mixture.shape[0]
        slots, ring_cap = (0, 0) if self.ring is None else self.ring[2]
        codes = [self._compile(0, 0, False, cap), None, None, None]
        if self.ring is not None:
            codes[1] = self._compile(slots, ring_cap, False, cap)
        if self.aux is not None:
            codes[2] = self._compile(0, 0, True, cap)
            if self.ring is not None:
                codes[3] = self._compile(slots, ring_cap, True, cap)
        views = to_cute(mixture), to_cute(scal)
        self.stats = frozenset(rows), codes, views

    def _variant(self, code, index, rows):
        """`code`, or its precomputed variant (`codes[index]`) and statistics."""
        if self.stats is None or rows not in self.stats[0]:
            return code, ()
        return self.stats[1][index], self.stats[2]

    def run_ring(self, prefix, blocks, out, aux=None):
        code = self.ring[0] if aux is None else self.aux[1]
        code, stats = self._variant(code, 1 if aux is None else 3, prefix.shape[0])
        return self._launch(code, self.ring[1], prefix, blocks, out, aux, stats)

    def run(self, delta, prefix, blocks, out, aux=None):
        code = self.code if aux is None else self.aux[0]
        code, stats = self._variant(code, 0 if aux is None else 2, prefix.shape[0])
        source = dynamic_view(delta)
        return self._launch(code, source, prefix, blocks, out, aux, stats)

    def _launch(self, code, source, prefix, blocks, out, aux=None, stats=()):
        rows = (prefix, blocks, out) + (() if aux is None else (aux,))
        prefix, blocks, view, *aux = map(dynamic_view, rows)
        args = *self.views, view, *self.eps, *stats, *aux
        code(source, prefix, blocks, *args, stream(self.device))
        return out
