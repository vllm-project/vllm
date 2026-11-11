# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What a MoE layer of the glue paths can hand the next decoder layer instead of
all-reduced rows."""


class Mxfp8Carry:
    """The next layer's attention input, already normed and quantized to MXFP8
    (codes and row-major E8M0 scales of every token), and the new residual. The
    next layer's input norm is then skipped (`MiniMaxM3DecoderLayer.forward`)."""

    __slots__ = ("x_q", "x_s", "res_local", "rows_total")

    def __init__(self, x_q, x_s, res_local, rows_total):
        self.x_q, self.x_s = x_q, x_s
        self.res_local, self.rows_total = res_local, rows_total

    @property
    def shape(self):
        return self.x_q.shape
