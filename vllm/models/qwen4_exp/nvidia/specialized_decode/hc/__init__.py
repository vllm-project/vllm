# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hyper-connection read kernels of the Qwen4Exp specialized decode.

Every decoder block (attention, MoE) of Qwen4Exp reads its input from four
residual streams of 2560 values (`[rows, 10240]`). The stock read
(`GatedResidual.combine_and_mix`) is four kernels: the combine-norm
(previous block's output into the streams, grouped Gemma RMSNorm into `xn`),
the down projection with its SiLU (`hc_down_silu`), the up projection (a
BF16 GEMM into a `[rows, 10240]` gate) and the gated stream mean
(`hc_gate_mix`). At decode sizes the two projections each stream 6.6 MB of
replicated weight for a handful of rows, about 100 reads per step.

This package replaces the last three with two kernels that load their whole
weight share into registers before the grid-dependency wait, so the weights
stream while the previous kernel runs (programmatic dependent launch):

* `hc_down`: the down projection with the stock epilogue and rounding;
* `hc_up_mix`: the up projection with the sigmoid gate (kept in FP32),
  the product with `xn` and the stream mean in its epilogue.

The combine-norm stays the stock kernel, run with its option to launch the
next kernel right after its own grid-dependency wait. Both kernels serve 1-64
token rows and are capturable in CUDA graphs; `.read` composes them into
the executor's read (`hc_read`). `.down_tc.hc_down_tc` computes the
down projection on the tensor cores (tcgen05) for up to 128 rows; TP1 reads
take it, followed by the fused up projection and gate mix of `..ops.hc`,
from `read.TC_MIN_ROWS` rows.
"""

from .down import hc_down
from .up_mix import hc_up_mix

__all__ = ["hc_down", "hc_up_mix"]
