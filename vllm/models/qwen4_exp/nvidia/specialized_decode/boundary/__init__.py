# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sublayer boundary kernels of the Qwen4Exp specialized decode.

A sublayer boundary is where a decoder block's output re-enters the residual
streams: the tensor-parallel all-reduce, the hyper-connection combine and the
next read's grouped norm, which the stock decode runs as two to five kernels.
`SublayerBoundary` (CuTe DSL) runs them as one, after attention blocks
and, with the shared expert's gate and the routed add folded in, after MoE
blocks.
"""

from .all_reduce_hc import SublayerBoundary

__all__ = ["SublayerBoundary"]
