# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 TP4 MoE-half paths on gfx950 (`runtime.M3Glue` decides per
forward which one, if any, a MoE layer takes).

SP prefill: eager prefill steps run a sequence-parallel norm / router / shared
expert / quant around the stock AITER MoE (`sp.py`)."""

from .runtime import M3Glue

__all__ = ["M3Glue"]
