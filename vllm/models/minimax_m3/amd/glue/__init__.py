# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 TP4 MoE-half paths on gfx950 (`runtime.M3Glue` decides per
forward which one, if any, a MoE layer takes).

Decode glue: pure-decode steps of the compiled widths run one FlyDSL launch from
the o_proj partial to the MoE inputs (`kernel.py`, `decode.py`).
"""

from .runtime import M3Glue

__all__ = ["M3Glue"]
