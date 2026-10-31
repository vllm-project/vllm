# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL collectives of MiniMax-M3's sequence-parallel prefill (TP4, MI355X)."""

from .host import WORLD, Collectives, Pools

__all__ = ["WORLD", "Collectives", "Pools"]
