# SPDX-License-Identifier: Apache-2.0
"""Genesis decision oracles — single source of truth for routing logic."""
from __future__ import annotations

from vllm._genesis.oracle.moe_select import select_moe_expert_impl

__all__ = ["select_moe_expert_impl"]
