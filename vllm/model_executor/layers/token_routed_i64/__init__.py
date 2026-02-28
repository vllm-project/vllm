# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""
Token-Routed I64 — Deterministic expert routing for vLLM.

Components:
  - TokenRoutedMLP: expert routing via token_id % num_experts
  - INLDynamics: PID-like control with velocity tracking
  - fused_token_routed_forward: BMM/chunked expert dispatch

INL Innovation: no learned router, no softmax, no top-k.
Routing is deterministic integer arithmetic.
"""

from vllm.model_executor.layers.token_routed_i64.fused_experts import (
    fused_token_routed_forward,
)
from vllm.model_executor.layers.token_routed_i64.inl_dynamics import (
    INLDynamics,
)
from vllm.model_executor.layers.token_routed_i64.layer import TokenRoutedMLP

__all__ = [
    "TokenRoutedMLP",
    "INLDynamics",
    "fused_token_routed_forward",
]
