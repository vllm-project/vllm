# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""
Token-Routed I64 — Deterministic expert routing for vLLM.

Components:
  - TokenRoutedMLP: expert routing via token_id % num_experts
  - INLDynamics: PID-like control with velocity tracking
  - fused_i64_experts: CUDA graph safe BMM expert dispatch
  - I64Router: deterministic modulo router with mu-guided bias
  - I64ExpertRunner: runner wrapper for fused_i64_experts

INL Innovation: no learned router, no softmax, no top-k.
Routing is deterministic integer arithmetic.
"""

from vllm.model_executor.layers.token_routed_i64.fused_i64_moe import (
    fused_i64_experts,
)
from vllm.model_executor.layers.token_routed_i64.inl_dynamics import (
    INLDynamics,
)
from vllm.model_executor.layers.token_routed_i64.layer import TokenRoutedMLP
from vllm.model_executor.layers.token_routed_i64.router import I64Router
from vllm.model_executor.layers.token_routed_i64.runner import I64ExpertRunner

__all__ = [
    "TokenRoutedMLP",
    "INLDynamics",
    "fused_i64_experts",
    "I64Router",
    "I64ExpertRunner",
]
