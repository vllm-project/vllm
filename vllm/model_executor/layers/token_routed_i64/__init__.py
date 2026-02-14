# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""
Token-Routed I64 - Deterministic expert routing layer for vLLM.

INL Innovation: token_id % num_experts (no learned router).
"""

from vllm.model_executor.layers.token_routed_i64.layer import TokenRoutedMLP

__all__ = ["TokenRoutedMLP"]
