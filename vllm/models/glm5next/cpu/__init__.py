# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU backend for the GLM5Next KDA operator.

Exposes the same two entry points as
``vllm.models.glm5next.nvidia.ops.third_party.kda`` so ``common/kda.py`` can
dispatch to this package without changing any call site.
"""

from .kda import (
    causal_conv1d_update_cpu,
    chunk_kda_with_fused_gate,
    fused_recurrent_kda,
    gather_initial_states_cpu,
    scatter_states_cpu,
)

__all__ = [
    "causal_conv1d_update_cpu",
    "chunk_kda_with_fused_gate",
    "fused_recurrent_kda",
    "gather_initial_states_cpu",
    "scatter_states_cpu",
]
