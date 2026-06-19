# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE activation enum — re-exported from upstream.

Originally vendored as a copy of
``vllm/model_executor/layers/fused_moe/activation.py``, but kept as a
re-export so the ``MoEActivation`` enum has the SAME class identity as
upstream's. Otherwise the upstream FP8 MoE oracle's
``TritonExperts._supports_activation`` check (which compares the
config's ``activation`` against ``MoEActivation.SILU``) fails with
``"kernel does not support MoEActivation.SILU activation."`` because
the vendored ``MoEActivation.SILU`` is a different enum member from
the upstream one.

A dedicated lint carve-out
(``model_executor.layers.fused_moe.activation``) lets this re-export
through. The justification matches the
``QuantizationConfig`` / ``QuantizeMethodBase`` carve-out: it's a
pure data class that the vendored code uses to communicate with
upstream's quant-method registry; both sides must reference the same
enum identity for isinstance / equality checks to work.
"""

from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    activation_without_mul,
    apply_moe_activation,
)

__all__ = [
    "MoEActivation",
    "activation_without_mul",
    "apply_moe_activation",
]
