# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FusedMoEMethodBase — re-exported from upstream.

The vendored DSv4 quant_config dispatches to upstream concrete
quant-method classes (``Fp8MoEMethod``, ``UnquantizedFusedMoEMethod``,
etc.) which inherit from upstream's ``FusedMoEMethodBase``. The
vendored ``RoutedExperts._get_quant_method`` then asserts
``isinstance(quant_method, FusedMoEMethodBase)``. If the vendored
class were a separate copy, the isinstance check would always fail.
Re-exporting upstream keeps both sides referencing the same ABC.

A dedicated lint carve-out
(``model_executor.layers.fused_moe.fused_moe_method_base``) lets this
through. Same justification as the
``MoEActivation`` / ``QuantizationConfig`` / ``QuantizeMethodBase``
carve-outs: it's a pure ABC that has to have identity-matching
cross-boundary.
"""

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)

__all__ = ["FusedMoEMethodBase"]
