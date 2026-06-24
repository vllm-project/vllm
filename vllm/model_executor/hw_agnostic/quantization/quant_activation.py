# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Re-export of upstream ``QuantizedActivation``.

The upstream module performs an ``isinstance(x, QuantizedActivation)``
check inside ``as_quantized_activation``; a vendored copy would
silently bypass that check.
"""

from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)

__all__ = ["QuantizedActivation", "as_quantized_activation"]
