# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Re-export of ``QuantizedActivation``.

``as_quantized_activation`` performs an ``isinstance(x, QuantizedActivation)``
check; a separate class here would silently bypass it.
"""

from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)

__all__ = ["QuantizedActivation", "as_quantized_activation"]
