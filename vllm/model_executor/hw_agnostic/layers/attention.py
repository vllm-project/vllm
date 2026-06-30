# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-export of ``Attention`` for hw-agnostic modelling code.

``Attention`` is the integration point between the hw-agnostic modelling
tree and the hw-specific attention / KV-cache infrastructure: layer
registration, KV-cache group discovery and the V1 attention dispatch all
key off the class object itself (``isinstance`` checks, class-name
lookups).

"""

from vllm.model_executor.layers.attention import Attention

__all__ = ["Attention"]
