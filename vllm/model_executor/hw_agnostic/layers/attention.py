# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-export of upstream ``Attention``.

The DSv4 quant config does ``isinstance(layer, Attention)`` against
upstream ``Attention`` instances built by the worker; vendoring a copy
would break that identity check.
"""

from vllm.model_executor.layers.attention import Attention

__all__ = ["Attention"]
