# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Re-export of ``QuantKey`` and helpers. ``vllm/config/quantization.py`` keys
off ``isinstance`` and singleton ``kFp8...`` constant identity, so the class
and constants must be the canonical ones."""

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    create_fp8_quant_key,
    is_layer_skipped,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)

__all__ = [
    "GroupShape",
    "QuantKey",
    "create_fp8_quant_key",
    "is_layer_skipped",
    "kFp8DynamicTokenSym",
    "kFp8StaticTensorSym",
]
