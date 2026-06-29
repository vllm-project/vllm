# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Re-export of ``QuantKey`` and helpers. ``vllm/config/quantization.py`` keys
off ``isinstance`` and singleton ``kFp8...`` constant identity, so the class
and constants must be the canonical ones."""

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    create_fp8_quant_key,
    is_layer_skipped,
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt4Static,
    kInt4Static32,
    kInt4Static32Asym,
    kInt4StaticAsym,
    kInt4W4A8StaticGroup32Sym,
    kInt4W4A8StaticGroup64Sym,
    kInt4W4A8StaticGroup128Sym,
    kInt4W4A8StaticGroupSym,
    kInt8DynamicTokenSym,
    kInt8Static,
    kInt8StaticChannelSym,
)

__all__ = [
    "GroupShape",
    "QuantKey",
    "ScaleDesc",
    "create_fp8_quant_key",
    "is_layer_skipped",
    "kFp8Dynamic128Sym",
    "kFp8Dynamic64Sym",
    "kFp8DynamicTensorSym",
    "kFp8DynamicTokenSym",
    "kFp8Static128BlockSym",
    "kFp8StaticChannelSym",
    "kFp8StaticTensorSym",
    "kInt4Static",
    "kInt4Static32",
    "kInt4Static32Asym",
    "kInt4StaticAsym",
    "kInt4W4A8StaticGroup32Sym",
    "kInt4W4A8StaticGroup64Sym",
    "kInt4W4A8StaticGroup128Sym",
    "kInt4W4A8StaticGroupSym",
    "kInt8DynamicTokenSym",
    "kInt8Static",
    "kInt8StaticChannelSym",
]
