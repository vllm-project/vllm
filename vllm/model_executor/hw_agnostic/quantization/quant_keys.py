# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Re-export of upstream ``QuantKey`` and friends.

The upstream registry at ``vllm/config/quantization.py`` and several
attention backends do ``isinstance(v, QuantKey)`` and ``key == kFp8...Sym``
against the upstream class and module-level constants. Vendoring fresh
copies would either break the isinstance check (different class identity)
or silently bypass the registry (different singleton identity).

This module exposes the upstream symbols under a hw_agnostic-shaped path
so the lint regex no longer needs an ``utils.quant_utils`` carve-out.
"""

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
