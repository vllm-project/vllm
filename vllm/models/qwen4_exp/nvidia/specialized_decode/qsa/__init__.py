# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The QSA decode kernels of the specialized decode, in CuTe DSL.

`qsa_row_attention` computes the stock `qsa_sparse_paged_attention`
operator for a BF16 KV cache with one CTA per (row, KV head) instead of the
stock split-K kernel and its merge (`attention.py`).

`qsa_select_expand` replaces the stock top-k and index expansion of a
decode step's indexer with one cluster kernel a row that reads the stock score
kernel's logits and writes the packed token list the sparse attention reads
(`select.py`).
"""

from .attention import qsa_row_attention, row_attention_unsupported
from .select import (
    MAX_BLOCKS,
    PACKED_WIDTH,
    RATIO,
    TOKENS,
    VARIANTS,
    default_variant,
    qsa_select_expand,
)

__all__ = [
    "MAX_BLOCKS",
    "PACKED_WIDTH",
    "RATIO",
    "TOKENS",
    "VARIANTS",
    "default_variant",
    "qsa_row_attention",
    "qsa_select_expand",
    "row_attention_unsupported",
]
