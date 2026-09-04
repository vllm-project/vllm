# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse attention indexer (kpool) — hardware-isolated entry point.

The implementation lives under ``nvidia/`` and ``amd/`` (with shared helpers
in ``common/``); this module picks the right one for the current platform.
"""

from vllm.platforms import current_platform

if current_platform.is_rocm():
    from .amd.sparse_indexer import (
        SparseAttnIndexerKpool,
    )
else:
    from .nvidia.sparse_indexer import (
        SparseAttnIndexerKpool,
    )

__all__ = [
    "SparseAttnIndexerKpool",
]
