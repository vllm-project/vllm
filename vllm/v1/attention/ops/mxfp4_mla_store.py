# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Old import path, kept for PRs stacked on this one that still import it.

Use :mod:`vllm.v1.attention.ops.mxfp4_mla.store`.
"""

from vllm.v1.attention.ops.mxfp4_mla.store import store_mxfp4_mla

__all__ = ["store_mxfp4_mla"]
