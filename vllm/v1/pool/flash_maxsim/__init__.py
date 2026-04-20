# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored flash-maxsim Triton kernels for ColBERT/ColPali MaxSim scoring.

Originally from https://github.com/roipony/flash-maxsim (Apache 2.0).
Forward-pass only; backward/training kernels are not included.
"""

from vllm.v1.pool.flash_maxsim.flash_maxsim_rerank import flash_maxsim_rerank_direct
from vllm.v1.pool.flash_maxsim.flash_maxsim_varlen import (
    flash_maxsim_packed,
    pack_docs,
)

__all__ = [
    "flash_maxsim_packed",
    "flash_maxsim_rerank_direct",
    "pack_docs",
]
