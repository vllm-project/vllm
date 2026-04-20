# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored flash-maxsim Triton kernels for ColBERT/ColPali MaxSim scoring.

Originally from https://github.com/roipony/flash-maxsim (Apache 2.0).
Forward-pass only; backward/training kernels are not included.
"""

from vllm.v1.pool.flash_maxsim.flash_maxsim import (
    flash_maxsim,
    flash_maxsim_batched,
)
from vllm.v1.pool.flash_maxsim.flash_maxsim_rerank import (
    flash_maxsim_rerank,
    flash_maxsim_rerank_direct,
)

__all__ = [
    "flash_maxsim",
    "flash_maxsim_batched",
    "flash_maxsim_rerank",
    "flash_maxsim_rerank_direct",
]
