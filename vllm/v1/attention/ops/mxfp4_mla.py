# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row layout of the mxfp4_mla KV cache.

OCP MXFP4, group 32: E2M1 elements packed two per byte (the low nibble holds
the lower index) and one E8M0 scale per group. A row is the packed data
followed by its scales, ``row_bytes(d) == d // 2 + d // 32``.
"""

from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_BLOCK_SIZE,
)
from vllm.triton_utils import tl

GROUP_SIZE = OCP_MX_BLOCK_SIZE
# A tl.constexpr so the store and read kernels can use it directly.
E8M0_BIAS = tl.constexpr(127)


def row_bytes(latent_dim: int, group: int = GROUP_SIZE) -> int:
    """Packed row size: data bytes + scale bytes. 272 for a 512-wide latent."""
    if latent_dim % group:
        raise ValueError(f"latent dim {latent_dim} not divisible by group {group}")
    return latent_dim // 2 + latent_dim // group
