# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny BF16 projections for decode steps: the CuTe skinny GEMM that issues its
static weight loads before the PDL wait at 1-4 tokens, and Kimi K3's split-K
tcgen05 GEMM (cluster DSMEM reduction, evict-first weight TMA) for the DSA and
MTP draft projections at 4-64 tokens.

The main model keeps cuBLAS at 8+ tokens on the KDA projections: the skinny
GEMM falls behind it there, and the split-K tactics measured slower on the KDA
in_proj at B4/B8 (14.4 vs 10.7 us). The DSA projections take the split-K GEMM
at 4-64 tokens where their shape has a tactic (policy ``dsa_proj``).
"""

import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)

# (block size, outputs per block, K unroll) per token count, by weight (N, K),
# from a GB300 sweep with the weights rotated out of L2.
TACTICS = {
    (6416, 4096): {1: (64, 4, 2), 2: (64, 8, 2), 4: (64, 8, 2)},  # KDA in_proj
    (4096, 2048): {1: (64, 8, 2), 2: (128, 4, 2), 4: (64, 8, 2)},  # KDA o_proj
    (4096, 4096): {1: (64, 4, 2), 2: (128, 2, 1), 4: (64, 2, 2)},  # DSA o_proj
    (4096, 8192): {1: (128, 4, 2), 2: (128, 4, 2)},  # MTP eh_proj
    (38720, 4096): {1: (128, 2, 1), 2: (128, 2, 1)},  # MTP LM head shard
}
DEFAULT = {1: (64, 4, 2), 2: (64, 8, 2), 4: (64, 8, 2)}

# Split-K (N tile, token tile, K split, stages) by weight (N, K) and token tile,
# from the same sweep (us, cuBLAS -> split-K, at each tile's token count). Token
# counts a shape has no tile for keep cuBLAS.
SPLIT_K = {
    # MTP LM head shard: 8 51.6 -> 46.7, 16 51.9 -> 46.9, 64 53.8 -> 48.9
    (38720, 4096): {8: (128, 8, 1, 4), 16: (128, 16, 1, 4), 64: (128, 64, 1, 4)},
    # MTP eh_proj: 8 13.5 -> 11.3, 16 13.6 -> 11.6, 32 13.9 -> 11.7, 64 14.2 -> 13.2
    (4096, 8192): {
        8: (128, 8, 4, 6),
        16: (64, 16, 2, 10),
        32: (64, 32, 2, 8),
        64: (64, 64, 2, 6),
    },
    # DSA fused_qkv_a_proj: 8 6.5 -> 4.1, 16 8.3 -> 4.3, 32 7.5 -> 4.8, 64 7.4 -> 6.3
    (2048, 4096): {
        8: (64, 8, 4, 8),
        16: (64, 16, 4, 6),
        32: (64, 32, 2, 8),
        64: (64, 64, 2, 6),
    },
    # DSA q_b_proj and indexer wq_b: 8 5.2 -> 3.2, 16 4.0 -> 3.4, 32 4.0 -> 4.0
    (4096, 1536): {8: (64, 8, 2, 6), 16: (128, 16, 2, 6), 32: (64, 32, 2, 8)},
    # DSA o_proj: 8 9.0 -> 6.6, 16 10.9 -> 6.7, 32 8.9 -> 7.1
    (4096, 4096): {8: (64, 8, 2, 8), 16: (64, 16, 2, 10), 32: (64, 32, 2, 8)},
}


class Projection:
    """``x @ weight.T`` for 1, 2 or 4 tokens; ``split_k_tokens`` take the split-K
    GEMM instead where their token tile has a tactic."""

    def __init__(self, weight, split_k_tokens=()):
        self.weight = weight
        n, k = weight.shape
        self.configs = {}
        for m, (block, outputs, unroll) in TACTICS.get((n, k), DEFAULT).items():
            if k % (block * 8) or n % outputs:
                continue
            cfg = SkinnyGemmConfig(m, block, outputs, unroll, 8, k)
            shape_dynamic_skinny_gemm(weight.new_zeros(m, k), weight, cfg, None)
            self.configs[m] = cfg  # compiled before capture
        tactics = SPLIT_K.get((n, k), {})
        policies = {}
        for m in split_k_tokens:
            tile = next((t for t in sorted(tactics) if t >= m), None)
            if tile is not None:
                policies[m] = tactics[tile]
        self.split_k = None
        if policies:
            from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (  # noqa: E501
                PreparedProjection,
            )

            self.split_k = PreparedProjection(weight, policies).prepare()

    def supports(self, tokens: int) -> bool:
        return tokens in self.configs or (
            self.split_k is not None and tokens in self.split_k.kernels
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        m = x.shape[0]
        if self.split_k is not None and m in self.split_k.kernels:
            out = x.new_empty((m, self.weight.shape[0]))
            return self.split_k.run(x, out)
        return shape_dynamic_skinny_gemm(x, self.weight, self.configs[m], None)
