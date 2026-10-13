# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout plumbing for DeepSeek V4.1 mega attention.

The kernel's Q and O layouts are produced by permuting ``wq_b`` rows and
``wo_a`` columns in their weight loaders, so what has to hold is that the
permuted GEMMs agree with a reference that permutes the activation instead.
"""

import torch

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.deepseek_v41.common.ops.fused_layout import (
    WV_GROUP_SIZE,
    o_fused_chunk_permutation,
    o_fused_permutation,
    permute_on_load,
    permute_q_to_fused,
    q_fused_permutation,
)

HEAD_DIM = 512


def _load(checkpoint: torch.Tensor, perm: torch.Tensor, dim: int) -> torch.Tensor:
    """Load ``checkpoint`` into a fresh parameter through a permuting loader."""
    param = torch.nn.Parameter(torch.empty_like(checkpoint), requires_grad=False)
    param.weight_loader = default_weight_loader  # type: ignore[attr-defined]
    permute_on_load(param, perm, dim)
    param.weight_loader(param, checkpoint)  # type: ignore[attr-defined]
    return param.data


def test_permuted_wq_b_gemm_matches_permuted_activation():
    """Permuting wq_b's rows makes the Q GEMM emit the fused layout directly."""
    torch.manual_seed(0)
    num_heads, q_lora_rank, num_tokens = 16, 64, 5
    weight = torch.randn(num_heads * HEAD_DIM, q_lora_rank)
    # One scale row per weight row, as an MXFP8 wq_b shard carries.
    scale = torch.randint(
        0, 256, (num_heads * HEAD_DIM, q_lora_rank // 32), dtype=torch.uint8
    )
    qr = torch.randn(num_tokens, q_lora_rank)

    standard = (qr @ weight.T).view(num_tokens, num_heads, HEAD_DIM)
    expected = permute_q_to_fused(standard)

    perm = q_fused_permutation(num_heads, HEAD_DIM)
    loaded_weight = _load(weight, perm, dim=0)
    fused = (qr @ loaded_weight.T).view(num_tokens, num_heads, HEAD_DIM)
    torch.testing.assert_close(fused, expected)
    # The scale has to follow its row, or dequant reads another row's scale.
    torch.testing.assert_close(loaded_weight, weight[perm])
    torch.testing.assert_close(_load(scale, perm, dim=0), scale[perm])


def test_permuted_wo_a_consumes_fused_output():
    """Permuting wo_a's input columns lets it read the kernel's O layout."""
    torch.manual_seed(1)
    out_features, num_tokens = 32, 4
    in_features = WV_GROUP_SIZE * HEAD_DIM
    weight = torch.randn(out_features, in_features)
    scale = torch.randint(0, 256, (out_features, in_features // 32), dtype=torch.uint8)
    standard_o = torch.randn(num_tokens, in_features)

    expected = standard_o @ weight.T
    # The kernel emits the same values in the fused chunk order.
    perm = o_fused_permutation(WV_GROUP_SIZE, HEAD_DIM)
    fused_o = standard_o[:, perm]

    loaded_weight = _load(weight, perm, dim=1)
    torch.testing.assert_close(loaded_weight, weight[:, perm])
    chunk_perm = o_fused_chunk_permutation(WV_GROUP_SIZE, HEAD_DIM)
    torch.testing.assert_close(_load(scale, chunk_perm, dim=1), scale[:, chunk_perm])
    # Unlike wq_b, this permutation sits inside the 4096-term reduction, so the
    # summation order changes and the result differs in the last few ulps.
    torch.testing.assert_close(
        fused_o @ loaded_weight.T, expected, rtol=1e-3, atol=1e-3
    )
