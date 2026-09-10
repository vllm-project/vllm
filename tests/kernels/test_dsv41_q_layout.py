# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F

from tests.kernels.attention.test_flashmla_fused_sparse import (
    make_cos_sin_cache,
    rope_gptj,
)
from vllm.models.deepseek_v4_1.common.ops.fused_layout import (
    permute_q_from_fused,
    permute_q_to_fused,
)
from vllm.models.deepseek_v4_1.common.ops.q_layout import dsv41_q_layout


@pytest.mark.parametrize("num_tokens", [1, 5, 300])
@pytest.mark.parametrize("local_heads", [16, 32, 64])
def test_fused_mode_pads_fused_layout(num_tokens, local_heads):
    device = torch.device("cuda")
    q_std = torch.randn(
        num_tokens, local_heads, 512, device=device, dtype=torch.bfloat16
    )
    out = dsv41_q_layout(permute_q_to_fused(q_std), 64, "fused")
    expected = permute_q_to_fused(F.pad(q_std, (0, 0, 0, 64 - local_heads)))
    assert out.shape == (num_tokens, 64, 512)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("num_tokens", [1, 5, 300])
def test_standard_rope_mode_matches_reference(num_tokens):
    device = torch.device("cuda")
    local_heads = 16
    cos_sin = make_cos_sin_cache(4096, device)
    positions = torch.randint(0, 4096, (num_tokens,), device=device)
    q_std = torch.randn(
        num_tokens, local_heads, 512, device=device, dtype=torch.bfloat16
    )
    out = dsv41_q_layout(
        permute_q_to_fused(q_std),
        64,
        "standard_rope",
        positions=positions,
        cos_sin_cache=cos_sin,
    )
    expected = rope_gptj(F.pad(q_std, (0, 0, 0, 64 - local_heads)), positions, cos_sin)
    torch.testing.assert_close(out.float(), expected.float(), rtol=1e-2, atol=1e-2)
    assert torch.equal(permute_q_from_fused(permute_q_to_fused(q_std)), q_std)


def test_standard_rope_mode_matches_cuda_q_path():
    """Same RoPE convention as the existing fused Q-norm/RoPE/KV-insert op."""
    device = torch.device("cuda")
    num_tokens, local_heads, block_size = 7, 16, 32
    cos_sin = make_cos_sin_cache(4096, device)
    positions = torch.randint(0, 4096, (num_tokens,), device=device)
    q_std = torch.randn(
        num_tokens, local_heads, 512, device=device, dtype=torch.bfloat16
    )
    kv = torch.randn(num_tokens, 512, device=device, dtype=torch.bfloat16)
    k_cache = torch.zeros(2, block_size * 584, dtype=torch.uint8, device=device)
    slot_mapping = torch.full((num_tokens,), -1, dtype=torch.int64, device=device)
    q_ref = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q_std,
        kv,
        k_cache,
        slot_mapping,
        positions,
        cos_sin,
        64,
        1e-20,
        block_size,
        False,
    )
    out = dsv41_q_layout(
        permute_q_to_fused(q_std),
        64,
        "standard_rope",
        positions=positions,
        cos_sin_cache=cos_sin,
    )
    torch.testing.assert_close(out.float(), q_ref.float(), rtol=1e-2, atol=1e-2)
