# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout plumbing for DeepSeek V4.1 mega attention.

The kernel's Q and O layouts are produced by permuting ``wq_b`` rows and
``wo_a`` columns once at load, so what has to hold is that the permuted GEMMs
agree with a reference that permutes the activation instead.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.common.ops.fused_layout import (
    WV_GROUP_SIZE,
    o_fused_chunk_permutation,
    o_fused_permutation,
    permute_q_to_fused,
    permute_wo_a_,
    permute_wq_b_,
    q_fused_permutation,
)
from vllm.models.deepseek_v41.common.ops.q_layout import pad_fused_q_heads
from vllm.platforms import current_platform

HEAD_DIM = 512


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
    orig_weight, orig_scale = weight.clone(), scale.clone()

    standard = (qr @ weight.T).view(num_tokens, num_heads, HEAD_DIM)
    expected = permute_q_to_fused(standard)

    permute_wq_b_(weight, scale, num_heads)
    fused = (qr @ weight.T).view(num_tokens, num_heads, HEAD_DIM)
    torch.testing.assert_close(fused, expected)
    # The scale has to follow its row, or dequant reads another row's scale.
    perm = q_fused_permutation(num_heads, HEAD_DIM)
    torch.testing.assert_close(weight, orig_weight[perm])
    torch.testing.assert_close(scale, orig_scale[perm])


def test_permuted_wo_a_consumes_fused_output():
    """Permuting wo_a's input columns lets it read the kernel's O layout."""
    torch.manual_seed(1)
    out_features, num_tokens = 32, 4
    in_features = WV_GROUP_SIZE * HEAD_DIM
    weight = torch.randn(out_features, in_features)
    scale = torch.randint(0, 256, (out_features, in_features // 32), dtype=torch.uint8)
    standard_o = torch.randn(num_tokens, in_features)
    orig_weight, orig_scale = weight.clone(), scale.clone()

    expected = standard_o @ weight.T
    # The kernel emits the same values in the fused chunk order.
    perm = o_fused_permutation(WV_GROUP_SIZE, HEAD_DIM)
    fused_o = standard_o[:, perm]

    permute_wo_a_(weight, scale, WV_GROUP_SIZE)
    torch.testing.assert_close(weight, orig_weight[:, perm])
    torch.testing.assert_close(
        scale, orig_scale[:, o_fused_chunk_permutation(WV_GROUP_SIZE, HEAD_DIM)]
    )
    # Unlike wq_b, this permutation sits inside the 4096-term reduction, so the
    # summation order changes and the result differs in the last few ulps.
    torch.testing.assert_close(fused_o @ weight.T, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("local_heads,padded_heads", [(16, 64), (64, 64), (64, 128)])
def test_pad_fused_q_heads(local_heads: int, padded_heads: int):
    """Padding keeps every live chunk and zeroes the padding heads."""
    torch.manual_seed(2)
    num_tokens = 3
    q = torch.randn(
        num_tokens, local_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    out = pad_fused_q_heads(q, padded_heads)
    assert out.shape == (num_tokens, padded_heads, HEAD_DIM)
    if padded_heads == local_heads:
        assert out.data_ptr() == q.data_ptr()
        return
    # Each of the 32 head-dim chunks holds [padded_heads, 16]: the first
    # local_heads rows are the input's chunk, the rest are zero.
    got = out.reshape(num_tokens, 32, padded_heads, 16)
    want = q.reshape(num_tokens, 32, local_heads, 16)
    torch.testing.assert_close(got[:, :, :local_heads], want)
    assert (got[:, :, local_heads:] == 0).all()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("local_heads", [16, 64])
@pytest.mark.parametrize("capture", [False, True])
def test_query_preparation_replays_padded_heads(local_heads, capture):
    """Attention input preparation publishes padded Q before the eager break."""
    from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import (
        DeepseekV4MegaAttnAttention,
    )

    layer = SimpleNamespace(padded_heads=64, _insert_swa_kv=lambda *args: None)
    q = torch.randn(7, local_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    kv = torch.empty(7, HEAD_DIM, dtype=q.dtype, device=q.device)
    positions = torch.arange(7, device=q.device)

    def prepare():
        return DeepseekV4MegaAttnAttention._prepare_q_and_insert_kv(
            layer, q, kv, positions, None
        )

    out = prepare()
    graph = None
    if capture and local_heads != 64:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = prepare()
    for factor in (0.25, -0.75):
        q.copy_(torch.randn_like(q) * factor)
        if graph is None:
            out = prepare()
        else:
            graph.replay()
        assert out.shape == (7, 64, HEAD_DIM)
        chunks = out.reshape(7, 32, 64, 16)
        torch.testing.assert_close(
            chunks[:, :, :local_heads],
            q.reshape(7, 32, local_heads, 16),
            rtol=0,
            atol=0,
        )
        assert (chunks[:, :, local_heads:] == 0).all()
