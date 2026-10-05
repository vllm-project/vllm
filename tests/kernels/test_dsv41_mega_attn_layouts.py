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


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("tokens,hidden", [(0, 5120), (129, 8192), (4097, 5120)])
def test_packed_scale_swizzle_matches_flashinfer_layout(tokens, hidden):
    """DeepGEMM's packed scales, padded stride included, become F8_128x4."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        swizzle_mxfp8_scale,
    )
    from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import (
        swizzle_packed_mxfp8_scale,
    )
    from vllm.utils.deep_gemm import get_tma_aligned_size

    raw = torch.randint(0, 255, (tokens, hidden // 32), device="cuda").to(torch.uint8)
    aligned = get_tma_aligned_size(tokens, 4) + 4
    packed = torch.empty((hidden // 128, aligned), device="cuda", dtype=torch.int32)
    packed = packed.t()[:tokens]
    packed.copy_(raw.view(torch.int32))
    torch.testing.assert_close(
        swizzle_packed_mxfp8_scale(packed), swizzle_mxfp8_scale(raw, tokens, hidden)
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Mega attention requires SM100",
)
@pytest.mark.parametrize("tokens,groups", [(1024, 1), (8193, 8)])
@torch.inference_mode()
def test_quantized_o_proj_matches_quantized_bf16(tokens, groups):
    """wo_a's MXFP8 output equals wo_b quantizing the BF16 z, also on replay."""
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import (
        DeepseekV4MegaAttnAttention,
        alloc_mega_attn_output,
    )

    def ue8m0_near_one(*shape):
        return torch.randint(121, 128, shape, device="cuda").to(torch.uint8)

    torch.manual_seed(7)
    k, n = 4096, 1024
    o = alloc_mega_attn_output(tokens, groups, torch.device("cuda"))
    o.data.copy_(torch.randn(o.data.shape, device="cuda").to(o.data.dtype))
    o.scale.copy_(ue8m0_near_one(tokens, groups, k // 32).view(torch.int32))
    weight_scale = torch.empty((groups, k // 128, n), device="cuda", dtype=torch.int32)
    weight_scale = weight_scale.transpose(1, 2)
    weight_scale.copy_(ue8m0_near_one(groups, n, k // 32).view(torch.int32))

    def wo_b(z):
        if isinstance(z, QuantizedActivation):
            return z.data, z.scale
        return mxfp8_e4m3_quantize(z, is_sf_swizzled_layout=True)

    layer = SimpleNamespace(
        n_local_groups=groups,
        o_lora_rank=n,
        wo_a=SimpleNamespace(
            weight=torch.randn(groups, n, k, device="cuda").to(torch.float8_e4m3fn),
            weight_scale=weight_scale,
        ),
        _einsum_recipe=(1, 1, 32),
        _wo_b_proj=wo_b,
    )

    def projection(quantize):
        layer._can_quantize_o_proj = quantize
        return DeepseekV4MegaAttnAttention._o_proj(layer, o, None)

    def assert_same(actual, expected):
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a.view(torch.uint8), e.view(torch.uint8))

    assert_same(projection(True), projection(False))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = projection(True)
    o.data.copy_((torch.randn(o.data.shape, device="cuda") * 0.3).to(o.data.dtype))
    graph.replay()
    assert_same(graph_output, projection(False))
