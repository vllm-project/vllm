# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout plumbing for DeepSeek V4.1 mega attention.

The kernel's Q and O layouts are produced by permuting ``wq_b`` rows and
``wo_a`` columns in their weight loaders, so what has to hold is that the
permuted GEMMs agree with a reference that permutes the activation instead.
"""

from types import SimpleNamespace

import pytest
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
from vllm.platforms import current_platform

HEAD_DIM = 512


def _load(checkpoint: torch.Tensor, perm: torch.Tensor, dim: int) -> torch.Tensor:
    """Load ``checkpoint`` into a fresh parameter through a permuting loader."""
    param = torch.nn.Parameter(torch.empty_like(checkpoint), requires_grad=False)
    param.weight_loader = default_weight_loader
    permute_on_load(param, perm, dim)
    param.weight_loader(param, checkpoint)
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


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Mega attention requires SM100",
)
@pytest.mark.parametrize("tokens,groups", [(1, 8), (129, 1), (8193, 8)])
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

    def scale_rows(scale):
        # F8_128x4 words back to one row per token; padded rows are unwritten.
        words = scale.view(torch.int32).view(-1, groups * n // 128, 32, 4)
        return words.permute(0, 3, 2, 1).reshape(-1, groups * n // 128)[:tokens]

    def assert_same(actual, expected):
        torch.testing.assert_close(
            actual[0].view(torch.uint8), expected[0].view(torch.uint8)
        )
        torch.testing.assert_close(scale_rows(actual[1]), scale_rows(expected[1]))

    assert_same(projection(True), projection(False))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = projection(True)
    o.data.copy_((torch.randn(o.data.shape, device="cuda") * 0.3).to(o.data.dtype))
    graph.replay()
    assert_same(graph_output, projection(False))
