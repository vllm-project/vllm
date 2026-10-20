# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LUT-B weight x MXFP8 activation MoE (emulated and Rubin SM107)."""

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.fused_moe.experts.lut_b_moe import (
    LutBEmulationTritonExperts,
    LutBSm107Experts,
    make_lut_b_moe_kernel,
    mxfp8_qdq,
)
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    LUT_B_PACKED_TILE_BYTES,
    dequantize_lut_b,
    lut_b_to_sm107_tiles,
    pack_lut_b_indices,
    quantize_lut_b,
    unpack_lut_b_indices,
)
from vllm.platforms import current_platform

requires_gpu = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like GPU"
)
requires_sm107 = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.is_device_capability((10, 7))
        and ops.lut_b_moe_gemm_supported()
    ),
    reason="requires a Rubin (SM107) GPU and the compiled LUT-B MoE kernel",
)


def test_pack_roundtrip() -> None:
    indices = torch.randint(0, 8, (5, 64), dtype=torch.uint8)
    packed = pack_lut_b_indices(indices)
    assert packed.shape == (5, 24)
    assert torch.equal(unpack_lut_b_indices(packed), indices)


def test_quantize_format_and_fit() -> None:
    torch.manual_seed(0)
    # Different magnitudes per expert exercise the tile scales.
    weight = torch.randn(3, 16, 128) * torch.tensor([1.0, 1e-3, 1e2])[:, None, None]

    packed, codebooks, scales = quantize_lut_b(weight)
    assert packed.shape == (3, 2, 2, LUT_B_PACKED_TILE_BYTES)
    assert codebooks.shape == (3, 2, 2, 8)
    assert codebooks.dtype == torch.float8_e4m3fn
    assert scales.shape == (3, 2, 2) and scales.dtype == torch.uint8
    bits = (packed.numel() + codebooks.numel() + scales.numel()) * 8
    assert bits / weight.numel() == pytest.approx(3.14, abs=0.01)

    again = quantize_lut_b(weight)
    assert all(torch.equal(x, y) for x, y in zip((packed, codebooks, scales), again))

    reconstructed = dequantize_lut_b(packed, codebooks, scales, torch.float32)
    error = (reconstructed - weight).square().sum((1, 2))
    # 8-level Lloyd fits of Gaussian data keep ~4% of the energy as error,
    # independent of the tile magnitude.
    assert torch.all(error / weight.square().sum((1, 2)) < 0.06)


def test_quantize_preserves_zeros() -> None:
    weight = torch.linspace(-1, 1, 512).reshape(8, 64)
    weight[:, -1] = 0
    reconstructed = dequantize_lut_b(*quantize_lut_b(weight), torch.float32)
    assert torch.all(reconstructed[weight == 0] == 0)


def _decode_sm107_tiles(tiles: torch.Tensor) -> torch.Tensor:
    """Independent per-element decoder for the SM107 tile layout."""
    num_experts, n_tiles, k_tiles, _ = tiles.shape
    e4m3 = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    e4m3 = e4m3.view(torch.float8_e4m3fn).float().tolist()
    out = torch.empty(num_experts, n_tiles * 128, k_tiles * 128)
    for e in range(num_experts):
        for nt in range(n_tiles):
            for kt in range(k_tiles):
                tile = tiles[e, nt, kt].tolist()
                for r in range(128):
                    row = [
                        tile[(r // 8) * 384 + (j // 16) * 128 + (r % 8) * 16 + j % 16]
                        for j in range(48)
                    ]
                    bits = int.from_bytes(bytes(row), "little")
                    for k in range(128):
                        index = (bits >> (3 * k)) & 7
                        book = tile[6656 + (r // 8) * 16 + (k // 64) * 8 + index]
                        scale = tile[6144 + (r % 32) * 16 + (r // 32) * 4 + k // 32]
                        value = e4m3[book] * 2.0 ** (scale - 127)
                        out[e, nt * 128 + r, kt * 128 + k] = value
    return out


def test_sm107_tile_layout() -> None:
    torch.manual_seed(0)
    weight = torch.randn(1, 128, 256) * torch.logspace(-3, 0, 128)[:, None]
    packed, codebooks, scales = quantize_lut_b(weight)
    tiles = lut_b_to_sm107_tiles(packed, codebooks, scales)
    assert tiles.shape == (1, 1, 2, 7168)
    expected = dequantize_lut_b(packed, codebooks, scales, torch.float32)
    torch.testing.assert_close(_decode_sm107_tiles(tiles), expected, rtol=0, atol=0)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # Accumulation order changes the BF16 intermediate slightly, which flips
    # the MXFP8 rounding of a few elements before the second GEMM, so compare
    # error norms. Skipping that MXFP8 rounding entirely gives about 3%.
    actual, expected = actual.float(), expected.float()
    assert torch.isfinite(actual).all()
    error = (actual - expected).norm() / expected.norm()
    assert error < 2e-2, f"relative error {error:.3g}"


def _make_problem(
    num_tokens: int, num_experts: int, top_k: int, hidden: int, inter: int
):
    hidden_states = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w13 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    scores = torch.randn(num_tokens, num_experts, device="cuda")
    topk_weights, topk_ids = scores.softmax(-1).topk(top_k, dim=-1)
    return hidden_states, w13, w2, topk_weights, topk_ids.to(torch.int32)


def _run(experts_cls, w1, w2, hidden_states, topk_weights, topk_ids, num_experts):
    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=topk_ids.size(1),
        hidden_dim=hidden_states.size(1),
        intermediate_size=w2.size(-1) if w2.ndim == 3 else w2.size(2) * 128,
        in_dtype=hidden_states.dtype,
    )
    kernel = make_lut_b_moe_kernel(
        experts_cls, moe_config, FUSED_MOE_UNQUANTIZED_CONFIG, None
    )
    return kernel.apply(
        hidden_states,
        w1,
        w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )


@requires_gpu
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
def test_emulation_matches_reference(workspace_init, num_tokens: int) -> None:
    torch.manual_seed(1)
    num_experts, top_k, hidden, inter = 4, 2, 128, 128
    x, w13, w2, topk_weights, topk_ids = _make_problem(
        num_tokens, num_experts, top_k, hidden, inter
    )
    w13 = dequantize_lut_b(*quantize_lut_b(w13), torch.bfloat16)
    w2 = dequantize_lut_b(*quantize_lut_b(w2), torch.bfloat16)

    actual = _run(
        LutBEmulationTritonExperts, w13, w2, x, topk_weights, topk_ids, num_experts
    )

    expected = torch.zeros_like(x, dtype=torch.float32)
    x_q = mxfp8_qdq(x).float()
    for t in range(num_tokens):
        for j in range(top_k):
            e = topk_ids[t, j]
            gate, up = F.linear(x_q[t], w13[e].float()).bfloat16().chunk(2)
            h = mxfp8_qdq((F.silu(gate.float()) * up.float()).bfloat16()[None])[0]
            expected[t] += topk_weights[t, j] * F.linear(h.float(), w2[e].float())
    _assert_close(actual, expected)


@requires_sm107
@pytest.mark.parametrize("num_tokens", [1, 33, 300])
@pytest.mark.parametrize("hidden,inter", [(256, 128), (512, 384)])
def test_sm107_matches_emulation(
    workspace_init, num_tokens: int, hidden: int, inter: int
) -> None:
    torch.manual_seed(2)
    num_experts, top_k = 8, 4
    x, w13, w2, topk_weights, topk_ids = _make_problem(
        num_tokens, num_experts, top_k, hidden, inter
    )
    q13, q2 = quantize_lut_b(w13), quantize_lut_b(w2)

    native = _run(
        LutBSm107Experts,
        lut_b_to_sm107_tiles(*q13),
        lut_b_to_sm107_tiles(*q2),
        x,
        topk_weights,
        topk_ids,
        num_experts,
    )
    emulated = _run(
        LutBEmulationTritonExperts,
        dequantize_lut_b(*q13, torch.bfloat16),
        dequantize_lut_b(*q2, torch.bfloat16),
        x,
        topk_weights,
        topk_ids,
        num_experts,
    )
    _assert_close(native, emulated)


@requires_sm107
def test_sm107_gemm_exact(workspace_init) -> None:
    """One grouped GEMM against an FP32 reference of the decoded operands."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
        mxfp8_e4m3_quantize,
    )

    torch.manual_seed(3)
    num_tokens, num_experts, top_k, n, k = 200, 3, 2, 256, 384
    a = torch.randn(num_tokens, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(num_experts, n, k, device="cuda")
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")
    topk_weights = torch.rand(num_tokens, top_k, device="cuda")
    q = quantize_lut_b(w)

    a_q, a_s = mxfp8_e4m3_quantize(a)
    sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
        topk_ids.to(torch.int32), 128, num_experts
    )
    out = torch.full((num_tokens * top_k, n), float("nan"), device="cuda").bfloat16()
    ops.lut_b_moe_gemm(
        a_q,
        a_s.view(num_tokens, -1),
        lut_b_to_sm107_tiles(*q),
        out,
        topk_weights.flatten(),
        sorted_ids,
        expert_ids,
        num_post_pad,
        top_k,
    )

    a_ref = dequant_mxfp8_to_bf16(a_q, a_s.view(num_tokens, -1)).float()
    w_ref = dequantize_lut_b(*q, torch.float32)
    ref = (
        torch.einsum("tk,tjnk->tjn", a_ref.double(), w_ref.double()[topk_ids])
        * topk_weights[..., None].double()
    )
    torch.testing.assert_close(
        out.view(num_tokens, top_k, n).double(), ref, rtol=1e-2, atol=1e-2
    )
