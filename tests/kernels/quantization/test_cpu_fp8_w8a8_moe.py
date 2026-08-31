# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU FP8 W8A8 fused MoE kernel.

Tests the full MoE computation pipeline:
  - weight packing (float8_linear_prepack_cpu for w13, convert_weight_packed for w2)
  - activation quantization (quantize_fp8e4m3_vec)
  - fused_experts_cpu with CPUQuantMethod.FP8_W8A8

Scale conventions (DeepSeek-V3 style block quantization):
  - w13_scale: [E, 2N, G] where G = K//QUANT_GROUP (per-K-group scales)
  - w2_scale:  [E, K//QUANT_GROUP, N//QUANT_GROUP] (2-D block scales)
  - block_size: [QUANT_GROUP, QUANT_GROUP] = [128, 128]

Run:
  pytest tests/kernels/quantization/test_cpu_fp8_w8a8_moe.py -v
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm._custom_ops import CPUQuantMethod, fused_experts_cpu
from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

if not ops._supports_cpu_fp8_w8a8:
    pytest.skip(
        "float8_linear_prepack_cpu op not available", allow_module_level=True
    )

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
BLOCK_N = 32    # block_size_n() in gemm.h (tiling block, fixed)
BLOCK_K = 128   # BLOCK_K macro in gemm.h (tiling block, fixed)
QUANT_GROUP = 128   # quantization group size for both K and N dimensions


# ---------------------------------------------------------------------------
# Helpers: weight creation with block quantization
# ---------------------------------------------------------------------------

def make_fp8_weight_w13(
    E: int, two_n: int, K: int, group_K: int = QUANT_GROUP
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create w13 [E, 2N, K] FP8 with group scales [E, 2N, G] where G = K//group_K."""
    G = K // group_K
    w_list, s_list = [], []
    for _ in range(E):
        w_f32 = torch.randn(two_n, K)
        # Per-row, per-K-group quantization: scale [2N, G]
        w_re = w_f32.view(two_n, G, group_K)
        abs_max = w_re.abs().amax(dim=2, keepdim=True).clamp(min=1e-7)  # [2N, G, 1]
        scale = (abs_max / FP8_MAX).squeeze(2)  # [2N, G]
        w_q = (w_re / abs_max).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        w_list.append(w_q.view(two_n, K).contiguous())
        s_list.append(scale.float())
    return torch.stack(w_list), torch.stack(s_list)  # [E, 2N, K], [E, 2N, G]


def make_fp8_weight_w2(
    E: int, K: int, N: int, group_K: int = QUANT_GROUP, group_N: int = QUANT_GROUP
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create w2 [E, K, N] FP8 with block scales [E, K//gK, N//gN]."""
    nK = K // group_K
    nN = N // group_N
    w_list, s_list = [], []
    for _ in range(E):
        w_f32 = torch.randn(K, N)
        w_re = w_f32.view(nK, group_K, nN, group_N)
        abs_max = w_re.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-7)  # [nK, 1, nN, 1]
        scale = (abs_max / FP8_MAX).squeeze(1).squeeze(2)  # [nK, nN]
        w_q = (w_re / abs_max).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        w_list.append(w_q.view(K, N).contiguous())
        s_list.append(scale.float())
    return torch.stack(w_list), torch.stack(s_list)  # [E, K, N], [E, K//gK, N//gN]


def pack_w13_for_cpu(
    w13: torch.Tensor,        # [E, 2N, K] FP8
    w13_scale: torch.Tensor,  # [E, 2N, G] float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack w13 for CPU FP8 W8A8 MoE kernel using float8_linear_prepack_cpu.

    float8_linear_prepack_cpu([2N, K], [2N, G]) → packed_w [Nc, Kc, BLOCK_K, BLOCK_N],
                                                   packed_s [Nc, G, BLOCK_N]
    After stacking E experts: packed_w [E, Nc, Kc, BLOCK_K, BLOCK_N],
                               packed_s [E, Nc, G, BLOCK_N]
    """
    E = w13.size(0)
    packed_list, scale_list = [], []
    for i in range(E):
        pw, ps = torch.ops._C.float8_linear_prepack_cpu(
            w13[i].contiguous(), w13_scale[i].contiguous()
        )
        packed_list.append(pw)
        scale_list.append(ps)
    return torch.stack(packed_list), torch.stack(scale_list)


def reference_moe_fp8_w8a8(
    hidden_states: torch.Tensor,  # BF16 [M, K_hidden] (after FP8 round-trip dequant)
    w13: torch.Tensor,             # FP8 [E, 2N, K_hidden]
    w13_scale: torch.Tensor,       # float32 [E, 2N, G] (G = K_hidden//group_K)
    w2: torch.Tensor,              # FP8 [E, K_hidden, N]
    w2_scale: torch.Tensor,        # float32 [E, K_hidden//gK, N//gN]
    topk_weights: torch.Tensor,    # float32 [M, top_k]
    topk_ids: torch.Tensor,        # int64 [M, top_k]
    N: int,
    group_K: int = QUANT_GROUP,
    group_N: int = QUANT_GROUP,
) -> torch.Tensor:
    """Reference implementation: dequantize weights then compute MoE."""
    M, K_hidden = hidden_states.shape
    top_k = topk_ids.shape[1]
    output = torch.zeros(M, K_hidden, dtype=torch.bfloat16)

    for tok in range(M):
        x = hidden_states[tok].float()  # [K_hidden]
        for k_idx in range(top_k):
            expert_id = topk_ids[tok, k_idx].item()
            w = topk_weights[tok, k_idx].item()

            # Stage 1: x @ w13.T — gate and up projections
            # Dequantize w13 per K-group
            w13_e = w13[expert_id].float()      # [2N, K_hidden]
            ws_e = w13_scale[expert_id].float() # [2N, G]
            G = ws_e.shape[1]
            w13_dq = torch.zeros_like(w13_e)
            for g in range(G):
                c0, c1 = g * group_K, (g + 1) * group_K
                w13_dq[:, c0:c1] = w13_e[:, c0:c1] * ws_e[:, g:g+1]
            gate_up = (x.unsqueeze(0) @ w13_dq.T).squeeze(0)  # [2N]

            # SiLU×gate (gate is first N, up is second N)
            gate = gate_up[:N]
            up   = gate_up[N:]
            act  = torch.nn.functional.silu(gate) * up  # [N]

            # Stage 2: act @ w2.T — down projection
            # Dequantize w2 per block
            w2_e = w2[expert_id].float()       # [K_hidden, N]
            ws2_e = w2_scale[expert_id].float() # [K_hidden//gK, N//gN]
            nK2 = K_hidden // group_K
            nN2 = N // group_N
            w2_dq = torch.zeros_like(w2_e)
            for gi in range(nK2):
                for gj in range(nN2):
                    r0, r1 = gi * group_K, (gi + 1) * group_K
                    c0, c1 = gj * group_N, (gj + 1) * group_N
                    w2_dq[r0:r1, c0:c1] = w2_e[r0:r1, c0:c1] * ws2_e[gi, gj]
            out_e = (act.unsqueeze(0) @ w2_dq.T).squeeze(0)  # [K_hidden]

            output[tok] += (w * out_e).to(torch.bfloat16)

    return output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("E,N,K,M,top_k", [
    # N and K must be multiples of QUANT_GROUP=128 for block quantization
    (4, 128, 256, 8, 2),
    (8, 256, 512, 16, 2),
    (4, 256, 512, 4, 1),
])
def test_fused_experts_fp8_w8a8_shape(E: int, N: int, K: int, M: int, top_k: int):
    """Test that fused_experts_cpu FP8_W8A8 produces the correct output shape."""
    assert N % QUANT_GROUP == 0, f"N={N} must be multiple of QUANT_GROUP={QUANT_GROUP}"
    assert K % QUANT_GROUP == 0, f"K={K} must be multiple of QUANT_GROUP={QUANT_GROUP}"

    # w13: [E, 2N, K] FP8, w13_scale: [E, 2N, G] with G = K//QUANT_GROUP
    w13, w13_scale = make_fp8_weight_w13(E, 2 * N, K)
    # w2: [E, K, N] FP8, w2_scale: [E, K//QUANT_GROUP, N//QUANT_GROUP]
    w2, w2_scale = make_fp8_weight_w2(E, K, N)

    # Pack weights
    packed_w13, packed_w13_scale = pack_w13_for_cpu(w13, w13_scale)
    packed_w2 = torch.ops._C.convert_weight_packed(w2)

    # Hidden states (BF16) and FP8 quantization
    hidden_states = torch.randn(M, K, dtype=torch.bfloat16)
    x_fp8, x_scales = torch.ops._C.quantize_fp8e4m3_vec(hidden_states, True, None)
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert x_scales.shape == (M,)

    # Routing
    topk_weights = torch.softmax(torch.randn(M, top_k), dim=-1).float()
    topk_ids = torch.zeros(M, top_k, dtype=torch.int32)
    for i in range(M):
        topk_ids[i] = torch.randperm(E)[:top_k].int()

    output = fused_experts_cpu(
        x_fp8,
        packed_w13,
        packed_w2,
        topk_weights,
        topk_ids,
        False,                       # inplace
        CPUQuantMethod.FP8_W8A8,
        packed_w13_scale,            # w1_scale: [E, Nc, G, BLOCK_N]
        w2_scale,                    # w2_scale: [E, K//128, N//128]
        None,                        # w1_zero
        None,                        # w2_zero
        [QUANT_GROUP, QUANT_GROUP],  # block_size: [128, 128]
        None,                        # w1_bias
        None,                        # w2_bias
        None,                        # alpha
        None,                        # limit
        True,                        # is_vnni (weights already packed)
        x_scales,                    # a1_scale: per-token FP8 activation scales
    )

    assert output.shape == (M, K), f"Expected ({M}, {K}), got {output.shape}"
    assert output.dtype == torch.bfloat16, f"Expected bfloat16, got {output.dtype}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"


@pytest.mark.parametrize("E,N,K,M,top_k", [
    (4, 128, 256, 8, 2),
    (4, 256, 512, 4, 1),
])
def test_fused_experts_fp8_w8a8_accuracy(E: int, N: int, K: int, M: int, top_k: int):
    """Test FP8 W8A8 MoE output accuracy vs BF16 dequantized reference."""
    assert N % QUANT_GROUP == 0
    assert K % QUANT_GROUP == 0

    torch.manual_seed(42)

    w13, w13_scale = make_fp8_weight_w13(E, 2 * N, K)
    w2, w2_scale = make_fp8_weight_w2(E, K, N)

    packed_w13, packed_w13_scale = pack_w13_for_cpu(w13, w13_scale)
    packed_w2 = torch.ops._C.convert_weight_packed(w2)

    hidden_states = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
    x_fp8, x_scales = torch.ops._C.quantize_fp8e4m3_vec(hidden_states, True, None)

    gen = torch.Generator().manual_seed(0)
    topk_weights = torch.softmax(torch.randn(M, top_k, generator=gen), dim=-1).float()
    topk_ids = torch.zeros(M, top_k, dtype=torch.int32)
    for i in range(M):
        topk_ids[i] = torch.randperm(E, generator=torch.Generator().manual_seed(i))[:top_k].int()

    output = fused_experts_cpu(
        x_fp8,
        packed_w13,
        packed_w2,
        topk_weights,
        topk_ids,
        False,
        CPUQuantMethod.FP8_W8A8,
        packed_w13_scale,
        w2_scale,
        None, None,
        [QUANT_GROUP, QUANT_GROUP],
        None, None, None, None,
        True,
        x_scales,
    )

    # Build reference: use FP8 round-tripped activations
    x_dequant = (x_fp8.float() * x_scales.unsqueeze(1)).bfloat16()
    ref = reference_moe_fp8_w8a8(
        x_dequant,
        w13, w13_scale,
        w2, w2_scale,
        topk_weights,
        topk_ids.long(),
        N,
    )

    # FP8 quantization error is ~1%; use generous tolerance
    atol, rtol = 1.0, 1.0
    close = torch.allclose(output.float(), ref.float(), atol=atol, rtol=rtol)

    if not close:
        ae = (output.float() - ref.float()).abs()
        re = ae / (ref.float().abs() + 1e-6)
        print(f"\nMax abs err={ae.max():.4f} mean={ae.mean():.4f} "
              f"Max rel err={re.max():.4f} mean={re.mean():.4f}")

    assert close, (
        f"FP8 W8A8 MoE output too far from reference (atol={atol}, rtol={rtol})"
    )


def test_fused_experts_fp8_w8a8_single_token():
    """Test with M=1 (single token, decode phase)."""
    E, N, K, top_k = 4, 128, 256, 2
    M = 1

    torch.manual_seed(7)

    w13, w13_scale = make_fp8_weight_w13(E, 2 * N, K)
    w2, w2_scale = make_fp8_weight_w2(E, K, N)

    packed_w13, packed_w13_scale = pack_w13_for_cpu(w13, w13_scale)
    packed_w2 = torch.ops._C.convert_weight_packed(w2)

    hidden_states = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
    x_fp8, x_scales = torch.ops._C.quantize_fp8e4m3_vec(hidden_states, True, None)

    topk_weights = torch.softmax(torch.ones(M, top_k), dim=-1).float()
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)

    output = fused_experts_cpu(
        x_fp8,
        packed_w13,
        packed_w2,
        topk_weights,
        topk_ids,
        False,
        CPUQuantMethod.FP8_W8A8,
        packed_w13_scale,
        w2_scale,
        None, None,
        [QUANT_GROUP, QUANT_GROUP],
        None, None, None, None,
        True,
        x_scales,
    )

    assert output.shape == (M, K)
    assert output.dtype == torch.bfloat16
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
