# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import random

import pytest
import torch

from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    persistent_masked_m_silu_mul_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT, has_deep_gemm
from vllm.utils.math_utils import cdiv, round_up

fp8_dtype = torch.float8_e4m3fn

CASES = [
    (1, 1, 128, fp8_dtype),
    (1, 4, 128 * 1, fp8_dtype),
    (2, 4, 128 * 2, fp8_dtype),
    (1, 4, 128 * 3, fp8_dtype),
    (8, 16, 128 * 4, fp8_dtype),
    (8, 16, 128 * 5, fp8_dtype),
    (8, 16, 128 * 6, fp8_dtype),
    (8, 16, 128 * 7, fp8_dtype),
    (8, 16, 128 * 8, fp8_dtype),
    (8, 16, 128 * 9, fp8_dtype),
    (8, 64, 7168, fp8_dtype),
    (8, 128, 128 * 33, fp8_dtype),
    (1, 4, 128 * 10, fp8_dtype),
    (8, 128, 7168, fp8_dtype),
    (8, 512, 7168, fp8_dtype),
    (8, 1024, 7168, fp8_dtype),
    (17, 31, 768, fp8_dtype),
    (32, 64, 256, fp8_dtype),
    (256, 8, 7168, fp8_dtype),
    (256, 32, 7168, fp8_dtype),
    (256, 64, 7168, fp8_dtype),
    # Only add a few fnuz tests to help with long CI times.
    (8, 512, 7168, torch.float8_e4m3fnuz),
    (8, 1024, 7168, torch.float8_e4m3fnuz),
]


def as_uint8(x) -> torch.Tensor:
    return (
        torch.empty(x.shape, dtype=x.dtype, device=x.device).copy_(x).view(torch.uint8)
    )


def silu(x: torch.Tensor) -> torch.Tensor:
    one_f32 = torch.tensor([1.0], device=x.device, dtype=torch.float32)
    x_f32 = x.to(torch.float32)
    act_f32 = x_f32 / (one_f32 + torch.exp(-x_f32))
    assert act_f32.dtype == torch.float32
    return act_f32.to(torch.bfloat16)


def do_quant(x: torch.Tensor, group_size: int, ceil_ue8m0: bool):
    eps_bf16 = torch.tensor([1e-10], device=x.device, dtype=torch.bfloat16)
    one_bf16 = torch.tensor([1.0], device=x.device, dtype=torch.bfloat16)
    fp8_max_bf16 = torch.tensor(
        [torch.finfo(fp8_dtype).max], device=x.device, dtype=torch.bfloat16
    )
    fp8_min_bf16 = torch.tensor(
        [torch.finfo(fp8_dtype).min], device=x.device, dtype=torch.bfloat16
    )
    fp8_max_inv = one_bf16 / fp8_max_bf16
    assert fp8_max_inv.dtype == torch.bfloat16

    assert x.size(-1) % group_size == 0
    num_groups = x.numel() // group_size
    x_og_shape = x.shape

    x = x.to(torch.bfloat16)
    x = x.view((-1, group_size))
    amax = x.abs().amax(dim=1).clamp(min=eps_bf16)
    assert amax.dtype == torch.bfloat16
    s = amax * fp8_max_inv

    if ceil_ue8m0:
        s = torch.exp2(
            torch.ceil(torch.log2(s).to(torch.bfloat16)).to(torch.bfloat16)
        ).to(torch.bfloat16)

    inv_s = one_bf16 / s
    inv_s = inv_s.view((num_groups, 1))
    xq = torch.clamp(x * inv_s, min=fp8_min_bf16.item(), max=fp8_max_bf16.item()).to(
        fp8_dtype
    )

    xq = xq.view(x_og_shape)
    xs = s.view((-1, xq.size(-1) // group_size))
    return xq, xs


def silu_mul_quant(
    gate: torch.Tensor, up: torch.Tensor, group_size: int, ceil_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    assert gate.size(-1) % group_size == 0
    assert up.size(-1) % group_size == 0

    assert gate.dtype == torch.bfloat16
    assert up.dtype == torch.bfloat16

    act_bf16 = silu(gate)
    assert act_bf16.dtype == torch.bfloat16

    # act & mul
    a_m = act_bf16 * up
    assert a_m.dtype == torch.bfloat16

    q, s = do_quant(a_m, group_size, ceil_ue8m0)
    return q, s


def pack_scales(x: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
    """
    pack float32 scales into a int32 tensor
    """
    assert x.dtype == torch.float32
    E, T, G = x.size()

    # Add i32_padding here so we can view it as a i32 tensor later on.
    i32_padding = round_up(G, 4) - G
    ref_s_i8 = torch.empty((E, T, G + i32_padding), dtype=torch.uint8, device="cuda")
    for e in range(E):
        nt = tokens_per_expert[e].item()
        ref_s_i8[e, :nt, :G] = x[e, :nt].view(torch.int32) >> 23

    ref_s_i32 = ref_s_i8.view(torch.int32)

    return ref_s_i32


def ref_with_scale_fmt(
    E: int,
    T: int,
    H: int,
    group_size: int,
    tokens_per_expert: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    scale_fmt: DeepGemmQuantScaleFMT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The precision types of the operations triggered by this function
    match closely with the kernel implementation so we compare more
    accurately.
    """
    scale_dtype = (
        torch.int32 if scale_fmt == DeepGemmQuantScaleFMT.UE8M0 else torch.float32
    )
    ceil_ue8m0 = scale_fmt in [
        DeepGemmQuantScaleFMT.UE8M0,
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
    ]

    ref_q = torch.empty((E, T, H), dtype=fp8_dtype, device="cuda")
    ref_s_f32 = torch.empty(
        (E, T, cdiv(H, group_size)), dtype=torch.float32, device="cuda"
    )

    for e in range(E):
        nt = tokens_per_expert[e].item()
        if nt == 0:
            continue
        ref_q[e, :nt], ref_s_f32[e, :nt] = silu_mul_quant(
            gate[e, :nt], up[e, :nt], group_size, ceil_ue8m0=ceil_ue8m0
        )

    if scale_dtype == torch.float32:
        return ref_q, ref_s_f32

    assert scale_dtype == torch.int32
    return ref_q, pack_scales(ref_s_f32, tokens_per_expert)


def token_random(E, T, H2, tokens_per_expert):
    """
    Initialize each token in a random range so we test a range of
    scale values.
    """
    y = torch.empty((E, T, H2), dtype=torch.bfloat16, device="cuda")
    for e in range(E):
        for t in range(tokens_per_expert[e].item()):
            exp = random.choice(range(1, 20))
            y[e, t].uniform_(-(2**exp), 2**exp)
    return y


@pytest.mark.parametrize("E,T,H,fp8_type", CASES)
@torch.inference_mode()
def test_silu_mul_fp8_quant_deep_gemm(E: int, T: int, H: int, fp8_type: torch.dtype):
    group_size = 128
    current_platform.seed_everything(42)

    tokens_per_expert = torch.randint(
        low=0,
        high=T,
        size=(E,),
        dtype=torch.int32,
        device="cuda",
    )

    # Input tensor of shape (E, T, 2*H)
    y = token_random(E, T, 2 * H, tokens_per_expert)

    gate = y[..., :H].to(torch.bfloat16)
    up = y[..., H:].to(torch.bfloat16)

    scale_fmts = [
        DeepGemmQuantScaleFMT.FLOAT32,
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
        DeepGemmQuantScaleFMT.UE8M0,
    ]

    # Run the SiLU V2 kernel
    for scale_fmt in scale_fmts:
        y_q, y_s = persistent_masked_m_silu_mul_quant(
            y,
            tokens_per_expert,
            group_size=group_size,
            quant_scale_fmt=scale_fmt,
        )

        ref_y_q, ref_y_s = ref_with_scale_fmt(
            E, T, H, group_size, tokens_per_expert, gate, up, scale_fmt=scale_fmt
        )

        # deepgemm scales transform
        dg_scales = None
        if (
            has_deep_gemm()
            and current_platform.has_device_capability(100)
            and scale_fmt == DeepGemmQuantScaleFMT.UE8M0
        ):
            from deep_gemm import transform_sf_into_required_layout

            _q, _s = ref_with_scale_fmt(
                E,
                T,
                H,
                group_size,
                tokens_per_expert,
                gate,
                up,
                scale_fmt=DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
            )
            dg_scales = transform_sf_into_required_layout(
                sf=_s,
                mn=_q.size(1),
                k=_q.size(2),
                recipe=(1, 128, 128),
                num_groups=_q.size(0),
                is_sfa=True,
            )

        expected_scale_dtype = (
            torch.int32 if scale_fmt == DeepGemmQuantScaleFMT.UE8M0 else torch.float32
        )
        assert y_s.dtype == expected_scale_dtype
        assert ref_y_s.dtype == expected_scale_dtype

        for e in range(E):
            nt = tokens_per_expert[e].item()

            torch.testing.assert_close(
                y_q[e, :nt].to(torch.float32),
                ref_y_q[e, :nt].to(torch.float32),
            )

            if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
                G = H // group_size
                y_s_sliced = as_uint8(y_s[e])
                ref_s_sliced = as_uint8(ref_y_s[e])
                torch.testing.assert_close(y_s_sliced[:nt, :G], ref_s_sliced[:nt, :G])
                if dg_scales is not None:
                    dg_sliced = as_uint8(dg_scales[e])
                    torch.testing.assert_close(y_s_sliced[:nt, :G], dg_sliced[:nt, :G])
            else:
                torch.testing.assert_close(
                    y_s[e, :nt],
                    ref_y_s[e, :nt],
                )
