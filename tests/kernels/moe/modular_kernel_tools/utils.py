# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm._custom_ops as ops
from vllm.utils.deep_gemm import per_block_cast_to_fp8


def per_token_cast_to_fp8(
        x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (block_size - (n % block_size)) % block_size
    x = torch.nn.functional.pad(x,
                                (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, block_size)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)


def make_non_quant_weights(
    e: int,
    n: int,
    k: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return weights w1, w2
    """
    device = torch.cuda.current_device()
    w1 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / 15
    w2 = torch.randn((e, k, n), device=device, dtype=dtype) / 15
    return w1, w2


def make_block_quant_fp8_weights(
    e: int,
    n: int,
    k: int,
    block_size: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return weights w1, w2, w1_scale, w2_scale
    """
    dtype = torch.bfloat16
    device = torch.cuda.current_device()

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    w1_bf16, w2_bf16 = make_non_quant_weights(e, n, k, dtype)
    w1_bf16 = w1_bf16.clamp(min=fp8_min, max=fp8_max).to(dtype=dtype)
    w2_bf16 = w2_bf16.clamp(min=fp8_min, max=fp8_max).to(dtype=dtype)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles_w1 = ((2 * n) + block_n - 1) // block_n
    k_tiles_w1 = (k + block_k - 1) // block_k
    n_tiles_w2 = (k + block_n - 1) // block_n
    k_tiles_w2 = (n + block_k - 1) // block_k

    w1 = torch.empty_like(w1_bf16, dtype=torch.float8_e4m3fn, device=device)
    w2 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn, device=device)

    w1_s = torch.empty((e, n_tiles_w1, k_tiles_w1),
                       device=device,
                       dtype=torch.float32)
    w2_s = torch.empty((e, n_tiles_w2, k_tiles_w2),
                       device=device,
                       dtype=torch.float32)

    assert w1_s.shape == (e, (2 * n + (block_n - 1)) // block_n,
                          (k + (block_k - 1)) // block_k)
    assert (w2.shape[-2] + block_n - 1) // block_n == w2_s.shape[-2]

    for i in range(e):
        w1[i], w1_s[i] = per_block_cast_to_fp8(w1_bf16[i],
                                               block_size=[block_k, block_n])
        w2[i], w2_s[i] = per_block_cast_to_fp8(w2_bf16[i],
                                               block_size=[block_k, block_n])

    return w1, w2, w1_s, w2_s


def make_quant_fp8_weights(
    e: int,
    n: int,
    k: int,
    per_out_channel_quant: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return w1, w2, w1_scale, w2_scale
    """
    q_dtype = torch.float8_e4m3fn

    w1, w2 = make_non_quant_weights(e, n, k, dtype=torch.bfloat16)

    # w1 -> w1_q, w2 -> w2_q
    w1_q = torch.empty((e, 2 * n, k), device="cuda", dtype=q_dtype)
    w2_q = torch.empty((e, k, n), device="cuda", dtype=q_dtype)

    n_b_scales = 2 * n if per_out_channel_quant else 1
    k_b_scales = k if per_out_channel_quant else 1
    w1_scale = torch.empty((e, n_b_scales, 1),
                           device="cuda",
                           dtype=torch.float32)
    w2_scale = torch.empty((e, k_b_scales, 1),
                           device="cuda",
                           dtype=torch.float32)

    for expert in range(e):
        w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(
            w1[expert], use_per_token_if_dynamic=per_out_channel_quant)
        w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(
            w2[expert], use_per_token_if_dynamic=per_out_channel_quant)
    return w1_q, w2_q, w1_scale, w2_scale
