# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SwiGLU kernels for the Inkling MLP layers.

``silu_and_mul_triton``: SiLU-and-mul over the checkpoint's interleaved
fused gate/up layout (dense MLP). ``sink_silu_mul_epilogue``: the sink-expert
variant with the per-expert dequant scale and per-token gamma fused in.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["M"])
def _silu_and_mul_triton_kernel(
    gateup_out_ptr,
    down_inp_ptr,
    M,
    N: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    start_pid = tl.program_id(0)
    if INT64_INDEX:
        start_pid = start_pid.to(tl.int64)
        M = M.to(tl.int64)

    NUM_BLOCKS_N: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
    num_blocks_mn = tl.cdiv(M, BLOCK_SIZE_M) * NUM_BLOCKS_N

    for pid in tl.range(start_pid, num_blocks_mn, GRID_SIZE, num_stages=NUM_STAGES):
        pid_m = pid // NUM_BLOCKS_N
        pid_n = pid % NUM_BLOCKS_N

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        # Interleaved fused gate/up: [g0, u0, g1, u1, ...].
        mask_offs_2n = pid_n * BLOCK_SIZE_N + tl.arange(0, 2 * BLOCK_SIZE_N) // 2
        tl.static_assert(BLOCK_SIZE_N % 8 == 0, f"{BLOCK_SIZE_N=}")
        mask_2n = mask_offs_2n < N
        mask_2n = tl.max_constancy(mask_2n, [16])

        offs_2n = pid_n * 2 * BLOCK_SIZE_N + tl.arange(0, 2 * BLOCK_SIZE_N)
        offs_m2n = offs_m[:, None] * N * 2 + offs_2n[None, :]

        if EVEN_N or pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N:
            gateup_out = tl.load(
                gateup_out_ptr + offs_m2n, mask=mask_m[:, None], other=0.0
            )
        else:
            mask_m2n = mask_m[:, None] & mask_2n[None, :]
            gateup_out = tl.load(gateup_out_ptr + offs_m2n, mask=mask_m2n, other=0.0)

        gate_out, up_out = tl.split(
            tl.reshape(gateup_out, (BLOCK_SIZE_M, BLOCK_SIZE_N, 2))
        )
        gate_out = gate_out.to(tl.float32)
        up_out = up_out.to(tl.float32)

        down_inp = gate_out * tl.sigmoid(gate_out) * up_out

        mask_mn = mask_m[:, None] if EVEN_N else mask_m[:, None] & mask_n[None, :]
        offs_mn = offs_m[:, None] * N + offs_n[None, :]
        tl.store(down_inp_ptr + offs_mn, down_inp, mask=mask_mn)


def silu_and_mul_triton(gateup_output: torch.Tensor) -> torch.Tensor:
    """SiLU-and-mul for the interleaved fused gate/up layout.

    Adapted from ``inkling_kernels.activation.silu_and_mul_fwd`` (without MXFP).
    """
    assert gateup_output.is_contiguous(), (
        f"{gateup_output.shape=} {gateup_output.stride()=}"
    )
    assert gateup_output.ndim == 2, f"{gateup_output.shape=}"

    M = gateup_output.shape[0]
    hidden_size = gateup_output.shape[1]
    assert hidden_size % 2 == 0, f"{hidden_size=}"
    N = hidden_size // 2

    down_input = torch.empty(
        (M, N), device=gateup_output.device, dtype=gateup_output.dtype
    )
    if M == 0:
        return down_input

    BLOCK_SIZE_N = max(8, min(256, triton.next_power_of_2(N)))
    if M <= 1:
        BLOCK_SIZE_M = 4
    elif M <= 256:
        BLOCK_SIZE_M = 2
    elif M < 4096:
        BLOCK_SIZE_M = 4
    else:
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = max(8, min(128, triton.next_power_of_2(N)))
    max_grid_size = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)
    num_sms = torch.cuda.get_device_properties(
        gateup_output.device
    ).multi_processor_count
    grid_size = min(num_sms * 4, max_grid_size)

    _silu_and_mul_triton_kernel[(grid_size,)](
        gateup_out_ptr=gateup_output,
        down_inp_ptr=down_input,
        M=M,
        N=N,
        GRID_SIZE=grid_size,
        NUM_STAGES=1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        EVEN_N=N % BLOCK_SIZE_N == 0,
        INT64_INDEX=gateup_output.nbytes >= 2**31,
        num_warps=8,
    )

    return down_input


@triton.jit(do_not_specialize=["T"])
def _sink_epilogue_kernel(
    raw_ptr,  # [T, S * 2F] gemm1 output, interleaved g/u pairs per expert block
    alpha_ptr,  # [S] fp32 per-expert pre-SiLU dequant scale
    gamma_ptr,  # [T, S] fp32 per-token sink weights (may be strided)
    ratio_ptr,  # [S] fp32 per-expert post-SiLU scale (gemm2 alpha ratio)
    out_ptr,  # [T, S * F] output
    T,
    stride_raw_0,
    stride_gamma_0,
    F: tl.constexpr,
    S: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_sf = tl.program_id(1)
    if pid_t >= T:
        return
    s = pid_sf // (F // BLOCK_F)
    offs_f = (pid_sf % (F // BLOCK_F)) * BLOCK_F + tl.arange(0, BLOCK_F)

    base = pid_t * stride_raw_0 + s * 2 * F
    gate = tl.load(raw_ptr + base + 2 * offs_f).to(tl.float32)
    up = tl.load(raw_ptr + base + 2 * offs_f + 1).to(tl.float32)
    alpha = tl.load(alpha_ptr + s)
    weight = tl.load(gamma_ptr + pid_t * stride_gamma_0 + s) * tl.load(ratio_ptr + s)

    gate *= alpha
    up *= alpha
    h = gate * tl.sigmoid(gate) * up * weight
    tl.store(out_ptr + pid_t * (S * F) + s * F + offs_f, h)


def sink_silu_mul_epilogue(
    raw: torch.Tensor,  # [T, S * 2F] gemm1 output (interleaved gate/up rows)
    alphas: torch.Tensor,  # [S] fp32
    gammas: torch.Tensor,  # [T, S] fp32
    ratios: torch.Tensor,  # [S] fp32
    n_experts: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Fused sink-expert epilogue: silu(g * a_e) * (u * a_e) * (gamma * r_e).

    One kernel replaces the per-expert dequant column scale, the SwiGLU, and
    the per-token gamma multiply between the two sink GEMMs.
    """
    tokens = raw.shape[0]
    f = raw.shape[1] // (2 * n_experts)
    out = torch.empty((tokens, n_experts * f), device=raw.device, dtype=out_dtype)
    if tokens == 0:
        return out
    # raw may be a column-slice of a padded GEMM output (rows strided).
    assert raw.stride(1) == 1 and gammas.stride(1) == 1
    # Largest power-of-two divisor of f (f = 768 -> 256), capped at 512.
    block_f = min(512, f & (-f))
    _sink_epilogue_kernel[(tokens, n_experts * (f // block_f))](
        raw,
        alphas,
        gammas,
        ratios,
        out,
        tokens,
        raw.stride(0),
        gammas.stride(0),
        F=f,
        S=n_experts,
        BLOCK_F=block_f,
    )
    return out
