# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.triton_utils import tl, triton, use_tensor_descriptor
from vllm.triton_utils.allocation import set_triton_allocator

_TD_ALLOCATOR_DEVICES: set[torch.device] = set()


def is_weak_contiguous(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


@triton.jit
def _scaled_mm_tile_coords(
    pid,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    N_STRIPE: tl.constexpr,
):
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if N_STRIPE > 0:
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        stripe_span = num_pid_m * N_STRIPE
        stripe_id = pid // stripe_span
        first_n = stripe_id * N_STRIPE
        stripe_width = tl.minimum(N_STRIPE, num_pid_n - first_n)
        stripe_pid = pid % stripe_span
        pid_m = stripe_pid // stripe_width
        pid_n = first_n + stripe_pid % stripe_width
    elif GROUP_M > 1:
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_in_group = pid % num_pid_in_group
        pid_m = first_pid_m + pid_in_group % group_size_m
        pid_n = pid_in_group // group_size_m
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    return pid_m, pid_n


@triton.jit
def scaled_mm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACCUMULATOR_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_A: tl.constexpr,
    BLOCK_SIZE_SCALE_B: tl.constexpr,
    USE_TD: tl.constexpr = False,
    B_T: tl.constexpr = False,
    GROUP_M: tl.constexpr = 1,
    N_STRIPE: tl.constexpr = 0,
    UNMASKED_LOADS: tl.constexpr = False,
):
    pid_m, pid_n = _scaled_mm_tile_coords(
        tl.program_id(axis=0), M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_M, N_STRIPE
    )

    accumulator_dtype = ACCUMULATOR_DTYPE
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # NOTE: Some tensor inputs are so large, they will cause int32 overflow
    # so it is necessary to use tl.int64 for all the offsets, else SEGV will
    # eventually occur.

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    # NOTE: BLOCK_SIZE_SCALE_A could be 1 or BLOCK_SIZE_M, so need to create
    # appropriate offsets and masks for each case. Same goes for
    # BLOCK_SIZE_SCALE_B.
    offsets_scale_am = (
        tl.arange(0, BLOCK_SIZE_SCALE_A)
        + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M
    )
    masks_scale_am = offsets_scale_am < M

    offsets_scale_bn = (
        tl.arange(0, BLOCK_SIZE_SCALE_B)
        + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    )
    masks_scale_bn = offsets_scale_bn < N

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    scale_a_ptrs = scale_a_ptr + offsets_scale_am
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn

    if USE_TD:
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, K],
            strides=[stride_am, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        if B_T:
            # Checkpoint weights arrive as a transposed [N, K] view, so describe
            # the underlying N-major buffer and transpose each tile.
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[N, K],
                strides=[stride_bn, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
        else:
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[K, N],
                strides=[stride_bk, 1],
                block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
            )

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if USE_TD:
            a = a_desc.load([pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K])
            if B_T:
                b = tl.trans(b_desc.load([pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K]))
            else:
                b = b_desc.load([k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])
        else:
            masks_k = offsets_k < K
            masks_a = masks_am[:, None] & masks_k[None, :]
            a = tl.load(a_ptrs) if UNMASKED_LOADS else tl.load(a_ptrs, mask=masks_a)

            masks_b = masks_k[:, None] & masks_bn[None, :]
            b = tl.load(b_ptrs) if UNMASKED_LOADS else tl.load(b_ptrs, mask=masks_b)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply scale at end.
    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_a = tl.load(scale_a_ptrs[:, None], masks_scale_a)
    # Need to broadcast to the appropriate size, if scale_a is already
    # (BLOCK_SIZE_M, 1) then it will broadcast to its own shape. Same goes
    # for scale_b below.
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))
    accumulator = scale_a * accumulator.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    accumulator = scale_b.T * accumulator.to(tl.float32)

    # Convert to output format.
    c = accumulator.to(c_ptr.type.element_ty)

    # Add bias, it's already in output format, so add it after conversion.
    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        c += bias

    # Save output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


def _get_scaled_mm_launch_policy(
    input: torch.Tensor,
    weight: torch.Tensor,
    tile_shape: tuple[int, int, int],
    use_td: bool,
    group_m: int,
    n_stripe: int,
    unmasked_loads: bool,
) -> tuple[int, int, bool]:
    """Validate requested scheduling against the actual operand layout and tiles."""
    if (group_m == 1 and n_stripe == 0 and not unmasked_loads) or use_td:
        return 1, 0, False
    M, K = input.shape
    N = weight.shape[1]
    if input.stride() != (K, 1) or weight.stride() != (1, K):
        return 1, 0, False

    block_m, block_n, block_k = tile_shape
    num_pid_m, num_pid_n = triton.cdiv(M, block_m), triton.cdiv(N, block_n)
    max_pid = 2**31 - 1
    if group_m * num_pid_n > max_pid:
        group_m = 1
    if n_stripe and num_pid_m * max(num_pid_n, n_stripe) > max_pid:
        n_stripe = 0
    if n_stripe:
        group_m = 1

    unmasked_loads = (
        unmasked_loads
        and input.element_size() == weight.element_size() == 1
        and input.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and K % 16 == 0
        and block_k % 16 == 0
        and M % block_m == N % block_n == K % block_k == 0
    )
    return group_m, n_stripe, unmasked_loads


# input   - [M, K]
# weight - [K, N]
def triton_scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
    use_heuristic=True,
    use_td: bool | None = None,
    num_warps: int = 4,
    num_stages: int | None = None,
    group_m: int = 1,
    n_stripe: int = 0,
    unmasked_loads: bool = False,
) -> torch.Tensor:
    M, K = input.shape
    N = weight.shape[1]

    assert N > 0 and K > 0 and M > 0
    assert weight.shape[0] == K
    assert input.dtype == weight.dtype

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b

    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)
    assert out_dtype.is_floating_point
    assert bias is None or bias.is_floating_point()
    assert is_weak_contiguous(input)
    assert is_weak_contiguous(weight)
    assert group_m >= 1 and n_stripe >= 0

    # Scale addressing in the kernel assumes consecutive elements.
    scale_a, scale_b = scale_a.contiguous(), scale_b.contiguous()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    result = torch.empty((M, N), dtype=out_dtype, device=input.device)

    has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1

    if use_heuristic:
        is_small_N = N < 8192
        next_power_of_2_M = max(32, triton.next_power_of_2(M))
        if next_power_of_2_M <= 32:
            tile_shape = (64, 64, 256) if is_small_N else (64, 128, 256)
        elif next_power_of_2_M <= 64:
            tile_shape = (64, 64, 256)
        elif next_power_of_2_M <= 128:
            tile_shape = (64, 128, 128)
        else:
            tile_shape = (128, 128, 128)

        block_size_m, block_size_n, block_size_k = tile_shape

    block_size_sa = 1 if has_scalar(scale_a) else block_size_m
    block_size_sb = 1 if has_scalar(scale_b) else block_size_n

    accumulator_dtype = tl.float32 if input.is_floating_point() else tl.int32

    # TD operand loads; gated on inner-dim contiguity and 16-byte tile alignment.
    # Checkpoint weights are a transposed [N, K] view, so accept either
    # orientation and let the kernel transpose per tile.
    b_t = weight.stride(1) != 1 and weight.stride(0) == 1
    b_inner = K if b_t else N
    use_td = (
        use_tensor_descriptor(use_td)
        and input.stride(1) == 1
        and (weight.stride(1) == 1 or b_t)
        and (K * input.element_size()) % 16 == 0
        and (b_inner * weight.element_size()) % 16 == 0
        and (block_size_m & (block_size_m - 1)) == 0
        and (block_size_n & (block_size_n - 1)) == 0
        and (block_size_k & (block_size_k - 1)) == 0
    )
    if use_td and input.device not in _TD_ALLOCATOR_DEVICES:
        set_triton_allocator(input.device)
        _TD_ALLOCATOR_DEVICES.add(input.device)

    group_m, n_stripe, unmasked_loads = _get_scaled_mm_launch_policy(
        input,
        weight,
        (block_size_m, block_size_n, block_size_k),
        use_td,
        group_m,
        n_stripe,
        unmasked_loads,
    )

    # A = input, B = weight, C = result
    # A = M x K, B = K x N, C = M x N
    launch_options = {} if num_stages is None else {"num_stages": num_stages}
    scaled_mm_kernel[grid](
        input,
        weight,
        scale_a,
        scale_b,
        result,
        bias,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        result.stride(0),
        result.stride(1),
        accumulator_dtype,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_SCALE_A=block_size_sa,
        BLOCK_SIZE_SCALE_B=block_size_sb,
        USE_TD=use_td,
        B_T=b_t,
        GROUP_M=group_m,
        N_STRIPE=n_stripe,
        UNMASKED_LOADS=unmasked_loads,
        num_warps=num_warps,
        **launch_options,
    )

    return result
