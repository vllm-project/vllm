# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from functools import cache

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def is_weak_contiguous(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


# Tile configs are (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages).
# num_warps/num_stages of None means "use the Triton defaults" and GROUP_SIZE_M=1
# reduces the grouped program ordering to the original row-major order, so a
# (bm, bn, bk, 1, None, None) entry compiles to exactly the pre-existing kernel.
TileConfig = tuple[int, int, int, int, int | None, int | None]

# Tuned tile configs for NVIDIA GPUs, from an offline sweep over tile sizes,
# num_warps, num_stages and grouped program ordering on representative W8A8
# linear shapes (see https://github.com/vllm-project/vllm/issues/44998 for
# data and methodology). Keyed by (m_bucket, is_small_n), where m_bucket is
# next_power_of_2(M) clamped to [32, 1024] and is_small_n is N < 8192,
# mirroring the buckets of the default heuristic below.
_NVIDIA_TUNED_TILES: dict[tuple[int, int], dict[tuple[int, bool], TileConfig]] = {
    # Ada (e.g. L20/RTX 4090): 99KB shared memory favors small tiles with
    # shallow pipelines; (64, 64, 128) x 4 warps x 3 stages dominates M >= 256.
    (8, 9): {
        (32, True): (64, 64, 256, 8, 8, 3),
        (32, False): (64, 64, 256, 8, 8, 4),
        (64, True): (64, 64, 256, 8, 4, 4),
        (64, False): (64, 64, 256, 8, 4, 4),
        # The sweep found no config beating the default heuristic here.
        (128, True): (64, 128, 128, 1, None, None),
        (128, False): (64, 64, 256, 8, 8, 4),
        (256, True): (64, 64, 128, 8, 4, 3),
        (256, False): (64, 64, 128, 8, 4, 3),
        (512, True): (64, 64, 128, 8, 4, 3),
        (512, False): (64, 64, 128, 8, 4, 3),
        (1024, True): (64, 64, 128, 8, 4, 3),
        (1024, False): (64, 128, 128, 8, 4, 3),
    },
    # Hopper (e.g. H800/H100): 227KB shared memory rewards deeper pipelines
    # (5 stages at small M) and 8 warps on the (128, 128, 128) tile at large M.
    (9, 0): {
        (32, True): (64, 64, 256, 8, 4, 5),
        (32, False): (64, 64, 128, 8, 4, 5),
        (64, True): (64, 64, 256, 8, 4, 5),
        (64, False): (64, 64, 128, 8, 4, 5),
        (128, True): (64, 128, 256, 8, 8, 4),
        (128, False): (128, 128, 128, 8, 8, 5),
        (256, True): (64, 64, 128, 8, 4, 4),
        (256, False): (128, 64, 128, 8, 4, 4),
        (512, True): (64, 128, 128, 8, 8, 3),
        (512, False): (128, 128, 128, 8, 8, 3),
        (1024, True): (128, 128, 128, 8, 8, 3),
        (1024, False): (128, 128, 128, 8, 8, 3),
    },
}


@cache
def _tuned_tiles_for_current_device() -> dict[tuple[int, bool], TileConfig] | None:
    """Resolve the tuned table for this device once; None -> use the default
    heuristic. Cached because get_device_capability is not free on all paths."""
    if not current_platform.is_cuda():
        return None
    capability = current_platform.get_device_capability()
    if capability is None:
        return None
    return _NVIDIA_TUNED_TILES.get((capability.major, capability.minor))


def _get_tile_config(M: int, N: int, K: int) -> TileConfig:
    tuned = _tuned_tiles_for_current_device()
    if tuned is not None:
        m_bucket = min(max(32, triton.next_power_of_2(M)), 1024)
        return tuned[(m_bucket, N < 8192)]
    # Default heuristic (introduced in #11698, tuned on AMD): keyed on M only.
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
    block_m, block_n, block_k = tile_shape
    return (block_m, block_n, block_k, 1, None, None)


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
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # L2-cache-friendly grouped program ordering. NOTE: when GROUP_SIZE_M = 1,
    # this reduces exactly to the original row-major order
    # (pid_m = pid // num_pid_n, pid_n = pid % num_pid_n), so configs with
    # GROUP_SIZE_M=1 behave identically to the pre-existing kernel.
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

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

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)

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
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


# input   - [M, K]
# weight - [K, N]
def triton_scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
    use_heuristic=True,
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

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    result = torch.empty((M, N), dtype=out_dtype, device=input.device)

    has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1

    if use_heuristic:
        (
            block_size_m,
            block_size_n,
            block_size_k,
            group_size_m,
            num_warps,
            num_stages,
        ) = _get_tile_config(M, N, K)
    else:
        group_size_m = 1
        num_warps = None
        num_stages = None
    kwargs = {}
    if num_warps is not None:
        kwargs["num_warps"] = num_warps
    if num_stages is not None:
        kwargs["num_stages"] = num_stages

    block_size_sa = 1 if has_scalar(scale_a) else block_size_m
    block_size_sb = 1 if has_scalar(scale_b) else block_size_n

    accumulator_dtype = tl.float32 if input.is_floating_point() else tl.int32

    # A = input, B = weight, C = result
    # A = M x K, B = K x N, C = M x N
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
        GROUP_SIZE_M=group_size_m,
        **kwargs,
    )

    return result.to(out_dtype)
