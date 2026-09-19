# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    kernel_launcher,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
AWQ_FUSED_FP32_SUPPORTED = current_platform.is_cuda() and (
    current_platform.is_device_capability(89)
)


@triton.jit(do_not_specialize=["M"])
def awq_gemm_fused_fp32_kernel(
    input_ptr,
    qweight_ptr,
    scales_ptr,
    zeros_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    packed_offsets_n = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    packed_mask_n = packed_offsets_n < N // 8

    reverse_order = ((tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]).reshape(
        8
    )
    shifts = reverse_order * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offsets_k_base = tl.arange(0, BLOCK_SIZE_K)
    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offsets_k = k_block * BLOCK_SIZE_K + offsets_k_base
        mask_k = offsets_k < K
        inputs = tl.load(
            input_ptr + offsets_m[:, None] * K + offsets_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        packed = tl.load(
            qweight_ptr + offsets_k[:, None] * (N // 8) + packed_offsets_n[None, :],
            mask=mask_k[:, None] & packed_mask_n[None, :],
            other=0,
        )
        packed = tl.interleave(packed, packed)
        packed = tl.interleave(packed, packed)
        packed = tl.interleave(packed, packed)
        weights = (packed >> shifts) & 0xF

        group_offsets = offsets_k // GROUP_SIZE
        packed_zeros = tl.load(
            zeros_ptr + group_offsets[:, None] * (N // 8) + packed_offsets_n[None, :],
            mask=mask_k[:, None] & packed_mask_n[None, :],
            other=0,
        )
        packed_zeros = tl.interleave(packed_zeros, packed_zeros)
        packed_zeros = tl.interleave(packed_zeros, packed_zeros)
        packed_zeros = tl.interleave(packed_zeros, packed_zeros)
        zeros = (packed_zeros >> shifts) & 0xF
        scales = tl.load(
            scales_ptr + group_offsets[:, None] * N + offsets_n[None, :],
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        weights = ((weights - zeros) * scales).to(inputs.dtype)
        accumulator = tl.dot(inputs, weights, accumulator)

    output = accumulator.to(output_ptr.type.element_ty)
    tl.store(
        output_ptr + offsets_m[:, None] * N + offsets_n[None, :],
        output,
        mask=mask_m[:, None] & mask_n[None, :],
    )


class AwqGemmFusedFp32Kernel(VllmTritonJitKernel["AwqGemmFusedFp32Kernel.CompileKey"]):
    """Warmup-aware wrapper for awq_gemm_fused_fp32_kernel.

    ``M`` (token count) is excluded from the compile key via
    ``do_not_specialize``; only ``BLOCK_SIZE_M``/``BLOCK_SIZE_N`` (chosen from
    which side of the M<=128 threshold a call falls on) and the per-layer
    weight shape (N, K, GROUP_SIZE) affect specialization.
    """

    @dataclass(frozen=True)
    class CompileKey:
        N: int
        K: int
        GROUP_SIZE: int
        BLOCK_SIZE_M: int
        BLOCK_SIZE_N: int
        BLOCK_SIZE_K: int = 32

    kernel: Any = staticmethod(awq_gemm_fused_fp32_kernel)

    def dispatch(  # type: ignore[override]
        self, *, m: int, N: int, K: int, GROUP_SIZE: int
    ) -> CompileKey:
        block_m = 32 if m <= 128 else 128
        block_n = 32 if m <= 128 else 64
        return self.CompileKey(
            N=N, K=K, GROUP_SIZE=GROUP_SIZE, BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n
        )

    def get_warmup_keys(
        self, *, N: int, K: int, GROUP_SIZE: int = 128
    ) -> list[CompileKey]:
        # One representative M on each side of the 128 threshold covers both
        # BLOCK_SIZE_M/N configs dispatch(...) can ever select.
        return self._trace_dispatch(self.dispatch)(
            m=(1, 129), N=N, K=K, GROUP_SIZE=GROUP_SIZE
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        m = 1 if compile_key.BLOCK_SIZE_M == 32 else 129
        num_groups = compile_key.K // compile_key.GROUP_SIZE
        return dict(
            input=TritonWarmupTensor(torch.float16, shape=(m, compile_key.K)),
            qweight=TritonWarmupTensor(
                torch.int32, shape=(compile_key.K, compile_key.N // 8)
            ),
            scales=TritonWarmupTensor(
                torch.float16, shape=(num_groups, compile_key.N)
            ),
            zeros=TritonWarmupTensor(
                torch.int32, shape=(num_groups, compile_key.N // 8)
            ),
        )

    @kernel_launcher
    def __call__(
        self,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
    ) -> LaunchSpec:
        m, k = input.shape
        n = qweight.shape[1] * 8
        group_size = k // scales.shape[0]
        compile_key = self.dispatch(m=m, N=n, K=k, GROUP_SIZE=group_size)
        output = (
            TritonWarmupTensor(input.dtype, shape=(m, n))
            if self._warming
            else torch.empty((m, n), device=input.device, dtype=input.dtype)
        )
        grid = (
            triton.cdiv(m, compile_key.BLOCK_SIZE_M)
            * triton.cdiv(n, compile_key.BLOCK_SIZE_N),
        )
        return (
            grid,
            dict(
                output_ptr=output,
                M=m,
                N=compile_key.N,
                K=compile_key.K,
                GROUP_SIZE=compile_key.GROUP_SIZE,
                BLOCK_SIZE_M=compile_key.BLOCK_SIZE_M,
                BLOCK_SIZE_N=compile_key.BLOCK_SIZE_N,
                BLOCK_SIZE_K=compile_key.BLOCK_SIZE_K,
                num_warps=4,
                num_stages=2,
            ),
            output,
        )


_AWQ_GEMM_FUSED_FP32_KERNEL = AwqGemmFusedFp32Kernel()


def register_awq_fused_fp32_warmup(*, N: int, K: int, GROUP_SIZE: int = 128) -> None:
    _AWQ_GEMM_FUSED_FP32_KERNEL.register_warmup(N=N, K=K, GROUP_SIZE=GROUP_SIZE)


def _awq_gemm_fused_fp32_impl(
    inputs: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
) -> torch.Tensor:
    m, k = inputs.shape
    n = qweight.shape[1] * 8
    num_groups = scales.shape[0]
    if inputs.dtype != torch.float16 or scales.dtype != torch.float16:
        raise ValueError("fused AWQ GEMM supports FP16 inputs and scales only")
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError("fused AWQ GEMM requires int32 packed weights and zeros")
    if not all(tensor.is_contiguous() for tensor in (inputs, qweight, scales, qzeros)):
        raise ValueError("fused AWQ GEMM requires contiguous tensors")
    if qweight.shape[0] != k:
        raise ValueError("fused AWQ GEMM weight K does not match the input")
    if num_groups == 0 or k % num_groups != 0:
        raise ValueError(
            "fused AWQ GEMM requires K to be an exact multiple of the number "
            "of quantization groups"
        )
    group_size = k // num_groups
    if group_size != 128:
        raise ValueError("fused AWQ GEMM supports group_size=128 only")
    if scales.shape != (num_groups, n):
        raise ValueError("fused AWQ GEMM scales have an invalid shape")
    if qzeros.shape != (num_groups, n // 8):
        raise ValueError("fused AWQ GEMM zeros have an invalid shape")
    if k % 32 != 0 or n % 32 != 0:
        raise ValueError("fused AWQ GEMM requires K and N aligned to 32")

    return _AWQ_GEMM_FUSED_FP32_KERNEL(inputs, qweight, scales, qzeros)


def _awq_gemm_fused_fp32_fake(
    inputs: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
) -> torch.Tensor:
    del scales, qzeros
    return torch.empty(
        (inputs.size(0), qweight.size(1) * 8),
        dtype=inputs.dtype,
        device=inputs.device,
    )


direct_register_custom_op(
    op_name="awq_gemm_fused_fp32",
    op_func=_awq_gemm_fused_fp32_impl,
    mutates_args=[],
    fake_impl=_awq_gemm_fused_fp32_fake,
    dispatch_key=current_platform.dispatch_key,
)


def awq_gemm_fused_fp32(
    inputs: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.vllm.awq_gemm_fused_fp32(inputs, qweight, scales, qzeros)


@triton.jit
def awq_dequantize_kernel(
    qweight_ptr,  # quantized matrix
    scales_ptr,  # scales, per group
    zeros_ptr,  # zeros, per group
    group_size,  # Should always be one of the supported group sizes
    result_ptr,  # Output matrix
    num_cols,  # input num cols in qweight
    num_rows,  # input num rows in qweight
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Set up the pids.
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Compute offsets and masks for qweight_ptr.
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols

    masks = masks_y[:, None] & masks_x[None, :]

    # Compute offsets and masks for result output ptr.
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    result_offsets = (
        8 * num_cols * result_offsets_y[:, None] + result_offsets_x[None, :]
    )

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    # Load the weights.
    iweights = tl.load(qweight_ptr + offsets, masks, 0.0)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)

    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    iweights = (iweights >> shifts) & 0xF

    # Compute zero offsets and masks.
    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    # Load the zeros.
    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks, 0.0)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    zeros = (zeros >> shifts) & 0xF

    # Compute scale offsets and masks.
    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    scale_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    scale_offsets = num_cols * 8 * scale_offsets_y[:, None] + scale_offsets_x[None, :]
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    # Load the scales.
    scales = tl.load(scales_ptr + scale_offsets, scale_masks, 0.0)
    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Dequantize.
    iweights = (iweights - zeros) * scales
    iweights = iweights.to(result_ptr.type.element_ty)

    # Finally, store.
    tl.store(result_ptr + result_offsets, iweights, result_masks)


@triton.jit
def awq_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    zeros_ptr,
    scales_ptr,
    M,
    N,
    K,
    group_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = c_ptr.type.element_ty

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # accumulator = tl.arange(0, BLOCK_SIZE_N)
    # accumulator = tl.broadcast_to(accumulator[None, :],
    # (BLOCK_SIZE_M, BLOCK_SIZE_N))
    # accumulator = accumulator & 0x0
    # accumulator = accumulator.to(accumulator_dtype)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)

    # Create the necessary shifts to use to unpack.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M

    offsets_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_bn = offsets_bn < N // 8

    offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_zn = offsets_zn < N // 8

    offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    masks_sn = offsets_sn < N

    offsets_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_a = K * offsets_am[:, None] + offsets_k[None, :]
    offsets_b = (N // 8) * offsets_k[:, None] + offsets_bn[None, :]

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    # NOTE: Use this in TRITON_INTERPRET=1 mode instead of tl.cdiv
    # block_offset = BLOCK_SIZE_K * SPLIT_K
    # for k in range(0, (K + block_offset - 1) // (block_offset)):
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a, other=0.0)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b, other=0.0)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        # Dequantize b.
        offsets_szk = (
            BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K
        ) // group_size + tl.arange(0, 1)
        offsets_z = (N // 8) * offsets_szk[:, None] + offsets_zn[None, :]
        masks_zk = offsets_szk < K // group_size
        masks_z = masks_zk[:, None] & masks_zn[None, :]
        zeros_ptrs = zeros_ptr + offsets_z
        zeros = tl.load(zeros_ptrs, mask=masks_z, other=0.0)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
        masks_sk = offsets_szk < K // group_size
        masks_s = masks_sk[:, None] & masks_sn[None, :]
        scales_ptrs = scales_ptr + offsets_s
        scales = tl.load(scales_ptrs, mask=masks_s, other=0.0)
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        b = (b - zeros) * scales
        b = b.to(c_ptr.type.element_ty)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N // 8)

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + pid_z * N * M + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# qweights - [K     , M // 8], int32
# scales   - [K // G, M     ], float16
# zeros    - [K // G, M // 8], int32
def awq_dequantize_triton(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    block_size_x: int = 32,
    block_size_y: int = 32,
) -> torch.Tensor:
    K = qweight.shape[0]
    M = scales.shape[1]
    group_size = qweight.shape[0] // scales.shape[0]

    assert K > 0 and M > 0
    assert scales.shape[0] == K // group_size and scales.shape[1] == M
    assert zeros.shape[0] == K // group_size and zeros.shape[1] == M // 8
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.empty(
        qweight.shape[0],
        qweight.shape[1] * 8,
        device=qweight.device,
        dtype=scales.dtype,
    )

    Y = qweight.shape[0]  # num rows
    X = qweight.shape[1]  # num cols

    grid = lambda META: (
        triton.cdiv(X, META["BLOCK_SIZE_X"]),
        triton.cdiv(Y, META["BLOCK_SIZE_Y"]),
    )
    awq_dequantize_kernel[grid](
        qweight,
        scales,
        zeros,
        group_size,
        result,
        X,
        Y,
        BLOCK_SIZE_X=block_size_x,
        BLOCK_SIZE_Y=block_size_y,
    )

    return result


# input   - [M, K]
# qweight - [K, N // 8]
# qzeros  - [K // G, N // 8]
# scales  - [K // G, N]
# split_k_iters - parallelism along K-dimension, int, power of 2.
def awq_gemm_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
) -> torch.Tensor:
    M, K = input.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qzeros.shape[0]

    assert N > 0 and K > 0 and M > 0
    assert qweight.shape[0] == K and qweight.shape[1] == N // 8
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    assert split_k_iters & (split_k_iters - 1) == 0 and split_k_iters != 0
    assert split_k_iters <= 32
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        split_k_iters,
    )

    result = torch.zeros((split_k_iters, M, N), dtype=scales.dtype, device=input.device)

    # A = input, B = qweight, C = result
    # A = M x K, B = K x N, C = M x N
    awq_gemm_kernel[grid](
        input,
        qweight,
        result,
        qzeros,
        scales,
        M,
        N,
        K,
        group_size,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        SPLIT_K=split_k_iters,
    )

    result = result.sum(0)

    return result
