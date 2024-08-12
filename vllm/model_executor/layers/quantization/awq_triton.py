import torch

import triton
import triton.language as tl

import argparse

device = "cuda"


@triton.jit
def awq_dequantize_kernel(
        qweight_ptr,  # quantized matrix
        scales_ptr,  # scales, per group
        zeros_ptr,  # zeros, per group
        split_k_iters,  # Not used
        thx,  # Not used
        thy,  # Not used
        group_size,  # Should always be 128
        result_ptr,  # Output matrix
        num_cols,  # input num cols in qweight
        num_rows,  # input num rows in qweight
        BLOCK_SIZE_X: tl.constexpr,
        BLOCK_SIZE_Y: tl.constexpr):
    # Setup the pids.
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Compute offsets and masks for qweight_ptr.
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X * 8) // 8
    offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols

    masks = masks_y[:, None] & masks_x[None, :]

    # Compute offsets and masks for result output ptr.
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(
        0, BLOCK_SIZE_X * 8)
    result_offsets = (8 * num_cols * result_offsets_y[:, None] +
                      result_offsets_x[None, :])

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    # Load the weights.
    iweights = tl.load(qweight_ptr + offsets, masks)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = ((tl.arange(0, 2) * 4)[None, :] +
                                tl.arange(0, 4)[:, None]).reshape(8)

    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    iweights = (iweights >> shifts) & 0xF

    # Compute zero offsets and masks.
    zero_offsets_y = (pid_y * BLOCK_SIZE_Y // group_size +
                      tl.arange(0, BLOCK_SIZE_Y) // group_size)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X * 8) // 8
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    # Load the zeros.
    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks)

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    zeros = (zeros >> shifts) & 0xF

    # Compute scale offsets and masks.
    scale_offsets_y = (pid_y * BLOCK_SIZE_Y // group_size +
                       tl.arange(0, BLOCK_SIZE_Y) // group_size)
    scale_offsets_x = (pid_x * BLOCK_SIZE_X * 8 +
                       tl.arange(0, BLOCK_SIZE_X * 8))
    scale_offsets = (num_cols * 8 * scale_offsets_y[:, None] +
                     scale_offsets_x[None, :])
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    # Load the scales.
    scales = tl.load(scales_ptr + scale_offsets, scale_masks)

    # Dequantize.
    iweights = (iweights - zeros) * scales
    iweights = iweights.to(tl.float16)

    # Finally, store.
    tl.store(result_ptr + result_offsets, iweights, result_masks)


@triton.jit
def awq_gemm_kernel(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, K,
                    awq_group_size, stride_am, stride_ak, stride_bk, stride_bn,
                    stride_cm, stride_cn, stride_zk, stride_zn, stride_sk,
                    stride_sn, BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                    SPLIT_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = tl.float16

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # accumulator = tl.arange(0, BLOCK_SIZE_N)
    # accumulator = tl.broadcast_to(accumulator[None, :],
    # (BLOCK_SIZE_M, BLOCK_SIZE_N))
    # accumulator = accumulator & 0x0
    # accumulator = accumulator.to(accumulator_dtype)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                           dtype=accumulator_dtype)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = ((tl.arange(0, 2) * 4)[None, :] +
                                tl.arange(0, 4)[:, None]).reshape(8)

    # Create the necessary shifts to use to unpack.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :],
                             (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M

    offsets_bn = (pid_n * (BLOCK_SIZE_N // 8) +
                  tl.arange(0, BLOCK_SIZE_N) // 8)
    masks_bn = offsets_bn < N // 8

    offsets_zn = (pid_n * (BLOCK_SIZE_N // 8) +
                  tl.arange(0, BLOCK_SIZE_N) // 8)
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
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)

        # Dequantize b.
        offsets_szk = ((BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K) //
                       awq_group_size +
                       tl.arange(0, BLOCK_SIZE_K) // awq_group_size)
        offsets_z = (N // 8) * offsets_szk[:, None] + offsets_zn[None, :]
        masks_zk = offsets_szk < K // awq_group_size
        masks_z = masks_zk[:, None] & masks_zn[None, :]
        zeros_ptrs = zeros_ptr + offsets_z
        zeros = tl.load(zeros_ptrs, mask=masks_z)

        offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
        masks_sk = offsets_szk < K // awq_group_size
        masks_s = masks_sk[:, None] & masks_sn[None, :]
        scales_ptrs = scales_ptr + offsets_s
        scales = tl.load(scales_ptrs, mask=masks_s)

        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        b = (b - zeros) * scales
        b = b.to(tl.float16)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N // 8)

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:,
                                         None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


# Example input:
#   qweight.size=torch.Size([3584, 576]),
#   qweight.dtype = torch.int32,
#   scales.size=torch.Size([28, 4608]),
#   scales.dtype=torch.float16,
#   zeros.size=torch.Size([28, 576]),
#   zeros.dtype=torch.int32
#   split_k_iters=0
#   thx=0
#   thy=0
def awq_dequantize_triton(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        split_k_iters: int,  # Not used
        thx: int,  # Not used
        thy: int  # Not used
) -> torch.Tensor:
    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.empty(qweight.shape[0],
                         qweight.shape[1] * 8,
                         device=qweight.device,
                         dtype=torch.float16)

    block_size_x = 32
    block_size_y = 32

    Y = qweight.shape[0]  # num rows
    X = qweight.shape[1]  # num cols
    group_size = 128
    grid = lambda META: (
        triton.cdiv(X, META['BLOCK_SIZE_X']),
        triton.cdiv(Y, META['BLOCK_SIZE_Y']),
    )
    awq_dequantize_kernel[grid](qweight,
                                scales,
                                zeros,
                                split_k_iters,
                                thx,
                                thy,
                                group_size,
                                result,
                                X,
                                Y,
                                BLOCK_SIZE_X=block_size_x,
                                BLOCK_SIZE_Y=block_size_y)

    return result


def reverse_awq_order(t: torch.Tensor):
    bits = 4
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_order_tensor = torch.arange(
        t.shape[-1],
        dtype=torch.int32,
        device=t.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    t = t[:, reverse_order_tensor] & 0xF
    return t


# qweights - [R     , C // 8], int32
# scales   - [R // G, C     ], float16
# zeros    - [R // G, C // 8], int32
def awq_dequantize_torch(qweight: torch.Tensor, scales: torch.Tensor,
                         qzeros: torch.Tensor, split_k_iters: int, thx: int,
                         thy: int) -> torch.Tensor:
    bits = 4
    group_size = 128
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    iweights = torch.bitwise_right_shift(qweight[:, :, None],
                                         shifts[None, None, :]).to(torch.int8)

    iweights = iweights.view(iweights.shape[0], -1)

    zeros = torch.bitwise_right_shift(qzeros[:, :, None],
                                      shifts[None, None, :]).to(torch.int8)
    zeros = zeros.view(qzeros.shape[0], -1)
    zeros = reverse_awq_order(zeros)

    iweights = reverse_awq_order(iweights)

    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)

    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = zeros.repeat_interleave(group_size, dim=0)
    return (iweights - zeros) * scales, zeros


def test_dequantize():
    print("=" * 10 + " TESTING DEQUANTIZE" + "=" * 10)
    use_triton = True
    use_torch = True

    qweight_rows = 3584
    qweight_cols = 576
    group_size = 128
    small_test_size = False
    if small_test_size:
        qweight_rows = 256
        qweight_cols = 128
    print(f"qweight_rows = {qweight_rows}, qweight_cols = {qweight_cols}")
    qweight_dtype = torch.int32
    scales_rows = qweight_rows // group_size
    scales_cols = qweight_cols * 8
    scales_dtype = torch.float16
    zeros_rows = scales_rows
    zeros_cols = qweight_cols
    zeros_dtype = torch.int32
    split_k_iters = 0
    thx = 0
    thy = 0
    torch.manual_seed(0)

    qweight = torch.randint(0,
                            10000000, (qweight_rows, qweight_cols),
                            dtype=qweight_dtype,
                            device=device)
    scales = torch.rand(scales_rows,
                        scales_cols,
                        dtype=scales_dtype,
                        device=device)
    zeros = torch.randint(0,
                          10000000, (zeros_rows, zeros_cols),
                          dtype=zeros_dtype,
                          device=device)
    print(f"qweight = {qweight}")
    if use_triton:
        iweights_triton = awq_dequantize_triton(qweight, scales, zeros,
                                                split_k_iters, thx, thy)
        print(f"Triton result:iweights_triton = {iweights_triton}")
        print("Any infs in triton result? -->"
              f"{torch.any(torch.isinf(iweights_triton))}")

    if use_torch:
        iweights_torch, _ = awq_dequantize_torch(qweight, scales, zeros,
                                                 split_k_iters, thx, thy)
        print(f"Torch result:iweights_torch = {iweights_torch}")

    if use_torch and use_triton:
        diff = iweights_torch - iweights_triton
        error = torch.sum(torch.sqrt(diff * diff))
        print(f"error = {error}")


def awq_gemm_triton(input: torch.Tensor, qweight: torch.Tensor,
                    scales: torch.Tensor, qzeros: torch.Tensor,
                    split_k_iters: int) -> torch.Tensor:
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(
            N, META['BLOCK_SIZE_N']),
        split_k_iters,
    )
    M, K = input.shape
    N = qweight.shape[1] * 8
    awq_group_size = 128
    block_size_m = 32
    block_size_n = 32
    block_size_k = 32

    result = torch.zeros((M, N), dtype=torch.float16, device=input.device)

    # A = input, B = qweight, C = result
    # A = M x K, B = K x N, C = M x N
    awq_gemm_kernel[grid](input,
                          qweight,
                          result,
                          qzeros,
                          scales,
                          M,
                          N,
                          K,
                          awq_group_size,
                          input.stride(0),
                          input.stride(1),
                          qweight.stride(0),
                          qweight.stride(1),
                          result.stride(0),
                          result.stride(1),
                          qzeros.stride(0),
                          qzeros.stride(1),
                          scales.stride(0),
                          scales.stride(1),
                          BLOCK_SIZE_M=block_size_m,
                          BLOCK_SIZE_N=block_size_n,
                          BLOCK_SIZE_K=block_size_k,
                          SPLIT_K=split_k_iters)

    return result


# input   - [N, K]
# qweight - [K, M // 8]
# qzeros  - [K // G, M // 8]
# scales  - [K // G, M]
# split_k_iters - parallelism along K-dimension, int, power of 2.
def awq_gemm_torch(input: torch.Tensor, qweight: torch.Tensor,
                   scales: torch.Tensor, qzeros: torch.Tensor,
                   split_k_iters: int) -> torch.Tensor:
    input_rows, input_cols = input.shape
    qweight_rows, qweight_cols = qweight.shape
    scales_rows, scales_cols = scales.shape
    print(f"awq_gemm_torch:input_rows = {input_rows} input_cols = {input_cols}"
          f" qweight_rows = {qweight_rows} qweight_cols = {qweight_cols}"
          f" scales_rows = {scales_rows} scales_cols = {scales_cols}")
    weights, zeros = awq_dequantize_torch(qweight, scales, qzeros,
                                          split_k_iters, 0, 0)
    return torch.matmul(input, weights)


def test_gemm():
    print("=" * 10 + " TESTING GEMM " + "=" * 10)

    split_k_iters = 1
    group_size = 128

    small_test_size = True

    # input.size = torch.Size([1, 3584]),
    # input.dtype = torch.float16
    # qweight.size = torch.Size([3584, 448]),
    # qweight.dtype = torch.int32
    # qzeros.size = torch.Size([28, 3584]),
    # qzeros.dtype = torch.float16
    # scales.size = torch.Size([28, 448]),
    # scales.dtype = torch.int32
    # split_k_iters = 8

    use_save_file = False

    if not use_save_file:
        input_rows = 1
        input_cols = 256 if small_test_size else 3584
        input_dtype = torch.float16
        qweight_rows = input_cols
        qweight_cols = 32 if small_test_size else 448
        scales_rows = qweight_rows // group_size
        scales_cols = qweight_cols * 8
        scales_dtype = torch.float16
        qzeros_rows = scales_rows
        qzeros_cols = qweight_cols
        print(f"input_rows = {input_rows} input_cols = {input_cols}"
              f" qweight_rows = {qweight_rows} qweight_cols = {qweight_cols}"
              f" scales_rows = {scales_rows} scales_cols = {scales_cols}")

        torch.manual_seed(2)
        input = torch.rand((input_rows, input_cols),
                           dtype=input_dtype,
                           device=device)
        qweight = torch.randint(0,
                                torch.iinfo(torch.int32).max,
                                (qweight_rows, qweight_cols),
                                device=device)
        qzeros = torch.randint(0,
                               torch.iinfo(torch.int32).max,
                               (qzeros_rows, qzeros_cols),
                               device=device)
        scales = torch.rand((scales_rows, scales_cols),
                            dtype=scales_dtype,
                            device=device)
    else:
        save_file = "/source/awq_gemm.pt"
        input, qweight, qzeros, scales, split_k_iters = torch.load(save_file)
        input = input.to(device=device)
        qweight = qweight.to(device=device)
        qzeros = qzeros.to(device=device)
        scales = scales.to(device=device)
        input = input.to(device=device)
        input_rows, input_cols = input.shape
        qweight_rows, qweight_cols = qweight.shape
        qzeros_rows, qzeros_cols = qzeros.shape
        qzeros_rows, qzeros_cols = qzeros.shape
        input_rows, input_cols = input.shape

    use_triton = True
    use_torch = True

    # NOTE: Use to see more data and accuracy during testing.
    # import numpy as np
    # import sys
    # torch.set_printoptions(precision = 3,
    #                        threshold=10000000000000000000000000000,
    #                        sci_mode = False)
    # np.set_printoptions(threshold=sys.maxsize)

    if use_torch:
        output_torch = awq_gemm_torch(input.cpu(), qweight.cpu(), scales.cpu(),
                                      qzeros.cpu(), split_k_iters)
        print(f"output_torch = {output_torch}")

    if use_triton:
        output_triton = awq_gemm_triton(input, qweight, scales, qzeros,
                                        split_k_iters)

        print(f"output_triton = {output_triton}")
        print(f"output_triton.shape = {output_triton.shape}")
        print(f"Any infs in triton result? --> "
              f"{torch.any(torch.isinf(output_triton))}")

    if use_torch and use_triton:
        diff = output_torch.cpu() - output_triton.cpu()
        error = torch.sum(torch.sqrt(diff * diff) / torch.numel(diff))
        print(f"error = {error}")


def main():
    parser = argparse.ArgumentParser(
        description="awq_triton test driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test")
    known_args, unknown_args = parser.parse_known_args()
    if known_args.test is not None:
        if known_args.test == "dequantize":
            test_dequantize()
        elif known_args.test == "gemm":
            test_gemm()
        else:
            print(f"Unknown test {known_args.test}")
    else:
        print("No test provided.")
        parser.print_help()


if __name__ == '__main__':
    main()
