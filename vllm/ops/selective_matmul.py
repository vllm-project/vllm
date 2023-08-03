import torch

import triton
import triton.language as tl


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_L': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, '1': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_L': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, '1': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_L': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, '1': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_L': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, '1': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_L': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, '1': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_L': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, '1': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_L': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, '1': 8}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_L': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, '1': 8}, num_stages=5, num_warps=2),
#     ],
#     key=['L', 'N', 'K'],
# )
@triton.jit
def selective_matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, index_ptr,
    # Matrix dimensions
    M, N, K, L,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_l = tl.cdiv(L, 1)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = 1 * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_l = group_id * 1
    group_size_l = min(num_pid_l - first_pid_l, 1)
    pid_l = first_pid_l + (pid % group_size_l)
    pid_n = (pid % num_pid_in_group) // group_size_l
    pid_m = tl.load(index_ptr + pid_l)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details

    offs_am = (pid_m * 1 + tl.arange(0, 1)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((1, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * 1 + tl.arange(0, 1)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


def selective_matmul(input, lora, index, output):
    # Check constraints.
    assert input.shape[1] == lora.shape[0], "Incompatible dimensions"
    assert input.is_contiguous(), "Matrix A must be contiguous"
    assert lora.is_contiguous(), "Matrix B must be contiguous"
    M, K = input.shape
    K, N = lora.shape
    L = index.shape[0]
    # Allocates output.
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    BLOCK_SIZE_N = 128 * 128
    BLOCK_SIZE_K = 32
    selective_matmul_kernel[grid](
        input, lora, output, index,
        M, N, K, L,
        input.stride(0), input.stride(1),
        lora.stride(0), lora.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_N, BLOCK_SIZE_K,
        num_warps=4,
        num_stages=3,
    )