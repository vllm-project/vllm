from typing import List, Optional

import torch
import triton
import triton.language as tl
# FIXME(woosuk): The performance model is not designed for quantized matmul.
# For the best performance, we need to implement a new performance model.
from triton.ops.matmul_perf_model import (
    early_config_prune, estimate_matmul_time)

from vllm import quantization_ops
from vllm.model_executor.layers.quantized_ops.matmul_utils import (
    get_configs_compute_bound, get_configs_io_bound)


def _prune_invalid_configs(
    configs: List[triton.Config],
    pack_factor: int,
    group_size: int,
) -> List[triton.Config]:
    valid_configs: List[triton.Config] = []
    for config in configs:
        block_n = config.kwargs['BLOCK_N']
        if block_n % pack_factor != 0:
            continue
        block_k = config.kwargs['BLOCK_K']
        if group_size % block_k != 0:
            continue
        valid_configs.append(config)
    return valid_configs


def _prune_configs(configs, named_args):
    pruned = early_config_prune(configs, named_args)
    pack_factor = named_args['AWQ_PACK_FACTOR']
    group_size = named_args['AWQ_GROUP_SIZE']
    return _prune_invalid_configs(pruned, pack_factor, group_size)


CONFIGS = get_configs_compute_bound() + get_configs_io_bound()
HEURISTICS = {
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
    'PACKED_BLOCK_N': lambda args: args['BLOCK_N'] // args['AWQ_PACK_FACTOR'],
}

# Grid: ((M // BLOCK_M) * (N // BLOCK_N), SPLIT_K)
@triton.autotune(
    configs=CONFIGS,
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': _prune_configs,
        'perf_model': estimate_matmul_time,
        'top_k': 1,
    },
)
@triton.heuristics(HEURISTICS)
@triton.jit
def _awq_kernel(
    A, B, C, M, N, K,
    Z, S, shifter_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_zk, stride_zn,
    stride_sk, stride_sn,
    AWQ_PACK_FACTOR: tl.constexpr,
    AWQ_GROUP_SIZE: tl.constexpr,
    PACKED_BLOCK_N: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,
):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    packed_rn = pid_n * PACKED_BLOCK_N + tl.arange(0, PACKED_BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(packed_rn % N, PACKED_BLOCK_N), PACKED_BLOCK_N)
    # rbn = packed_rn
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    AWQ_BIT_WIDTH = 32 // AWQ_PACK_FACTOR
    AWQ_MASK = (1 << AWQ_BIT_WIDTH) - 1
    weight_shifter = tl.load(shifter_ptr + tl.arange(0, AWQ_PACK_FACTOR))
    zero_shifter = tl.arange(0, AWQ_PACK_FACTOR) * AWQ_BIT_WIDTH

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)

        k_idx = pid_z * BLOCK_K + k * BLOCK_K * SPLIT_K
        awq_g_idx = k_idx // AWQ_GROUP_SIZE

        # 1. unpack b from [BLOCK_K, PACKED_BLOCK_N] to [BLOCK_K, BLOCK_N]
        b = (b[:, None, :] >> weight_shifter[None, :, None]) & AWQ_MASK
        b = tl.view(b, (BLOCK_K, BLOCK_N))

        # FIXME(woosuk): Currently, there's a bug in unpacking z.
        # As a temporary workaround, we unpack z before launching the kernel.

        # 2. load z: [PACKED_BLOCK_N]
        # z = tl.load(Z + awq_g_idx * stride_zk + rbn * stride_zn)
        # unpack z from [PACKED_BLOCK_N] to [BLOCK_N]
        # z = (z[:, None] * zero_shifter[None, :) & AWQ_MASK
        # z = tl.view(z, (1, BLOCK_N))
        z = tl.load(Z + awq_g_idx * stride_zk + rn * stride_zn)
        z = z.to(tl.int32)

        # 3. compute b - z
        b = (b - z).to(A.dtype.element_ty)

        # 4. load s: [BLOCK_N]
        s = tl.load(S + awq_g_idx * stride_sk + rn * stride_sn)

        # 5. compute b * s
        b = b * s[None, :]

        # 6. compute a @ b
        acc += tl.dot(a, b, out_dtype=dot_out_dtype)

        # 7. update pointers
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


def awq_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    pack_factor: int,
    group_size: int,
    shifter: Optional[torch.Tensor] = None,
    is_qzero_packed: bool = True,
) -> torch.Tensor:
    """Matrix multiplication for AWQ quantized weights.

    Args:
        a: An input activation tensor of shape `(M, K)`. FP16 or BF16.
        b: A packed weight tensor of shape `(K, N//P)`. INT32.
        qzeros: A tensor of shape `(K//G, N//P)`. INT32.
        scales: A tensor of shape `(K//G, N)`. FP16 or BF16.
        pack_factor: The packing factor abbreviated as `P`.
        group_size: The quantization group size abbreviated as `G`.
        shifter: A tensor of shape `(P,)`. INT32. The shifter for unpacking
            the packed weight tensor.
    """
    if pack_factor != 8:
        raise NotImplementedError("AWQ pack factor must be 8.")
    if group_size != 128:
        raise NotImplementedError("AWQ group size must be 128.")
    if shifter is None:
        shifter = torch.tensor(
            [0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32, device=a.device)
        shifter *= 4

    # Check if the tensors are contiguous.
    assert a.is_contiguous()
    assert b.is_contiguous()
    assert qzeros.is_contiguous()
    assert scales.is_contiguous()

    # Check dtypes.
    assert a.dtype in (torch.float16, torch.bfloat16)
    assert b.dtype == torch.int32
    if is_qzero_packed:
        assert qzeros.dtype == torch.int32
    else:
        assert qzeros.dtype == torch.int8
    assert scales.dtype == a.dtype

    # Check shapes.
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    _, PACKED_N = b.shape
    P = pack_factor
    N = P * PACKED_N
    G = group_size
    if is_qzero_packed:
        assert qzeros.shape == (K // G, PACKED_N)
    else:
        assert qzeros.shape == (K // G, N)
    assert scales.shape == (K // G, N)

    # FIXME: Unpack qzeros inside the kernel.
    if is_qzero_packed:
        qzeros = unpack_int32(qzeros, P, shifter)

    # Allocate output.
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    # Launch kernel.
    dot_out_dtype = tl.float32
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
    _awq_kernel[grid](a, b, c, M, N, K,
                    qzeros, scales, shifter,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    qzeros.stride(0), qzeros.stride(1),
                    scales.stride(0), scales.stride(1),
                    P, G,
                    dot_out_dtype=dot_out_dtype,
                    GROUP_M=8)
    return c


def unpack_int32(
    packed_tensor: torch.Tensor,
    pack_factor: int,
    shifter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert packed_tensor.dtype == torch.int32
    if shifter is None:
        shifter = torch.tensor(
            [0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32, device=packed_tensor.device)
        shifter *= 4

    bit_width = 32 // pack_factor
    bit_mask = (1 << bit_width) - 1
    packed_tensor = (packed_tensor[:, :, None] >> shifter[None, None, :]) & bit_mask
    packed_tensor = packed_tensor.to(torch.int8)
    return packed_tensor.view(packed_tensor.shape[0], -1)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    GROUP_SIZE = 128
    PACK_FACTOR = 8
    M = 12
    K = 256
    N = 128

    a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    b = torch.randint(0, 0x0fffffff, (K, N // PACK_FACTOR), dtype=torch.int32, device="cuda")
    qzeros = torch.randint(0, 0x0fffffff, (K // GROUP_SIZE, N // PACK_FACTOR), dtype=torch.int32, device="cuda")
    scales = torch.randn((K // GROUP_SIZE, N), dtype=torch.float16, device="cuda")

    c = awq_matmul(a, b, qzeros, scales, PACK_FACTOR, GROUP_SIZE)
    ans = quantization_ops.awq_gemm(a, b, scales, qzeros, PACK_FACTOR)
    print((c - ans).abs().max())
