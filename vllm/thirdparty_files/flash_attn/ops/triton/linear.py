# Adapted from https://github.com/ELS-RD/kernl/blob/main/src/kernl/implementations/linear_layer.py
# and https://github.com/openai/triton/blob/master/python/triton/ops/matmul.py
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

from flash_attn.ops.triton.k_activations import (
    gelu,
    gelu_approx,
    gelu_approx_grad,
    gelu_grad,
    squared_relu,
    squared_relu_grad,
)

# CREDITS: Initially inspired by the Triton tutorial on matrix multiplications


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
                                "SPLIT_K": 1,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
                    # split_k not used
                    # for split_k in [2, 4, 8, 16]:
                    #     configs.append(triton.Config(
                    #         {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                    #         num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=5, num_warps=2
        ),
        # good for int8
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=5, num_warps=2
        ),
    ]
    + get_configs_io_bound(),
    key=["CACHE_KEY_M", "CACHE_KEY_N", "CACHE_KEY_K"],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": 10,
    },
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def kernel_fwd(
    C,  # Pointers to matrices
    ACT_INPUT,
    A,
    B,
    bias,
    # Matrix dimensions
    M,
    N,
    K,
    CACHE_KEY_M,
    CACHE_KEY_N,
    CACHE_KEY_K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_cm,
    # stride_cn,  # Assume that stride_cn == 1
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # split k not used, not performant with activation, kept because early_config_prune is expecting it
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    A_ROWMAJOR: tl.constexpr,
    B_COLMAJOR: tl.constexpr,
    BIAS: tl.constexpr,
    SAVE_ACT_INPUT: tl.constexpr,
    ACTIVATION: tl.constexpr,
):

    """
    Kernel for computing Out = activation(A x W + C)
    - Input has shape (M, K)
    - Weight has shape (K, N)
    - Bias has shape (N,)
    - Output has shape (M, N)
    - ActInputs (optional) has shape (M, N)
    'ActInputs' optionally saves the A x W + C intermediate for backward computations
    This kernel will consolidate over K
    """

    pid = tl.program_id(axis=0)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # trick to avoid masking on M and N axis
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    if A_ROWMAJOR:
        A = A + (ram[:, None] * stride_am + rk[None, :])
    else:
        A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    if B_COLMAJOR:
        B = B + (rk[:, None] + rbn[None, :] * stride_bn)
    else:
        B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)

        if A_ROWMAJOR:
            A += BLOCK_K
        else:
            A += BLOCK_K * stride_ak
        if B_COLMAJOR:
            B += BLOCK_K
        else:
            B += BLOCK_K * stride_bk

    # Putting bias after the matmul (instead of before) is faster, idk why
    if BIAS:
        bias = tl.load(bias + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # optional: save the activation inputs
    if SAVE_ACT_INPUT:
        # act_in_ptrs = ACT_INPUT + ram[:, None] * stride_cm + rbn[None, :] * stride_cn
        act_in_ptrs = ACT_INPUT + ram[:, None] * stride_cm + rbn[None, :]
        tl.store(act_in_ptrs, acc)

    # optional: fused activation (while the data is in shared memory)
    if ACTIVATION == "gelu":
        acc = gelu(acc)
    elif ACTIVATION == "gelu_approx":
        acc = gelu_approx(acc)
    elif ACTIVATION == "squared_relu":
        acc = squared_relu(acc)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # write back result
    # C = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    C = C + rm[:, None] * stride_cm + rn[None, :]
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc)


def triton_linear_act(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "id",
    save_act_input: bool = False,
) -> torch.Tensor:
    """
    Compute e = activation(x @ weight.T + bias).
    This wrapper kicks the `kernel_fwd` Triton kernel
    :param x: input tensor
    :param weight: weight matrix
    :param bias: an optional bias tensor
    :param activation: Activation name. Needs to be a Triton kernel.
    :param act_input: an optional tensor to save the activation inputs (for backward)
    :return: result tensor
    """
    # if torch.is_autocast_enabled():
    #     dtype = torch.get_autocast_gpu_dtype()
    #     x, weight, bias = [a.to(dtype=dtype) for a in [x, weight, bias]]

    assert activation in ["id", "gelu", "gelu_approx", "squared_relu"]

    batch_shape, n = x.shape[:-1], x.shape[-1]
    batch_dim = batch_shape.numel()
    x_reshaped = x.reshape(batch_dim, n)

    if x_reshaped.stride(0) > 1 and x_reshaped.stride(1) > 1:
        x_reshaped = x_reshaped.contiguous()
    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None

    assert (
        x.dtype == weight.dtype
    ), f"Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}"
    if bias is not None:
        assert (
            x.dtype == bias.dtype
        ), f"Input and bias must have the same dtype, got {x.dtype} and {bias.dtype}"
    assert (
        x_reshaped.shape[1] == weight.shape[1]
    ), f"Incompatible dimensions: {x_reshaped.shape} - {weight.shape}"

    assert (
        bias is None or bias.shape[0] == weight.shape[0]
    ), "Incompatible dimensions in between weight and bias"

    M, K = x_reshaped.shape
    N, K = weight.shape

    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    act_input = torch.empty_like(output) if save_act_input else None

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa

    kernel_fwd[grid](
        output,
        act_input,
        x_reshaped,
        weight,  # data ptrs
        bias if bias is not None else x,  # auto skip bias if not present
        M,  # shapes
        N,
        K,
        M // 32,  # key for triton cache (limit number of compilations)
        N // 32,
        K // 32,
        stride_cm=output.stride(0),  # strides
        # stride_cn=output.stride(1),
        stride_am=x_reshaped.stride(0),
        stride_ak=x_reshaped.stride(1),
        stride_bk=weight.stride(1),
        stride_bn=weight.stride(0),
        BIAS=bias is not None,  # optional fused bias
        SAVE_ACT_INPUT=save_act_input,  # optional save activation inputs
        ACTIVATION=activation,  # optional fused activation
        A_ROWMAJOR=x_reshaped.stride(1) == 1,
        B_COLMAJOR=weight.stride(1) == 1,
        GROUP_M=8,  # speed optimization: group the programs
    )

    if not save_act_input:
        return output.reshape(*batch_shape, output.shape[-1])
    else:
        return (
            output.reshape(*batch_shape, output.shape[-1]),
            act_input.reshape(*batch_shape, act_input.shape[-1]),
        )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=5, num_warps=2
        ),
        # good for int8
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=5, num_warps=2
        ),
    ]
    + get_configs_io_bound(),
    key=["CACHE_KEY_M", "CACHE_KEY_N", "CACHE_KEY_K"],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": 10,
    },
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def kernel_bwd(
    C,  # Pointers to matrices
    ACT_INPUT,
    A,
    B,
    # Matrix dimensions
    M,
    N,
    K,
    CACHE_KEY_M,
    CACHE_KEY_N,
    CACHE_KEY_K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_cm,
    # stride_cn,  # Assume that stride_cn == 1
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # split k not used, not performant with activation, kept because early_config_prune is expecting it
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):

    """
    Kernel for computing Out = activation(A x W + C)
    - Input has shape (M, K)
    - Weight has shape (K, N)
    - Output has shape (M, N)
    - ActInputs (optional) has shape (M, N)
    'ActInputs' optionally saves the A x W + C intermediate for backward computations
    This kernel will consolidate over K
    """

    pid = tl.program_id(axis=0)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # trick to avoid masking on M and N axis
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)

        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # optional: fused activation (while the data is in shared memory)
    if ACTIVATION != "id":
        act_in_ptrs = ACT_INPUT + ram[:, None] * stride_cm + rbn[None, :]
        act_input = tl.load(act_in_ptrs).to(acc.dtype)
    if ACTIVATION == "gelu":
        acc *= gelu_grad(act_input)
    elif ACTIVATION == "gelu_approx":
        acc *= gelu_approx_grad(act_input)
    elif ACTIVATION == "squared_relu":
        acc *= squared_relu_grad(act_input)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # write back result
    C = C + rm[:, None] * stride_cm + rn[None, :]
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)


def triton_dgrad_act(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    activation: str = "id",
    act_input: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute e = activation(grad_output @ weight + bias).
    This wrapper kicks the `kernel_fwd` Triton kernel
    :param grad_output: input tensor
    :param weight: weight matrix
    :param activation: Activation name. Needs to be a Triton kernel.
    :param act_input: an optional tensor to save the activation inputs (for backward)
    :return: result tensor
    """
    assert activation in ["id", "gelu", "gelu_approx", "squared_relu"]

    batch_shape, n = grad_output.shape[:-1], grad_output.shape[-1]
    batch_dim = batch_shape.numel()
    grad_output_reshaped = grad_output.reshape(batch_dim, n)

    if grad_output_reshaped.stride(0) > 1 and grad_output_reshaped.stride(1) > 1:
        grad_output_reshaped = grad_output_reshaped.contiguous()
    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()

    assert (
        grad_output.dtype == weight.dtype
    ), f"grad_output and weight must have the same dtype, got {grad_output.dtype} and {weight.dtype}"
    assert (
        grad_output_reshaped.shape[1] == weight.shape[0]
    ), f"Incompatible dimensions: {grad_output_reshaped.shape} - {weight.shape}"
    if activation != "id":
        assert act_input is not None, f"act_input is required for activation {activation}"

    # M, N, K in bwd are different from M, N, K in fwd
    M, K = grad_output_reshaped.shape
    K, N = weight.shape

    grad_input = torch.empty((M, N), device=grad_output.device, dtype=grad_output.dtype)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa

    kernel_bwd[grid](
        grad_input,
        act_input,
        grad_output_reshaped,
        weight,  # data ptrs
        M,  # shapes
        N,
        K,
        M // 32,  # key for triton cache (limit number of compilations)
        N // 32,
        K // 32,
        stride_cm=grad_input.stride(0),  # strides
        # stride_cn=grad_input.stride(1),
        stride_am=grad_output_reshaped.stride(0),
        stride_ak=grad_output_reshaped.stride(1),
        stride_bk=weight.stride(0),
        stride_bn=weight.stride(1),
        ACTIVATION=activation,  # optional fused activation
        GROUP_M=8,  # speed optimization: group the programs
    )

    return grad_input.reshape(*batch_shape, grad_input.shape[-1])
