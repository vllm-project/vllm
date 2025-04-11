# SPDX-License-Identifier: Apache-2.0

# Cutlass bench utils
from collections.abc import Iterable

import torch

import vllm._custom_ops as ops


def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)


def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def to_bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.bfloat16)


def to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.float16)


def make_rand_tensors(dtype: torch.dtype, m: int, n: int,
                      k: int) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn((m, k), device='cuda') * 5
    b = torch.randn((n, k), device='cuda').t() * 5

    if dtype == torch.int8:
        return to_int8(a), to_int8(b)
    if dtype == torch.float8_e4m3fn:
        return to_fp8(a), to_fp8(b)

    raise ValueError("unsupported dtype")


def prune_to_2_4(tensor):
    # Reshape tensor to [N, 4] where N is number of groups of 4
    original_shape = tensor.shape
    reshaped = tensor.reshape(-1, 4)

    # Get indices of top 2 absolute values in each group of 4
    _, indices = torch.topk(torch.abs(reshaped), k=2, dim=1)

    # Create binary mask
    mask = torch.zeros_like(reshaped)
    mask.scatter_(dim=1,
                  index=indices,
                  src=torch.ones_like(indices, dtype=mask.dtype))

    # Apply mask and reshape back
    pruned = reshaped * mask

    # Turn all -0.0 to 0.0
    pruned[pruned == -0.0] = 0.0

    return pruned.reshape(original_shape)


def make_rand_sparse_tensors(dtype: torch.dtype, m: int, n: int,
                             k: int) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn((m, k), device='cuda') * 5
    b = torch.randn((n, k), device='cuda').t() * 5

    b = prune_to_2_4(b.t()).t()

    if dtype == torch.int8:
        a, b = to_int8(a), to_int8(b)
    elif dtype == torch.float8_e4m3fn:
        a, b = to_fp8(a), to_fp8(b)
    elif dtype == torch.float16:
        a, b = to_fp16(a), to_fp16(b)
    elif dtype == torch.bfloat16:
        a, b = to_bf16(a), to_bf16(b)
    else:
        raise ValueError("unsupported dtype")

    b_compressed, e = ops.cutlass_sparse_compress(b.t())

    # Compressed B, Metadata, Original A, B
    return b_compressed, e, a, b


def make_n_rand_sparse_tensors(num_tensors: int, dtype: torch.dtype,
                        m: int, n: int, k: int) -> \
                        tuple[Iterable[torch.Tensor], Iterable[torch.Tensor]]:
    ABs = []
    for _ in range(num_tensors):
        b_comp, e, a, b = make_rand_sparse_tensors(dtype, m, n, k)
        if b_comp is not None:
            ABs.append(make_rand_sparse_tensors(dtype, m, n, k))
    BComps, Es, As, Bs = zip(*ABs)
    return list(BComps), list(Es), list(As), list(Bs)
