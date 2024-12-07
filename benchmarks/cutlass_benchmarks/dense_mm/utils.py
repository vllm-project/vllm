# Cutlass bench utils
from typing import Iterable, Tuple

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
                             k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn((m, k), device='cuda') * 5
    b = torch.randn((n, k), device='cuda').t() * 5

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

    return a, b


def make_n_rand_tensors(num_tensors: int, dtype: torch.dtype,
                        m: int, n: int, k: int) -> \
                        Tuple[Iterable[torch.Tensor], Iterable[torch.Tensor]]:
    ABs = []
    for _ in range(num_tensors):
        a, b = make_rand_tensors(dtype, m, n, k)
        if a is not None:
            ABs.append(make_rand_tensors(dtype, m, n, k))
    As, Bs = zip(*ABs)
    return list(As), list(Bs)
