# Cutlass bench utils
from typing import Iterable, Tuple

import torch


def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)


def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def make_rand_tensors(dtype: torch.dtype, m: int, n: int,
                      k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn((m, k), device='cuda') * 5
    b = torch.randn((n, k), device='cuda').t() * 5

    if dtype == torch.int8:
        return to_int8(a), to_int8(b)
    if dtype == torch.float8_e4m3fn:
        return to_fp8(a), to_fp8(b)

    raise ValueError("unsupported dtype")

def make_n_rand_tensors(num_tensors: int, dtype: torch.dtype,
                        m: int, n: int, k: int) -> \
                        Tuple[Iterable[torch.Tensor], Iterable[torch.Tensor]]:
    ABs = []
    for _ in range(num_tensors):
        ABs.append(make_rand_tensors(dtype, m, n, k))
    As, Bs = zip(*ABs)
    return list(As), list(Bs)
