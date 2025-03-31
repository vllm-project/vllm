# SPDX-License-Identifier: Apache-2.0
import itertools
from functools import partial

import pytest
import torch
import torch_xla.core.xla_model as xm


def create_tensors(T, D, L, N, dtype=torch.bfloat16, device='xla'):
    """
    Inputs: (All integers)
        T: Total number of tokens
        D: Input dim
        L: LoRA Dim
        N: N LoRAs
    
    Outputs:
        inputs:     torch.Tensor - shape (T, D)
        loras:      torch.Tensor - shape (N, 1, L, D)
        idxs:       torch.IntTensor - shape (T, ) - all values must be in [0, N)
    """

    inputs = torch.randn((T, D), dtype=dtype, device=device)
    loras = torch.randn((N, L, D), dtype=dtype, device=device)
    idxs = torch.randint(0, N, (T, ), dtype=torch.int32, device=device)

    return inputs, loras, idxs


SEQ_LENS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 131072]
HIDDEN_DIM = [256, 1024, 4096, 8192, 14336, 28672]
LORA_RANKS = [8, 16, 32, 64, 128, 128]
N_LORAS = [1, 2, 4, 8]


@torch.compile(fullgraph=True, dynamic=False, backend="openxla")
def ref_bgmv(inputs: torch.Tensor, loras: torch.Tensor, idxs: torch.IntTensor):
    return torch.einsum("td,tld->tl", inputs, loras[idxs])


@torch.compile(fullgraph=True, dynamic=False, backend="openxla")
def bgmv_shrink(inputs: torch.Tensor, loras: torch.Tensor,
                idxs: torch.IntTensor):
    return torch.ops.xla.bgmv_shrink(inputs, loras, idxs)


@torch.compile(fullgraph=True, dynamic=False, backend="openxla")
def bgmv_expand(inputs: torch.Tensor, loras: torch.Tensor,
                idxs: torch.IntTensor, enable_laning: bool):
    return torch.ops.xla.bgmv_expand(inputs, loras, idxs, enable_laning)


@torch.compile(fullgraph=True, dynamic=False, backend="openxla")
def shrink_and_expand(inputs: torch.Tensor, loras_a: torch.Tensor,
                      loras_b: torch.Tensor, idxs: torch.IntTensor):
    return bgmv_expand(bgmv_shrink(inputs, loras_a, idxs),
                       loras_b,
                       idxs,
                       enable_laning=True)


@torch.compile(fullgraph=True, dynamic=False, backend="openxla")
def ref_shrink_and_expand(inputs: torch.Tensor, loras_a: torch.Tensor,
                          loras_b: torch.Tensor, idxs: torch.IntTensor):
    return ref_bgmv(ref_bgmv(inputs, loras_a, idxs), loras_b, idxs)


def run_and_wait_torch(func, *args):
    out = func(*args)
    xm.mark_step()
    xm.wait_device_ops()
    return out


@pytest.mark.parametrize(
    "T,D,L,N", itertools.product(SEQ_LENS, HIDDEN_DIM, LORA_RANKS, N_LORAS))
@pytest.mark.parametrize("func", [bgmv_shrink])
def test_bmark_shrink(benchmark, T, D, L, N, func):
    inputs, loras, idxs = create_tensors(T, D, L, N)

    benchmark.pedantic(partial(run_and_wait_torch, func),
                       args=(inputs, loras, idxs),
                       rounds=5,
                       warmup_rounds=5,
                       iterations=10)


@pytest.mark.parametrize(
    "T,D,L,N", itertools.product(SEQ_LENS, LORA_RANKS, HIDDEN_DIM, N_LORAS))
@pytest.mark.parametrize("func", [bgmv_expand])
def test_bmark_expand(benchmark, T, D, L, N, func):
    inputs, loras, idxs = create_tensors(T, D, L, N)

    benchmark.pedantic(partial(run_and_wait_torch, func),
                       args=(inputs, loras, idxs),
                       rounds=5,
                       warmup_rounds=5,
                       iterations=10)


@pytest.mark.parametrize(
    "T,D,L,N", itertools.product(SEQ_LENS, HIDDEN_DIM, LORA_RANKS, N_LORAS))
@pytest.mark.parametrize("func", [shrink_and_expand])
def test_bmark_shrink_and_expand(benchmark, T, D, L, N, func):
    inputs, loras_a, idxs = create_tensors(T, D, L, N)
    _, loras_b, _ = create_tensors(T, L, D, N)

    benchmark.pedantic(partial(run_and_wait_torch, func),
                       args=(inputs, loras_a, loras_b, idxs),
                       rounds=5,
                       warmup_rounds=5,
                       iterations=10)
