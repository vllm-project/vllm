import math

import pytest
import torch

from vllm.model_executor.layers.lora import sgmv_triton as sgmv

MAX_TEST_POWER = 6
SEED = 42


def assert_close(a, b, dtype, tl_dot=False):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3) if not tl_dot else (1e-2, 7e-2),
        torch.bfloat16: (3e-2, 2e-2) if not tl_dot else (3e-2, 1e-1),
        torch.float32: (2e-3, 3e-4) if not tl_dot else (1e-2, 7e-2),
    }[dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def setup_(S, R, H, dtype, repeats_per_lora=1):
    S = math.ceil(S / repeats_per_lora) * repeats_per_lora
    num_unique = S // repeats_per_lora
    if R is None:
        ranks = torch.randint(0, MAX_TEST_POWER, (S, ), device='cuda')
        ranks = 2**ranks  # random powers of 2 between [1, MAX_TEST_POWER]
        R = 2**(MAX_TEST_POWER - 1)
    else:
        ranks = torch.full((S, ), R, device='cuda', dtype=torch.int32)
    weights = torch.randn((S * R, H), device='cuda', dtype=dtype)
    w_locs = torch.randint(0,
                           weights.shape[0], (ranks.sum().item(), ),
                           device='cuda')
    w_start = torch.cat([
        torch.tensor([
            0,
        ], device='cuda', dtype=torch.int32),
        ranks.cumsum(dim=-1)[:-1]
    ])
    indices = torch.arange(num_unique, device='cuda')
    repeats = torch.full((num_unique, ),
                         repeats_per_lora,
                         device='cuda',
                         dtype=torch.int32)
    repeats = torch.cat([
        torch.zeros((1, ), device='cuda', dtype=torch.int32),
        repeats.cumsum(dim=-1)
    ])
    return (weights, w_start, ranks, w_locs, indices, repeats, num_unique, R,
            dtype)


@pytest.mark.parametrize("S", [16 * 2**i for i in range(3, 4)] + [4096])
@pytest.mark.parametrize("R", [2**r for r in range(MAX_TEST_POWER)])
@pytest.mark.parametrize("H", [64, 4096, 7491])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("repeats_per_lora", [1, 16])
@pytest.mark.parametrize("seed", [SEED])
@torch.inference_mode()
def test_correct(S, R, H, dtype, repeats_per_lora, seed):
    torch.manual_seed(seed)
    weights, w_start, ranks, w_locs, indices, repeats, num_unique, R, dtype = (
        setup_(S, R, H, dtype, repeats_per_lora))

    buffer = torch.randn((S, R), device='cuda', dtype=torch.float32)
    out_col_offset = 128
    out = torch.randn((S, H + out_col_offset), device='cuda', dtype=dtype)
    ref_outs = []
    for ui in range(num_unique):
        idx = indices[ui]
        w_rows = w_locs[w_start[idx]:w_start[idx] + ranks[idx]]
        w = weights[w_rows].contiguous()
        inp = buffer[repeats[ui]:repeats[ui + 1], :ranks[idx]].contiguous()
        ref_out = inp.to(dtype=torch.float32) @ w.to(dtype=torch.float32)
        ref_outs.append(ref_out)

    ref_out = torch.cat(ref_outs, dim=0)
    # doing this apparently leads to incorrect results in the first row
    # + out[:, out_col_offset:]
    ref_out += out[:, out_col_offset:].to(dtype=torch.float32)
    # but this does not (likely depends on torch version)

    # run the autotuner, add to a tmp output
    sgmv.sgmv_expand(buffer,
                     weights,
                     torch.rand((S, H + out_col_offset),
                                device='cuda',
                                dtype=dtype),
                     w_start,
                     ranks,
                     w_locs,
                     indices,
                     repeats,
                     out_col_offset=out_col_offset)

    sgmv.sgmv_expand(buffer,
                     weights,
                     out,
                     w_start,
                     ranks,
                     w_locs,
                     indices,
                     repeats,
                     out_col_offset=out_col_offset)

    # diff = (ref_out - out[:, out_col_offset:].to(dtype=torch.float32)).abs()
    # print(f'max diff {diff.max():0.5f}, mean {diff.mean():0.5f}')
    # triton.language.dot, which is used for improved speed when
    # rank and repeats are >= 16
    # gives larger differences from torch
    assert_close(ref_out,
                 out[:, out_col_offset:].to(dtype=torch.float32),
                 dtype=dtype,
                 tl_dot=repeats_per_lora >= 9)

    x = torch.rand((S, H), device='cuda', dtype=dtype)
    out = torch.zeros((S, R), device='cuda', dtype=torch.float32)
    ref_outs = []
    for ui in range(num_unique):
        idx = indices[ui]
        w_rows = w_locs[w_start[idx]:w_start[idx] + ranks[idx]]
        w = weights[w_rows].contiguous()
        inp = x[repeats[ui]:repeats[ui + 1]].contiguous()
        ref_out = inp.to(dtype=torch.float32) @ w.to(dtype=torch.float32).T
        ref_out = torch.cat([
            ref_out,
            torch.zeros((ref_out.shape[0], R - ref_out.shape[-1]),
                        dtype=ref_out.dtype,
                        device='cuda')
        ],
                            dim=-1)
        ref_outs.append(ref_out)

    ref_out = torch.cat(ref_outs, dim=0)
    ref_out += out

    # run the autotuner, add to a tmp output
    sgmv.sgmv_shrink(x, weights, torch.rand_like(out), w_start, ranks, w_locs,
                     indices, repeats, R)

    sgmv.sgmv_shrink(x, weights, out, w_start, ranks, w_locs, indices, repeats,
                     R)

    # diff = (ref_out - out).abs()
    # print(f'max diff {diff.max():0.5f}, mean {diff.mean():0.5f}')
    assert_close(ref_out, out, dtype=dtype, tl_dot=repeats_per_lora >= 9)
