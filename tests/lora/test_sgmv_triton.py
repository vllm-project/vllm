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
        ranks = torch.randint(3, MAX_TEST_POWER, (S, ), device='cuda')
        ranks = 2**ranks  # random powers of 2 between [8, MAX_TEST_POWER]
        R = 2**(MAX_TEST_POWER - 1)
    else:
        ranks = torch.full((S, ), R, device='cuda', dtype=torch.int32)
    weights = torch.randn((num_unique, 1, H, R), device='cuda', dtype=dtype)
    indices = torch.randint(0, num_unique, (num_unique, ), device='cuda')
    repeats = torch.full((num_unique, ),
                         repeats_per_lora,
                         device='cuda',
                         dtype=torch.int32)
    repeats = torch.cat([
        torch.zeros((1, ), device='cuda', dtype=torch.int32),
        repeats.cumsum(dim=-1)
    ])
    return (weights, ranks, indices, repeats, num_unique, R, dtype)


@pytest.mark.parametrize("S", [16 * 2**i for i in range(3, 4)] + [4096])
@pytest.mark.parametrize("R", [2**r for r in range(MAX_TEST_POWER)])
@pytest.mark.parametrize("H", [64, 4096, 7491])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("repeats_per_lora", [1, 16])
@pytest.mark.parametrize("seed", [SEED])
@torch.inference_mode()
def test_correct(S, R, H, dtype, repeats_per_lora, seed):
    torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)
    torch.manual_seed(seed)
    weights, ranks, indices, repeats, num_unique, R, dtype = (setup_(
        S, R, H, dtype, repeats_per_lora))

    buffer = torch.randn((S, R), device='cuda', dtype=torch.float32)
    out_col_offset = 77
    out = torch.randn((S, H + out_col_offset), device='cuda', dtype=dtype)
    ref_outs = []
    for ui in range(num_unique):
        idx = indices[ui]
        w = weights[idx, 0, :, :ranks[idx]].T.contiguous()
        inp = buffer[repeats[ui]:repeats[ui + 1], :ranks[idx]].contiguous()
        ref_out = inp.to(dtype=torch.float32) @ w.to(dtype=torch.float32)
        ref_outs.append(ref_out)

    ref_out = torch.cat(ref_outs, dim=0)
    # doing this apparently leads to incorrect results in the first row
    # + out[:, out_col_offset:]
    ref_out = (ref_out +
               out[:, out_col_offset:].to(dtype=torch.float32)).to(dtype=dtype)
    # but this does not (likely depends on torch version)

    sgmv.sgmv_expand(buffer,
                     weights,
                     out,
                     ranks,
                     indices,
                     repeats,
                     repeats_per_lora,
                     out_col_offset=out_col_offset)

    # diff = (ref_out - out[:, out_col_offset:]).abs()
    # print(f'max diff {diff.max():0.5f}, mean {diff.mean():0.5f}')
    # triton.language.dot, which is used for improved speed when
    # rank and repeats are >= 16
    # gives larger differences from torch
    assert_close(ref_out,
                 out[:, out_col_offset:],
                 dtype=dtype,
                 tl_dot=repeats_per_lora >= 9)

    weights = weights.permute(0, 1, 3, 2).contiguous()
    x = torch.rand((S, H), device='cuda', dtype=dtype)
    out = torch.zeros((S, R), device='cuda', dtype=torch.float32)
    ref_outs = []
    for ui in range(num_unique):
        idx = indices[ui]
        w = weights[idx, 0, :ranks[idx], :].T.contiguous()
        inp = x[repeats[ui]:repeats[ui + 1]].contiguous()
        ref_out = inp.to(dtype=torch.float32) @ w.to(dtype=torch.float32)
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

    sgmv.sgmv_shrink(x, weights, out, ranks, indices, repeats,
                     repeats_per_lora)

    # diff = (ref_out - out).abs()
    # print(f'max diff {diff.max():0.5f}, mean {diff.mean():0.5f}')
    assert_close(ref_out, out, dtype=dtype, tl_dot=repeats_per_lora >= 9)
    torch.cuda.empty_cache()
