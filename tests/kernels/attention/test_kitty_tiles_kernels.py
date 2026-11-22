# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Triton tile loaders in `vllm/v1/attention/ops/kitty/tiles.py`."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.kitty.tiles import (
    K_SINK_ROW,
    K_TAIL_ROW,
    KITTY_PAGE_SIZE,
    KITTY_RESIDUAL_ROWS,
    V_SINK_ROW,
    V_TAIL_ROW,
    kitty_dequant_k_page,
    kitty_dequant_v_page,
    kitty_first_unsealed_k,
    kitty_first_unsealed_v,
    kitty_load_k_tile,
    kitty_load_v_tile,
    kitty_page_bytes,
)

D = 128
PAGE = KITTY_PAGE_SIZE

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(80),
    reason="Kitty kernels need CUDA with compute capability >= 8.0",
)


# Both wrappers write tiles token-major as [128, D].


@triton.jit
def _dequant_page_kernel(out_ptr, page_ptr, IS_K: tl.constexpr, DB: tl.constexpr):
    offs_t = tl.arange(0, 128)
    offs_d = tl.arange(0, 128)
    if IS_K:
        k = kitty_dequant_k_page(page_ptr, offs_t, offs_d, 128, DB)
        tl.store(out_ptr + offs_t[None, :] * 128 + offs_d[:, None], k)
    else:
        v = kitty_dequant_v_page(page_ptr, offs_t, offs_d, 128, DB)
        tl.store(out_ptr + offs_t[:, None] * 128 + offs_d[None, :], v)


@triton.jit
def _load_tile_kernel(
    out_ptr, page_ptr, res_ptr, b, old_len, s, IS_K: tl.constexpr, DB: tl.constexpr
):
    offs_t = tl.arange(0, 128)
    offs_d = tl.arange(0, 128)
    if IS_K:
        k = kitty_load_k_tile(page_ptr, res_ptr, b, old_len, s, offs_t, offs_d, 128, DB)
        tl.store(out_ptr + offs_t[None, :] * 128 + offs_d[:, None], k)
    else:
        v = kitty_load_v_tile(page_ptr, res_ptr, b, old_len, s, offs_t, offs_d, 128, DB)
        tl.store(out_ptr + offs_t[:, None] * 128 + offs_d[None, :], v)


def pack2(codes: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit codes along the last dim, 4 per byte, lowest bits first."""
    c = codes.to(torch.uint8).reshape(*codes.shape[:-1], codes.shape[-1] // 4, 4)
    return c[..., 0] | c[..., 1] << 2 | c[..., 2] << 4 | c[..., 3] << 6


def fp16_bytes(*cols: torch.Tensor) -> torch.Tensor:
    return torch.stack(cols, dim=-1).view(torch.uint8).flatten()


def make_page(d_boost: int):
    """A random packed page and the `[128, D]` fp32 K and V tiles it encodes."""
    boosted = torch.zeros(D, dtype=torch.bool)
    boosted[torch.randperm(D)[:d_boost]] = True
    ranks = torch.full((D,), d_boost)
    ranks[boosted] = torch.randperm(d_boost)

    k_codes = torch.randint(0, 16, (PAGE, D)) & torch.where(boosted, 15, 3)
    v_codes = torch.randint(0, 4, (PAGE, D))
    k_scale, k_min = torch.randn(2, D, dtype=torch.float16)
    v_scale, v_min = torch.randn(2, PAGE, dtype=torch.float16)

    high_plane = torch.zeros(d_boost, PAGE, dtype=torch.long)
    high_plane[ranks[boosted]] = (k_codes >> 2).T[boosted]
    page = torch.cat(
        [
            pack2(k_codes.T & 3).flatten(),
            pack2(v_codes).flatten(),
            pack2(high_plane).flatten(),
            fp16_bytes(k_scale, k_min),
            fp16_bytes(v_scale, v_min),
            ranks.to(torch.uint8),
        ]
    )
    assert page.numel() == kitty_page_bytes(D, d_boost)

    k = k_codes * k_scale.float() + k_min.float()
    v = v_codes * v_scale.float()[:, None] + v_min.float()[:, None]
    return page.cuda(), k.cuda(), v.cuda()


@pytest.mark.parametrize("d_boost", [0, 25, 32, 128])
@pytest.mark.parametrize("kind", ["k", "v"])
def test_dequant_page(kind, d_boost):
    """Exact, since a code times an fp16 scale is exact in fp32."""
    page, k, v = make_page(d_boost)
    out = torch.empty(PAGE, D, dtype=torch.float32, device="cuda")
    _dequant_page_kernel[(1,)](out, page, IS_K=kind == "k", DB=d_boost)
    torch.testing.assert_close(out, k if kind == "k" else v, rtol=0, atol=0)


# kind: (sink row, tail row, tail ring size, first unsealed block)
LAYOUT = {
    "k": (K_SINK_ROW, K_TAIL_ROW, PAGE, kitty_first_unsealed_k),
    "v": (V_SINK_ROW, V_TAIL_ROW, 2 * PAGE, kitty_first_unsealed_v),
}


def expected_tile(kind, page_tile, residual, b, old_len):
    sink_row, tail_row, ring, first_unsealed = LAYOUT[kind]
    if 0 < b < first_unsealed(old_len):
        return page_tile.to(residual.dtype)
    pos = b * PAGE + torch.arange(PAGE, device=residual.device)
    rows = sink_row + pos if b == 0 else tail_row + pos % ring
    tile = residual[rows]
    tile[pos >= old_len] = 0
    return tile


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["k", "v"])
@pytest.mark.parametrize(
    "old_len, b",
    [
        (100, 0),  # partial sink
        (300, 0),  # full sink
        (300, 1),  # K sealed, V kept in the tail as the latest full block
        (300, 2),  # partial block in the tail
        (300, 3),  # past old_len
        (520, 2),  # K and V sealed
        (520, 3),  # K sealed, V in the tail
        (520, 4),  # partial block in the tail
    ],
)
def test_load_tile(kind, dtype, old_len, b):
    d_boost = 32
    page, k, v = make_page(d_boost)
    residual = torch.randn(KITTY_RESIDUAL_ROWS, D, dtype=dtype, device="cuda")
    s = int(LAYOUT[kind][3](old_len))
    out = torch.empty(PAGE, D, dtype=dtype, device="cuda")
    _load_tile_kernel[(1,)](
        out, page, residual, b, old_len, s, IS_K=kind == "k", DB=d_boost
    )
    page_tile = k if kind == "k" else v
    torch.testing.assert_close(
        out, expected_tile(kind, page_tile, residual, b, old_len), rtol=0, atol=0
    )
