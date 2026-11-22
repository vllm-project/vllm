# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Triton tile loaders in `vllm/v1/attention/ops/kitty/tiles.py`."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.kitty.store_seal import (
    _kitty_seal_k_kernel,
    _kitty_seal_v_kernel,
)
from vllm.v1.attention.ops.kitty.tiles import (
    KITTY_PAGE_SIZE,
    KITTY_RESIDUAL_ROWS,
    kitty_dequant_k_page,
    kitty_dequant_v_page,
    kitty_load_k_tile,
    kitty_load_v_tile,
    kitty_state_content_bytes,
)

D = 128
PAGE = KITTY_PAGE_SIZE

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(80),
    reason="Kitty kernels need CUDA with compute capability >= 8.0",
)

# Byte offsets of the page segments for D = 128, keyed by DB. K low plane and
# V data always start at 0 and 4096, and the K high plane at 8192.
OFFSETS = {
    0: {"k_scale_min": 8192, "v_scale_min": 8704, "k_rank": 9216, "total": 9344},
    4: {"k_scale_min": 8320, "v_scale_min": 8832, "k_rank": 9344, "total": 9472},
    32: {"k_scale_min": 9216, "v_scale_min": 9728, "k_rank": 10240, "total": 10368},
}
K_LOW, V_DATA, K_HIGH = 0, 4096, 8192


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


def dequant(page: torch.Tensor, kind: str, d_boost: int) -> torch.Tensor:
    out = torch.empty(PAGE, D, dtype=torch.float32, device="cuda")
    _dequant_page_kernel[(1,)](out, page.cuda(), IS_K=kind == "k", DB=d_boost)
    return out.cpu()


def empty_page(d_boost: int) -> torch.Tensor:
    """All-zero page whose rank table marks every K channel as not boosted."""
    page = torch.zeros(OFFSETS[d_boost]["total"], dtype=torch.uint8)
    page[OFFSETS[d_boost]["k_rank"] :] = d_boost
    return page


def put_fp16(page: torch.Tensor, offset: int, *values: float) -> None:
    data = torch.tensor(values, dtype=torch.float16).view(torch.uint8)
    page[offset : offset + data.numel()] = data


# kitty_dequant_k_page / kitty_dequant_v_page


@pytest.mark.parametrize("d_boost", [0, 32])
def test_dequant_k_low_plane_bit_order(d_boost):
    """K codes are packed along tokens, first token in the lowest bits, and
    each channel decodes as code * scale + min."""
    page = empty_page(d_boost)
    channel = 3
    page[K_LOW + 32 * channel + 0] = 0b11_10_01_00  # tokens 0-3
    page[K_LOW + 32 * channel + 1] = 0b00_00_00_11  # tokens 4-7
    put_fp16(page, OFFSETS[d_boost]["k_scale_min"] + 4 * channel, 2.0, 0.5)

    expected = torch.zeros(PAGE, D)
    expected[:, channel] = 0.5
    expected[:5, channel] = torch.tensor([0.5, 2.5, 4.5, 6.5, 6.5])
    torch.testing.assert_close(dequant(page, "k", d_boost), expected, rtol=0, atol=0)


def test_dequant_k_high_plane_row_is_the_channels_rank():
    """A boosted channel takes its 2 high bits from high-plane row rank[c];
    a channel with rank == DB ignores the high plane."""
    d_boost = 4
    offsets = OFFSETS[d_boost]
    page = empty_page(d_boost)
    page[offsets["k_rank"] + 7] = 2
    page[offsets["k_rank"] + 9] = 0
    page[K_HIGH + 32 * 0] = 0b01_01_01_01
    page[K_HIGH + 32 * 1] = 0b10_10_10_10  # no channel has rank 1
    page[K_HIGH + 32 * 2] = 0b11_10_01_00
    page[K_HIGH + 32 * 3] = 0b10_10_10_10  # no channel has rank 3
    page[K_LOW + 32 * 7] = 0b00_00_00_01
    page[K_LOW + 32 * 20] = 0b11_11_11_11
    for channel in (7, 9, 20):
        put_fp16(page, offsets["k_scale_min"] + 4 * channel, 1.0, 0.0)

    expected = torch.zeros(PAGE, D)
    expected[:4, 7] = torch.tensor([1.0, 4.0, 8.0, 12.0])
    expected[:4, 9] = 4.0
    expected[:4, 20] = 3.0
    torch.testing.assert_close(dequant(page, "k", d_boost), expected, rtol=0, atol=0)


@pytest.mark.parametrize("d_boost", [0, 32])
def test_dequant_v_bit_order(d_boost):
    """V codes are packed along channels, first channel in the lowest bits,
    and each token decodes as code * scale + min."""
    page = empty_page(d_boost)
    page[V_DATA + 32 * 5 + 0] = 0b11_10_01_00  # token 5, channels 0-3
    page[V_DATA + 32 * 5 + 1] = 0b00_00_00_11  # token 5, channels 4-7
    put_fp16(page, OFFSETS[d_boost]["v_scale_min"] + 4 * 5, 2.0, 0.5)
    put_fp16(page, OFFSETS[d_boost]["v_scale_min"] + 4 * 6, 0.0, -1.0)

    expected = torch.zeros(PAGE, D)
    expected[5] = 0.5
    expected[5, :5] = torch.tensor([0.5, 2.5, 4.5, 6.5, 6.5])
    expected[6] = -1.0
    torch.testing.assert_close(dequant(page, "v", d_boost), expected, rtol=0, atol=0)


def seal_page(kind: str, x: torch.Tensor, d_boost: int, ranks=None) -> torch.Tensor:
    """Seal x [128, D] with the production seal kernel; return the page bytes."""
    content = kitty_state_content_bytes(D, d_boost)
    cache = torch.zeros(1, 1, PAGE, content, dtype=torch.uint8, device="cuda")
    x = x.cuda()[None, None].contiguous()
    jobs = torch.tensor([[0, 1]], device="cuda")  # request 0, block 1
    block_table = torch.tensor([[0, 0]], dtype=torch.int32, device="cuda")
    cache_args = (jobs, block_table, block_table.stride(0), cache, *cache.stride()[:2])
    if kind == "k":
        ranks = ranks.cuda()[None, None].contiguous()
        _kitty_seal_k_kernel[(1, 1)](
            x, *x.stride()[:2], ranks, *ranks.stride()[:2], *cache_args, D=D, DB=d_boost
        )
    else:
        _kitty_seal_v_kernel[(1, 1)](x, *x.stride()[:2], *cache_args, D=D, DB=d_boost)
    return cache[0, 0].flatten()


@pytest.mark.parametrize("d_boost", [0, 25, 32, 128])
def test_dequant_reads_k_pages_written_by_the_seal_kernel(d_boost):
    """Each channel spans exactly 4 (or 16 if boosted) evenly spaced values,
    so sealing and decoding must reproduce K exactly."""
    gen = torch.Generator().manual_seed(0)
    ranks = torch.full((D,), d_boost, dtype=torch.uint8)
    boosted = torch.randperm(D, generator=gen)[:d_boost]
    ranks[boosted] = torch.randperm(d_boost, generator=gen).to(torch.uint8)
    levels = torch.where(ranks < d_boost, 16, 4)
    t = torch.arange(PAGE)[:, None]
    d = torch.arange(D)
    k = ((t + d) % levels + d % 3).half()
    page = seal_page("k", k, d_boost, ranks)
    torch.testing.assert_close(dequant(page, "k", d_boost), k.float(), rtol=0, atol=0)


@pytest.mark.parametrize("d_boost", [0, 25, 32, 128])
def test_dequant_reads_v_pages_written_by_the_seal_kernel(d_boost):
    """Each token spans exactly 4 evenly spaced values, so sealing and
    decoding must reproduce V exactly."""
    t = torch.arange(PAGE)[:, None]
    d = torch.arange(D)
    v = ((t + d) % 4 + t % 3).half()
    page = seal_page("v", v, d_boost)
    torch.testing.assert_close(dequant(page, "v", d_boost), v.float(), rtol=0, atol=0)


# kitty_load_k_tile / kitty_load_v_tile

FROM_PAGE = "page"

# (old_len, b, (s_k, K source), (s_v, V source)). s is the first unsealed
# block at old_len. A source is FROM_PAGE or (first residual row, number of
# positions < old_len); positions past old_len load as 0.
LOAD_TILE_CASES = [
    (100, 0, (1, (0, 100)), (1, (256, 100))),  # partial sink
    (300, 0, (2, (0, 128)), (1, (256, 128))),  # full sink
    (200, 1, (1, (128, 72)), (1, (512, 72))),  # block 1 not full yet
    (256, 1, (2, FROM_PAGE), (1, (512, 128))),  # block 1 full: K sealed, V kept
    (256, 2, (2, (128, 0)), (1, (384, 0))),  # nothing written yet
    (300, 1, (2, FROM_PAGE), (1, (512, 128))),
    (300, 2, (2, (128, 44)), (1, (384, 44))),  # partial block
    (300, 3, (2, (128, 0)), (1, (512, 0))),  # past old_len
    (384, 1, (3, FROM_PAGE), (2, FROM_PAGE)),  # block 2 full: V block 1 sealed
    (384, 2, (3, FROM_PAGE), (2, (384, 128))),
    (520, 2, (4, FROM_PAGE), (3, FROM_PAGE)),
    (520, 3, (4, FROM_PAGE), (3, (512, 128))),
    (520, 4, (4, (128, 8)), (3, (384, 8))),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["k", "v"])
@pytest.mark.parametrize("old_len, b, k_case, v_case", LOAD_TILE_CASES)
def test_load_tile_reads_the_right_source(kind, dtype, old_len, b, k_case, v_case):
    """Every value of the page decodes to 7.0 and the residual is random, so
    the tile shows whether it came from the page or from which residual rows."""
    d_boost = 32
    page = empty_page(d_boost)
    put_fp16(page, OFFSETS[d_boost]["k_scale_min"], *[0.0, 7.0] * D)
    put_fp16(page, OFFSETS[d_boost]["v_scale_min"], *[0.0, 7.0] * PAGE)
    gen = torch.Generator(device="cuda").manual_seed(0)
    residual = torch.randn(
        KITTY_RESIDUAL_ROWS, D, generator=gen, dtype=dtype, device="cuda"
    )
    s, source = k_case if kind == "k" else v_case

    out = torch.empty(PAGE, D, dtype=dtype, device="cuda")
    _load_tile_kernel[(1,)](
        out, page.cuda(), residual, b, old_len, s, IS_K=kind == "k", DB=d_boost
    )

    if source == FROM_PAGE:
        expected = torch.full_like(out, 7.0)
    else:
        row, num_valid = source
        expected = torch.zeros_like(out)
        expected[:num_valid] = residual[row : row + num_valid]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
