# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU tests for the Kitty write path in
`vllm/v1/attention/ops/kitty/store_seal.py`.

Inputs are chosen so that every expected page byte, scale and min is known by
hand. The CPU tests of the planning and gathering helpers and of
`kitty_write_residual` are in `test_kitty_store_seal.py`.
"""

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.kitty.store_seal import (
    _kitty_seal_k_kernel,
    _kitty_seal_v_kernel,
    kitty_seal,
    kitty_write_residual,
    plan_kitty_step,
)
from vllm.v1.attention.ops.kitty.tiles import KITTY_PAGE_SIZE, KITTY_RESIDUAL_ROWS

if not current_platform.is_cuda():
    pytest.skip("Kitty's Triton kernels run on CUDA only", allow_module_level=True)

PAGE = KITTY_PAGE_SIZE
HEAD_SIZE = 128  # the only head size Kitty supports
NUM_HEADS = 2
DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16]
D_BOOSTS = [0, 25, 32, 128]
SENTINEL = 0xAB  # byte that fills packed pages before anything is sealed
FLOOR_SCALE = 1.0132789611816406e-06  # the 1e-6 scale floor rounded to fp16

# Segment i of a packed page (D = 128) is bytes [bounds[i], bounds[i + 1]) of
# SEGMENT_BOUNDS[d_boost]. The padding fills the page up to 128 * C bytes.
SEGMENT_NAMES = [
    "k_low",
    "v_data",
    "k_high",
    "k_scale_min",
    "v_scale_min",
    "k_rank",
    "padding",
]
SEGMENT_BOUNDS = {
    0: [0, 4096, 8192, 8192, 8704, 9216, 9344, 9344],
    1: [0, 4096, 8192, 8224, 8736, 9248, 9376, 9472],
    4: [0, 4096, 8192, 8320, 8832, 9344, 9472, 9472],
    25: [0, 4096, 8192, 8992, 9504, 10016, 10144, 10240],
    32: [0, 4096, 8192, 9216, 9728, 10240, 10368, 10368],
    128: [0, 4096, 8192, 12288, 12800, 13312, 13440, 13440],
}
K_SEGMENTS = ["k_low", "k_high", "k_scale_min", "k_rank"]
V_SEGMENTS = ["v_data", "v_scale_min"]

# Bytes of four 2-bit codes, first code in the lowest bits.
CODES_0123 = 0b11_10_01_00
CODES_3210 = 0b00_01_10_11


def segment(page: torch.Tensor, d_boost: int, name: str) -> torch.Tensor:
    i = SEGMENT_NAMES.index(name)
    start, stop = SEGMENT_BOUNDS[d_boost][i : i + 2]
    return page[start:stop]


def scale_min(page: torch.Tensor, d_boost: int, name: str) -> torch.Tensor:
    """The fp16 (scale, min) pairs of a segment, as float32 [n, 2]."""
    return segment(page, d_boost, name).view(torch.float16).view(-1, 2).float()


def assert_equal(got: torch.Tensor, want) -> None:
    """Exact equality, broadcasting `want` (a scalar, list or tensor)."""
    want = torch.as_tensor(want, dtype=got.dtype, device=got.device)
    torch.testing.assert_close(got, want.expand_as(got), rtol=0, atol=0)


def assert_untouched(page: torch.Tensor, d_boost: int, names: list[str]) -> None:
    for name in names:
        assert (segment(page, d_boost, name) == SENTINEL).all(), name


def empty_cache(num_pages: int, d_boost: int) -> torch.Tensor:
    """Packed cache [num_pages, H, 128, C] filled with SENTINEL."""
    content = SEGMENT_BOUNDS[d_boost][-1] // PAGE
    shape = (num_pages, NUM_HEADS, PAGE, content)
    return torch.full(shape, SENTINEL, dtype=torch.uint8, device=DEVICE)


def page_bytes(cache: torch.Tensor, page_id: int, head: int) -> torch.Tensor:
    """The flat uint8 bytes of one (page, KV head) of the packed cache."""
    return cache[page_id, head].flatten()


def first_channels_boosted(d_boost: int) -> torch.Tensor:
    """Rank table [H, D] boosting channels 0..DB-1, with rank = channel."""
    d = torch.arange(HEAD_SIZE, device=DEVICE)
    ranks = torch.where(d < d_boost, d, d_boost).to(torch.uint8)
    return ranks.expand(NUM_HEADS, -1)


# Launching the seal kernels directly


def launch_seal_kernel(kind, x, jobs, block_table, cache, d_boost, ranks=None):
    """Seal x [J, H, 128, D] into the pages that `jobs` [J, 2] name."""
    x = x.contiguous()
    cache_args = (
        jobs,
        block_table,
        block_table.stride(0),
        cache,
        cache.stride(0),
        cache.stride(1),
    )
    grid = (jobs.shape[0], NUM_HEADS)
    if kind == "k":
        ranks = ranks.contiguous()
        _kitty_seal_k_kernel[grid](
            x,
            x.stride(0),
            x.stride(1),
            ranks,
            ranks.stride(0),
            ranks.stride(1),
            *cache_args,
            D=HEAD_SIZE,
            DB=d_boost,
        )
    else:
        _kitty_seal_v_kernel[grid](
            x, x.stride(0), x.stride(1), *cache_args, D=HEAD_SIZE, DB=d_boost
        )


def seal_one_page(kind, x, d_boost, ranks=None) -> torch.Tensor:
    """Seal x [128, D], the same in every head, into page 0 of a fresh cache
    and return its bytes per head, [H, 128 * C]."""
    cache = empty_cache(1, d_boost)
    jobs = torch.tensor([[0, 1]], device=DEVICE)  # request 0, block 1
    block_table = torch.tensor([[0, 0]], dtype=torch.int32, device=DEVICE)
    x = x.expand(1, NUM_HEADS, -1, -1)
    if ranks is not None:
        ranks = ranks.expand(1, NUM_HEADS, -1)
    launch_seal_kernel(kind, x, jobs, block_table, cache, d_boost, ranks)
    return cache[0].flatten(1)


# _kitty_seal_k_kernel


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d_boost", D_BOOSTS)
def test_seal_k_page(dtype, d_boost):
    """Boosted channels (0..DB-1, rank = channel) hold t % 16 at token t and
    the others t % 4, so every channel has scale 1, min 0 and codes equal to
    its values. The V segments and the padding stay untouched."""
    t = torch.arange(PAGE, device=DEVICE)[:, None]
    boosted = torch.arange(HEAD_SIZE, device=DEVICE) < d_boost
    x = torch.where(boosted, t % 16, t % 4).to(dtype)
    pages = seal_one_page("k", x, d_boost, first_channels_boosted(d_boost))

    # High bits of tokens 4i..4i+3 are all i % 4.
    high_row = [0b00_00_00_00, 0b01_01_01_01, 0b10_10_10_10, 0b11_11_11_11] * 8
    ranks = list(range(d_boost)) + [d_boost] * (HEAD_SIZE - d_boost)
    for page in pages:
        assert_equal(segment(page, d_boost, "k_low"), CODES_0123)
        assert_equal(segment(page, d_boost, "k_high").view(d_boost, 32), high_row)
        assert_equal(scale_min(page, d_boost, "k_scale_min"), [1.0, 0.0])
        assert_equal(segment(page, d_boost, "k_rank"), ranks)
        assert_untouched(page, d_boost, [*V_SEGMENTS, "padding"])


def test_seal_k_high_plane_row_is_the_channels_rank():
    """Boosted channel c is stored in high-plane row rank[c], not row c.
    Channels 0-3 hold 0 at token 0, 15 at token 1 and 4 * c (high bits c) at
    every later token."""
    d_boost = 4
    x = torch.zeros(PAGE, HEAD_SIZE, device=DEVICE)
    x[1, :4] = 15
    x[2:, :4] = torch.tensor([0.0, 4.0, 8.0, 12.0], device=DEVICE)
    ranks = torch.full((HEAD_SIZE,), d_boost, dtype=torch.uint8, device=DEVICE)
    ranks[:4] = torch.tensor([2, 0, 3, 1])
    pages = seal_one_page("k", x.half(), d_boost, ranks)

    # Byte 0 packs the high bits 0, 3, c, c of tokens 0-3.
    high_rows = [
        [0b01_01_11_00] + [0b01_01_01_01] * 31,  # rank 0: channel 1
        [0b11_11_11_00] + [0b11_11_11_11] * 31,  # rank 1: channel 3
        [0b00_00_11_00] + [0b00_00_00_00] * 31,  # rank 2: channel 0
        [0b10_10_11_00] + [0b10_10_10_10] * 31,  # rank 3: channel 2
    ]
    for page in pages:
        assert_equal(segment(page, d_boost, "k_high").view(d_boost, 32), high_rows)


def test_seal_k_boosted_channel_has_16_levels():
    """Channels 0 (boosted) and 1 (not) both hold 0, 0.5, 1.5, 7, 14.5, 15 at
    tokens 0-5 and 0 after. Channel 0 gets scale 1 and codes 0, 1, 2, 7, 15,
    15; channel 1 gets scale 5 and codes 0, 0, 0, 1, 3, 3."""
    d_boost = 1
    x = torch.zeros(PAGE, HEAD_SIZE, device=DEVICE)
    x[:6, :2] = torch.tensor([0, 0.5, 1.5, 7, 14.5, 15], device=DEVICE)[:, None]
    ranks = torch.ones(HEAD_SIZE, dtype=torch.uint8, device=DEVICE)
    ranks[0] = 0
    pages = seal_one_page("k", x.half(), d_boost, ranks)

    for page in pages:
        low = segment(page, d_boost, "k_low").view(HEAD_SIZE, 32)
        assert_equal(low[0], [0b11_10_01_00, 0b00_00_11_11] + [0] * 30)
        assert_equal(low[1], [0b01_00_00_00, 0b00_00_11_11] + [0] * 30)
        high = segment(page, d_boost, "k_high")
        assert_equal(high, [0b01_00_00_00, 0b00_00_11_11] + [0] * 30)
        sm = scale_min(page, d_boost, "k_scale_min")
        assert_equal(sm[:2], [[1.0, 0.0], [5.0, 0.0]])


# _kitty_seal_v_kernel


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d_boost", D_BOOSTS)
def test_seal_v_page(dtype, d_boost):
    """Token t holds t + d % 4 in channel d, so every token has scale 1, min t
    and codes d % 4. The K segments and the padding stay untouched."""
    t = torch.arange(PAGE, device=DEVICE)[:, None]
    d = torch.arange(HEAD_SIZE, device=DEVICE)
    pages = seal_one_page("v", (t + d % 4).to(dtype), d_boost)

    mins = torch.arange(PAGE, dtype=torch.float32, device=DEVICE)
    for page in pages:
        assert_equal(segment(page, d_boost, "v_data"), CODES_0123)
        sm = scale_min(page, d_boost, "v_scale_min")
        assert_equal(sm, torch.stack([torch.ones_like(mins), mins], dim=1))
        assert_untouched(page, d_boost, [*K_SEGMENTS, "padding"])


# Both seal kernels

# (values, stored scale, stored min, packed codes of the values)
QUANT_CASES = [
    pytest.param(
        [-1, -0.75, -0.5, 0, 0.5, 1.4, 1.5, 2],
        1.0,
        -1.0,
        [0b01_01_00_00, 0b11_11_10_10],  # codes 0, 0, 1, 1, 2, 2, 3, 3
        id="half-rounds-up",
    ),
    pytest.param(
        [0, 0.25, 0.5, 0.75, 1],
        0.333251953125,  # 1/3 rounded to fp16
        0.0,
        [0b10_10_01_00, 0b00_00_00_11],  # codes 0, 1, 2, 2, 3
        id="fp16-scale",
    ),
    pytest.param([2.5], FLOOR_SCALE, 2.5, [], id="constant"),
]


@pytest.mark.parametrize("kind", ["k", "v"])
@pytest.mark.parametrize("values, scale, minimum, packed", QUANT_CASES)
def test_seal_quantizes_to_nearest_code(kind, values, scale, minimum, packed):
    """K channel 0 (not boosted) or V token 0 holds `values`, padded with
    `values[0]`. Every other channel or token is the constant `values[0]`,
    so it gets the scale floor and codes 0."""
    x = torch.full((PAGE, HEAD_SIZE), values[0], dtype=torch.half, device=DEVICE)
    x[0, : len(values)] = torch.tensor(values, device=DEVICE)
    if kind == "k":
        x = x.T
    pages = seal_one_page(kind, x, 0, first_channels_boosted(0))

    codes_name, sm_name = ("k_low", "k_scale_min") if kind == "k" else V_SEGMENTS
    for page in pages:
        # K channel rows or V token rows of 32 bytes.
        rows = segment(page, 0, codes_name).view(128, 32)
        assert_equal(rows[0], packed + [0] * (32 - len(packed)))
        assert_equal(rows[1:], 0)
        sm = scale_min(page, 0, sm_name)
        assert_equal(sm[0], [scale, minimum])
        assert_equal(sm[1:], [FLOOR_SCALE, values[0]])


@pytest.mark.parametrize("kind", ["k", "v"])
def test_seal_kernel_writes_the_page_in_the_block_table(kind):
    """Job (request, block) seals page `block_table[request, block]` in each
    head. Job j holds a constant in head h, which becomes the stored min, and
    boosts 4 channels of its own."""
    d_boost = 4
    block_table = torch.tensor([[0, 3, 5], [0, 1, 4]], dtype=torch.int32, device=DEVICE)
    jobs = torch.tensor([[0, 2], [1, 1]], device=DEVICE)
    # (page, head): (job's value, first of its boosted channels)
    expected = {
        (5, 0): (1.0, 0),
        (5, 1): (2.0, 4),
        (1, 0): (11.0, 8),
        (1, 1): (12.0, 12),
    }
    x = torch.tensor([[1.0, 2.0], [11.0, 12.0]], device=DEVICE)  # [job, head]
    x = x[:, :, None, None].expand(-1, -1, PAGE, HEAD_SIZE).half()
    ranks = torch.full((2, NUM_HEADS, HEAD_SIZE), d_boost, dtype=torch.uint8)
    for j, h, c in [(0, 0, 0), (0, 1, 4), (1, 0, 8), (1, 1, 12)]:
        ranks[j, h, c : c + 4] = torch.arange(4)
    cache = empty_cache(6, d_boost)
    launch_seal_kernel(kind, x, jobs, block_table, cache, d_boost, ranks.to(DEVICE))

    for page_id in range(6):
        for h in range(NUM_HEADS):
            page = page_bytes(cache, page_id, h)
            if (page_id, h) not in expected:
                assert (page == SENTINEL).all()
                continue
            value, first = expected[page_id, h]
            if kind == "k":
                sm = scale_min(page, d_boost, "k_scale_min")
                want_ranks = [d_boost] * HEAD_SIZE
                want_ranks[first : first + 4] = [0, 1, 2, 3]
                assert_equal(segment(page, d_boost, "k_rank"), want_ranks)
            else:
                sm = scale_min(page, d_boost, "v_scale_min")
            assert_equal(sm, [FLOOR_SCALE, value])


# kitty_seal then kitty_write_residual, over whole steps

# Request r's block b is page BLOCK_TABLE[r][b]; its residual is block
# RESIDUAL_BLOCK_IDS[r].
BLOCK_TABLE = [[7, 3, 12, 0, 9], [14, 1, 5, 11, 2], [4, 10, 8, 13, 6]]
RESIDUAL_BLOCK_IDS = [2, 0, 1]

# steps: query lengths of requests 0-2 at each step. Per request, the K and V
# blocks sealed at the end and the residual contents as (first row, first
# position, count) runs.
WRITE_PATH_CASES = {
    # Every block is sealed straight from the step's K/V.
    "prefill": dict(
        steps=[[600, 300, 100]],
        k_sealed=[[1, 2, 3], [1], []],
        v_sealed=[[1, 2], [], []],
        k_residual=[
            [(0, 0, 128), (128, 512, 88)],
            [(0, 0, 128), (128, 256, 44)],
            [(0, 0, 100)],
        ],
        v_residual=[
            [(256, 0, 128), (512, 384, 128), (384, 512, 88)],
            [(256, 0, 128), (512, 128, 128), (384, 256, 44)],
            [(256, 0, 100)],
        ],
    ),
    # Request 0 prefills then decodes, request 1 prefills in chunks that
    # straddle page boundaries, request 2 fills pages exactly at step ends.
    # Later blocks are sealed from tokens read back from the residual, so a
    # wrong ring row or writing the residual before sealing corrupts pages.
    "steps": dict(
        steps=[[600, 100, 255], [1, 100, 1], [1, 100, 128], [1, 100, 129]],
        k_sealed=[[1, 2, 3], [1, 2], [1, 2, 3]],
        v_sealed=[[1, 2], [1], [1, 2]],
        k_residual=[
            [(0, 0, 128), (128, 512, 91)],
            [(0, 0, 128), (128, 384, 16)],
            [(0, 0, 128), (128, 512, 1)],
        ],
        v_residual=[
            [(256, 0, 128), (512, 384, 128), (384, 512, 91)],
            [(256, 0, 128), (384, 256, 144)],
            [(256, 0, 128), (512, 384, 128), (384, 512, 1)],
        ],
    ),
}


def boost_weight(d_boost: int) -> torch.Tensor:
    """W [D]: 5 for the last d_boost channels, 1 for the others."""
    d = torch.arange(HEAD_SIZE, device=DEVICE)
    return torch.where(d >= HEAD_SIZE - d_boost, 5, 1)


def request_kv(length: int, req: int, d_boost: int):
    """K and V [length, H, D] of request `req`; head 1 is head 0 negated.

    At position p = 128 * b + t, head 0 of K channel d is
    8 * req + b + d / 32 + (t % 4) * w[d]. So the last d_boost channels have
    the largest mean |K|, ranked from channel 127 down, and every channel has
    scale 1 and min 8 * req + b + d / 32. Head 0 of V is
    640 * req + p + d % 4, so every token has scale 1 and min 640 * req + p.
    """
    p = torch.arange(length, device=DEVICE)[:, None]
    d = torch.arange(HEAD_SIZE, device=DEVICE)
    k = 8 * req + p // PAGE + d / 32 + (p % 4) * boost_weight(d_boost)
    v = 640 * req + p + d % 4
    return torch.stack([k, -k], dim=1).half(), torch.stack([v, -v], dim=1).half()


def run_steps(steps: list[list[int]], d_boost: int):
    """Run kitty_seal and kitty_write_residual over the steps, the way the
    Kitty backend does, and return the cache, residual and full K/V."""
    lengths = [sum(q) for q in zip(*steps)]
    kv = [request_kv(n, r, d_boost) for r, n in enumerate(lengths)]
    block_table = torch.tensor(BLOCK_TABLE, dtype=torch.int32, device=DEVICE)
    residual_block_ids = torch.tensor(
        RESIDUAL_BLOCK_IDS, dtype=torch.int32, device=DEVICE
    )
    cache = empty_cache(block_table.numel(), d_boost)
    residual = torch.zeros(
        len(lengths),
        NUM_HEADS,
        KITTY_RESIDUAL_ROWS,
        HEAD_SIZE,
        dtype=torch.float16,
        device=DEVICE,
    )

    def gpu(a: np.ndarray, dtype=torch.int64) -> torch.Tensor:
        return torch.from_numpy(a).to(DEVICE, dtype)

    old = [0] * len(lengths)
    for query_lens in steps:
        new = [o + q for o, q in zip(old, query_lens)]
        query_start_loc = np.cumsum([0, *query_lens])
        plan = plan_kitty_step(query_start_loc, np.array(new))
        key, value = (
            torch.cat([seqs[i][o:n] for seqs, o, n in zip(kv, old, new)])
            for i in range(2)
        )
        kitty_seal(
            key,
            value,
            cache,
            residual,
            residual_block_ids,
            block_table,
            gpu(plan.k_jobs),
            gpu(plan.v_jobs),
            gpu(plan.old_lens),
            gpu(query_start_loc, torch.int32),
            d_boost,
        )
        kitty_write_residual(
            key,
            value,
            residual,
            residual_block_ids,
            gpu(plan.k_writes),
            gpu(plan.v_writes),
        )
        old = new
    return cache, residual, kv


def assert_sealed_k(page, d_boost, req, block, head):
    d = torch.arange(HEAD_SIZE, dtype=torch.float32, device=DEVICE)
    mins = 8 * req + block + d / 32
    if head == 1:
        mins = -(mins + 3 * boost_weight(d_boost))
    codes = CODES_0123 if head == 0 else CODES_3210
    assert_equal(segment(page, d_boost, "k_low"), codes)
    assert_equal(segment(page, d_boost, "k_high"), codes)
    sm = scale_min(page, d_boost, "k_scale_min")
    assert_equal(sm, torch.stack([torch.ones_like(mins), mins], dim=1))
    ranks = [d_boost] * (HEAD_SIZE - d_boost) + list(range(d_boost - 1, -1, -1))
    assert_equal(segment(page, d_boost, "k_rank"), ranks)


def assert_sealed_v(page, d_boost, req, block, head):
    t = torch.arange(PAGE, dtype=torch.float32, device=DEVICE)
    mins = 640 * req + 128 * block + t
    if head == 1:
        mins = -(mins + 3)
    codes = CODES_0123 if head == 0 else CODES_3210
    assert_equal(segment(page, d_boost, "v_data"), codes)
    sm = scale_min(page, d_boost, "v_scale_min")
    assert_equal(sm, torch.stack([torch.ones_like(mins), mins], dim=1))


@pytest.mark.parametrize("d_boost", D_BOOSTS)
@pytest.mark.parametrize("case", WRITE_PATH_CASES)
def test_kitty_seal_pages(case, d_boost):
    """Exactly the full blocks past the sink are sealed (V one block behind
    K), each from its own request's tokens in order, with the K channels of
    largest mean |K| boosted."""
    case = WRITE_PATH_CASES[case]
    cache, _, _ = run_steps(case["steps"], d_boost)
    for req, pages in enumerate(BLOCK_TABLE):
        for block, page_id in enumerate(pages):
            for head in range(NUM_HEADS):
                page = page_bytes(cache, page_id, head)
                if block in case["k_sealed"][req]:
                    assert_sealed_k(page, d_boost, req, block, head)
                else:
                    assert_untouched(page, d_boost, K_SEGMENTS)
                if block in case["v_sealed"][req]:
                    assert_sealed_v(page, d_boost, req, block, head)
                else:
                    assert_untouched(page, d_boost, V_SEGMENTS)
                assert_untouched(page, d_boost, ["padding"])


@pytest.mark.parametrize("case", WRITE_PATH_CASES)
def test_write_path_residual(case):
    """The residual holds each request's sink and unsealed tail tokens."""
    case = WRITE_PATH_CASES[case]
    _, residual, kv = run_steps(case["steps"], d_boost=32)
    for req, (key, value) in enumerate(kv):
        block = residual[RESIDUAL_BLOCK_IDS[req]]
        for src, runs in (
            (key, case["k_residual"][req]),
            (value, case["v_residual"][req]),
        ):
            for row, pos, count in runs:
                got = block[:, row : row + count]
                assert_equal(got, src[pos : pos + count].transpose(0, 1))


def test_kitty_seal_without_jobs_leaves_cache_unchanged():
    """Prefills shorter than two pages seal nothing."""
    cache, _, _ = run_steps([[100, 255, 128]], d_boost=32)
    assert (cache == SENTINEL).all()
