# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The receive kernel of a compact head (`kv_ce_topup`): a row it misses, takes
from the wrong place, or copies to the wrong staging part or region slot, is a row
of the context the attention reads uninitialized."""

import inspect
import os
import subprocess
import sys

import pytest
import torch

from tests.v1.attention import kv_ce_triton_launch as launch
from vllm.platforms import current_platform
from vllm.v1.attention.ops.kv_ce_topup import (
    FP8_ENTRY,
    MAX_RECEIVE_BLOCKS,
    RECEIVE_ROWS,
    ROW_DIM,
    peer_deltas,
    receive_blocks,
)


def test_peer_deltas_are_byte_offsets():
    block = torch.zeros((8, 16), dtype=torch.int16)
    deltas = peer_deltas(block, [block[2:], block[5:], block])
    assert deltas.dtype == torch.int64
    assert deltas.tolist() == [2 * 16 * 2, 5 * 16 * 2, 0]


def test_the_grid_follows_the_rows_the_host_copied():
    """Enough blocks for the largest copy below the limit, at least one, at most
    the cap: a union larger than the copy is walked in strides."""
    assert receive_blocks([0, 0], 100) == 1
    assert receive_blocks([5, 9, 3], 100) == -(-9 // RECEIVE_ROWS)
    assert receive_blocks([500, 7], 40) == -(-40 // RECEIVE_ROWS)
    huge = RECEIVE_ROWS * MAX_RECEIVE_BLOCKS * 3
    assert receive_blocks([huge], huge) == MAX_RECEIVE_BLOCKS


# Run under the Triton interpreter. The scales are powers of two, so every
# upconverted value is exact in BF16 and the reference needs no rounding mode.
TOPUP_SCRIPT = r"""
import torch
from vllm.v1.attention.ops import kv_ce_topup
from vllm.v1.attention.ops.kv_ce_topup import (
    FP8_ENTRY, MAX_SOURCES, RECEIVE_ROWS, ROW_DIM, peer_deltas,
)

# A small cap, so that the interpreter walks rows in strides.
kv_ce_topup.MAX_RECEIVE_BLOCKS = 4

def entries(rows, generator):
    raw = torch.empty((rows, FP8_ENTRY), dtype=torch.uint8)
    values = (torch.randn(rows, 512, generator=generator) * 50).clamp(-448, 448)
    raw[:, :512] = values.to(torch.float8_e4m3fn).view(torch.uint8)
    exponents = torch.randint(-8, 1, (rows, 4), generator=generator)
    raw[:, 512:528] = (2.0 ** exponents).float().view(torch.uint8).view(rows, 16)
    rope = torch.randn(rows, 64, generator=generator).to(torch.bfloat16)
    raw[:, 528:] = rope.view(torch.uint8).view(rows, 128)
    return raw

def upconvert(raw):
    nope = raw[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scale = raw[:, 512:528].contiguous().view(torch.float32)
    out = torch.empty((raw.shape[0], ROW_DIM), dtype=torch.bfloat16)
    out[:, :512] = (nope * scale.repeat_interleave(128, dim=1)).to(torch.bfloat16)
    out[:, 512:] = raw[:, 528:].contiguous().view(torch.bfloat16)
    return out

stride = 4 * RECEIVE_ROWS
cases = [
    # An over-copy, an exact copy, a miss and an empty union.
    (300, [120, 200, 290, 0], [150, 200, 0, 0]),
    # Misses longer than one stride of the grid; a union past the limit.
    (3 * stride + 5, [3 * stride + 5, 3 * stride + 45, 5], [0, 7, 0]),
    # As many sources as a launch takes, each with its own copy.
    (700, [0, 1, 16, 17, 650, 700, 699], [0, 0, 1, 16, 700, 0, 698]),
    # Nothing copied: every row comes from the peers.
    (50, [50, 3], [0, 0]),
]
for seed, (limit, counts, begins) in enumerate(cases):
    generator = torch.Generator().manual_seed(seed)
    sources = len(counts)
    assert sources <= MAX_SOURCES
    world = sources + 1
    slots = torch.randperm(world, generator=generator)[:sources].tolist()
    parts = torch.randperm(sources, generator=generator).tolist()
    sizes = [0] * world
    for slot, count in zip(slots, counts):
        sizes[slot] = count
    ends = torch.tensor([0, *torch.tensor(sizes).cumsum(0).tolist()],
                        dtype=torch.int32)
    peers = [entries(max(counts) + 3, generator) for _ in range(sources)]
    stage_rows, shard_rows = limit + 11, limit + 5
    stage = torch.full((sources * stage_rows, FP8_ENTRY), 0x5A, dtype=torch.uint8)
    for peer, part, begin in zip(peers, parts, begins):
        # What the copy engines brought; rows past a peer's union are junk.
        copied = min(begin, peer.shape[0])
        stage[part * stage_rows : part * stage_rows + copied] = peer[:copied]
    region = torch.full((world * shard_rows, ROW_DIM), float("nan"),
                        dtype=torch.bfloat16)
    want_stage, want_region = stage.clone(), region.clone()
    for peer, slot, part, count, begin in zip(peers, slots, parts, counts, begins):
        end = min(count, limit)
        if end > begin:
            start = part * stage_rows
            want_stage[start + begin : start + end] = peer[begin:end]
        start = slot * shard_rows
        want_region[start : start + end] = upconvert(peer[:end])
    receive_rows(
        stage, peer_deltas(stage, peers), ends,
        torch.tensor(slots, dtype=torch.int32),
        torch.tensor(parts, dtype=torch.int32),
        begins, limit, stage_rows, region, shard_rows,
    )
    assert torch.equal(stage, want_stage), seed
    assert torch.equal(region.view(torch.int16), want_region.view(torch.int16)), seed
print("ok")
"""


def test_every_union_row_is_upconverted_from_where_it_lies():
    """Run under the Triton interpreter: no GPU needed."""
    env = {**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-c", inspect.getsource(launch) + TOPUP_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-2000:]


@pytest.mark.skipif(
    not (current_platform.is_cuda() and torch.cuda.is_available()),
    reason="needs a CUDA GPU",
)
def test_received_rows_are_upconverted_like_the_serial_gather():
    """On the device, with arbitrary scales: the rows the kernel upconverts, from
    the staging and from the peers, are bitwise those
    `cp_gather_and_upconvert_fp8_kv_cache` writes."""
    from vllm import _custom_ops as ops

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(7)
    sources, block, limit, shard_rows = 3, 64, 640, 700
    stage_rows = 704
    counts, begins = [500, 64, 640], [100, 64, 630]
    slots, parts = [2, 0, 3], [1, 2, 0]
    sizes = [0] * (sources + 1)
    for slot, count in zip(slots, counts):
        sizes[slot] = count
    ends = torch.tensor(
        [0, *torch.tensor(sizes).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    peers = []
    for _ in range(sources):
        raw = torch.empty((stage_rows, FP8_ENTRY), dtype=torch.uint8, device=device)
        values = torch.randn(stage_rows, 512, device=device, generator=generator) * 50
        raw[:, :512] = values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        scales = torch.rand(stage_rows, 4, device=device, generator=generator) * 1e-2
        raw[:, 512:528] = scales.view(torch.uint8).view(stage_rows, 16)
        rope = torch.randn(stage_rows, 64, device=device, generator=generator)
        raw[:, 528:] = rope.to(torch.bfloat16).view(torch.uint8).view(stage_rows, 128)
        peers.append(raw)
    stage = torch.zeros((sources * stage_rows, FP8_ENTRY), dtype=torch.uint8).cuda()
    for peer, part, begin in zip(peers, parts, begins):
        stage[part * stage_rows : part * stage_rows + begin] = peer[:begin]
    region = torch.zeros(((sources + 1) * shard_rows, ROW_DIM), dtype=torch.bfloat16)
    region = region.cuda()
    launch.receive_rows(
        stage,
        peer_deltas(stage, peers),
        ends,
        torch.tensor(slots, dtype=torch.int32, device=device),
        torch.tensor(parts, dtype=torch.int32, device=device),
        begins,
        limit,
        stage_rows,
        region,
        shard_rows,
    )
    table = torch.arange(stage_rows // block, dtype=torch.int32, device=device)
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    for peer, slot, part, count in zip(peers, slots, parts, counts):
        end = min(count, limit)
        want = torch.empty((end, ROW_DIM), dtype=torch.bfloat16, device=device)
        ops.cp_gather_and_upconvert_fp8_kv_cache(
            peer.view(-1, block, FP8_ENTRY), want, table.view(1, -1), zero, 1
        )
        got = region[slot * shard_rows : slot * shard_rows + end]
        assert torch.equal(got.view(torch.int16), want.view(torch.int16))
        staged = stage[part * stage_rows : part * stage_rows + end]
        assert torch.equal(staged, peer[:end])
