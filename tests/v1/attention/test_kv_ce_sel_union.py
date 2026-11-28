# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The union pull of the PCP + DCP KV prefetch (`kv_ce_sel_union`).

A layer judged to reuse its predecessor's top-k must really read the same top-k,
or it attends rows nobody copied; the ranks must agree on those layers, or they
wait on each other's barriers; and a remapped index must address a row holding the
bytes the whole-context gather has at the original index.
"""

import inspect
import os
import subprocess
import sys

import pytest
import torch

from tests.v1.attention import kv_ce_triton_launch as launch
from vllm.platforms import current_platform
from vllm.v1.attention.ops.kv_ce_sel_union import (
    TopkUse,
    sel_digest,
    sel_union_positions,
)

# Layers of a 15-layer model with index groups: "full" layers run an indexer and
# start a group, "shared" layers reuse its top-k.
GLM_TYPES = ["full"] * 3 + ["shared"] * 3 + ["full"] + ["shared"] * 3
GLM_TYPES += ["full"] + ["shared"] * 3 + ["full"]
SHARED = frozenset({3, 4, 5, 7, 8, 9, 11, 12, 13})
ROWS = (0, 0, 4096)


def _types(num_layers: int) -> list[str]:
    """Three "full" layers, then one in four."""
    return ["full"] * 3 + [
        "shared" if (i - 2) % 4 else "full" for i in range(3, num_layers)
    ]


def _uses(types, metadata=None, rows=None, buffers=None) -> list[TopkUse | None]:
    """Each layer's use of the one top-k buffer; one metadata object for all
    layers unless `metadata[p]` names another."""
    buffer, shared_metadata = object(), object()
    uses: list[TopkUse | None] = []
    group, index = None, 0
    for position, kind in enumerate(types):
        if kind == "full" or group is None:
            group, index = object(), 0
        else:
            index += 1
        uses.append(
            TopkUse(
                group,
                index,
                (buffers or {}).get(position, buffer),
                (rows or {}).get(position, ROWS),
                (metadata or {}).get(position, shared_metadata),
            )
        )
    return uses


def _check_builders(uses, positions, builds):
    """No builder follows a compact layer: a build rewrites the remapped top-k on
    the auxiliary stream while the predecessor's attention still reads it. Every
    other compact layer continues its predecessor's run, with its metadata."""
    assert builds <= positions
    for p in builds:
        assert p - 1 not in positions
    for p in positions - builds:
        assert p - 1 in positions
        assert uses[p].metadata is uses[p - 1].metadata


def test_layers_after_the_indexer_layer_of_their_group_reuse_its_topk():
    positions, builds = sel_union_positions(_uses(GLM_TYPES))
    assert positions == SHARED
    # One union per run of reusing layers.
    assert builds == frozenset({3, 7, 11})


def test_the_first_reuser_builds_when_the_indexer_has_other_metadata():
    indexer_metadata = object()
    full = {p: indexer_metadata for p, kind in enumerate(GLM_TYPES) if kind == "full"}
    positions, builds = sel_union_positions(_uses(GLM_TYPES, metadata=full))
    assert positions == SHARED
    assert builds == frozenset({3, 7, 11})


def test_a_reuser_with_its_own_metadata_keeps_the_whole_prefetch():
    uses = _uses(GLM_TYPES, metadata={8: object()})
    positions, builds = sel_union_positions(uses)
    # Layer 8 would have to build right after the compact layer 7; 9 starts a run.
    assert positions == SHARED - {8}
    assert builds == frozenset({3, 7, 9, 11})
    _check_builders(uses, positions, builds)


def test_interleaved_metadata_alternates_compact_layers():
    # Reusers split over two KV cache groups: neighbours have different metadata.
    odd, even = object(), object()
    uses = _uses(GLM_TYPES, metadata={p: odd if p % 2 else even for p in SHARED})
    positions, builds = sel_union_positions(uses)
    assert positions == frozenset({3, 5, 7, 9, 11, 13})
    assert builds == positions
    _check_builders(uses, positions, builds)


@pytest.mark.parametrize("num_layers", [15, 78])
@pytest.mark.parametrize("seed", range(8))
def test_no_builder_follows_a_compact_layer(num_layers, seed):
    generator = torch.Generator().manual_seed(seed)
    pool = [object() for _ in range(3)]
    choice = torch.randint(len(pool), (num_layers,), generator=generator).tolist()
    uses = _uses(_types(num_layers), metadata=dict(enumerate(pool[c] for c in choice)))
    positions, builds = sel_union_positions(uses)
    _check_builders(uses, positions, builds)
    whole, _ = sel_union_positions(_uses(_types(num_layers)))
    assert positions <= whole


def test_a_layer_reading_other_topk_rows_is_not_compact():
    positions, builds = sel_union_positions(_uses(GLM_TYPES, rows={8: (16, 0, 4096)}))
    # Layer 8 reads other rows than layer 7, and layer 9 other rows than layer 8.
    assert positions == SHARED - {8, 9}
    assert builds == frozenset({3, 7, 11})


def test_a_layer_with_another_topk_buffer_is_not_compact():
    positions, builds = sel_union_positions(_uses(GLM_TYPES, buffers={4: object()}))
    assert positions == frozenset({3, 7, 8, 9, 11, 12, 13})
    assert builds == frozenset({3, 7, 11})


def test_layers_that_cannot_reuse_a_topk_break_the_run():
    # No index group with an indexer, a HiSparse group, or several prefill chunks.
    uses = _uses(GLM_TYPES)
    for position in (6, 7, 8, 9):
        uses[position] = None
    assert sel_union_positions(uses)[0] == frozenset({3, 4, 5, 11, 12, 13})
    assert sel_union_positions([None] * 15) == (frozenset(), frozenset())


def test_an_order_that_skips_the_indexer_layer_starts_no_run():
    uses = _uses(GLM_TYPES)
    # Order ..., 5, 7, ...: layer 7's predecessor is in another group.
    positions, builds = sel_union_positions(uses[:6] + uses[7:])
    assert positions == frozenset({3, 4, 5, 7, 8, 10, 11, 12})
    assert builds == frozenset({3, 7, 10})


@pytest.mark.parametrize("num_layers", [2, 15, 61, 78, 300])
def test_the_digest_tells_every_set_of_positions_apart(num_layers):
    positions, builds = sel_union_positions(_uses(_types(num_layers)))
    digest = sel_digest(positions, builds)
    # What the plan hash packs as int64.
    assert torch.tensor(digest, dtype=torch.int64).tolist() == digest
    assert digest == [len(positions), *sorted(positions), len(builds), *sorted(builds)]
    if num_layers == 78:
        assert len(positions) == 57 and max(positions) == 77 and len(builds) == 19
    # A position moved from one set to the other changes the digest.
    first = frozenset({3, 4})
    assert sel_digest(first, frozenset({3})) != sel_digest(frozenset({3}), first)
    assert sel_digest(first, frozenset()) != sel_digest(frozenset({3}), frozenset())


# Run under the Triton interpreter. The scales are powers of two, so every
# upconverted value is exact in BF16 and the reference needs no rounding mode.
KERNELS_SCRIPT = r"""
import torch
from vllm.v1.attention.ops import kv_ce_sel_union as su

def convert(topk, dcp, shard_rows):
    # Rank-major positions of request-local token indices (interleave 1).
    return torch.where(topk >= 0, topk % dcp * shard_rows + topk // dcp, -1)

def ref_union(ps, dcp, shard_rows, head_rows):
    marks = torch.zeros(dcp * head_rows, dtype=torch.uint8)
    for p in ps:
        v = p[p >= 0]
        owner, row = v // shard_rows, v % shard_rows
        hit = row < head_rows
        marks[owner[hit] * head_rows + row[hit]] = 1
    cum = marks.view(dcp, head_rows).cumsum(1, dtype=torch.int32).flatten()
    return marks, cum

def ref_remap(p, cum, shard_rows, head_rows):
    owner, row = p // shard_rows, p % shard_rows
    head = (p >= 0) & (row < head_rows)
    idx = torch.where(head, owner * head_rows + row, 0)
    return torch.where(head, owner * shard_rows + cum[idx] - 1, p)

# Mark, OR, scan and remap: every rank gets the same numbering.
dcp, seq_len, k, epoch = 4, 40_000, 256, 7
shard_rows = -(-seq_len // dcp)
head_rows = shard_rows - 300
n = dcp * head_rows
assert n > 2 * su.SCAN_BLOCK and n % su.SCAN_BLOCK
ps, marks = [], []
for rank in range(dcp):
    g = torch.Generator().manual_seed(rank)
    topk = torch.randint(0, seq_len, (9, k), generator=g)
    topk[:, k - rank * 10 - 1 :] = -1
    p = convert(topk, dcp, shard_rows)
    ps.append(p)
    # Tags of earlier builds stay: only this epoch's count.
    m = torch.randint(0, epoch, (n + 100,), generator=g).to(torch.uint8)
    mark_selection(p.to(torch.int32), m, shard_rows, head_rows, epoch)
    marks.append(m)
want_marks, want_cum = ref_union(ps, dcp, shard_rows, head_rows)
sizes = want_marks.view(dcp, head_rows).sum(1).to(torch.int32)
want_ends = torch.cat([torch.zeros(1, dtype=torch.int32), sizes.cumsum(0)])
for rank in range(dcp):
    peers = torch.stack(marks)
    peers[rank].fill_(epoch)  # this rank's row is never read
    union = torch.full((n + 100,), 9, dtype=torch.uint8)
    counts = torch.full((-(-n // su.SCAN_BLOCK),), -5, dtype=torch.int32)
    union_marks(marks[rank], peers, union, counts, n, rank, dcp, epoch)
    assert torch.equal(union[:n], want_marks) and bool((union[n:] == 9).all())
    cum = torch.zeros(n, dtype=torch.int32)
    ends = torch.full((dcp + 1,), -1, dtype=torch.int32)
    ends[0] = 0
    scan_union(union, counts, cum, ends, dcp, head_rows)
    assert torch.equal(ends, want_ends)
    assert torch.equal((cum.view(dcp, head_rows) - ends[:-1, None]).flatten(), want_cum)
    out = torch.empty(ps[rank].shape, dtype=torch.int32)
    remap_topk(ps[rank].to(torch.int32), cum, ends, out, shard_rows, head_rows)
    assert torch.equal(out.long(), ref_remap(ps[rank], want_cum, shard_rows, head_rows))

# Pack: this rank's (rank 1) union rows, raw to the front of the wire buffer and
# upconverted to the front of its slot, in row order.
def cache(blocks, size, pad, g):
    raw = torch.zeros((blocks, size, 656 + pad), dtype=torch.uint8)
    values = (torch.randn(blocks, size, 512, generator=g) * 50).clamp(-448, 448)
    raw[..., :512] = values.to(torch.float8_e4m3fn).view(torch.uint8)
    scales = 2.0 ** torch.randint(-8, 1, (blocks, size, 4), generator=g)
    raw[..., 512:528] = scales.float().view(torch.uint8).view(blocks, size, 16)
    rope = torch.randn(blocks, size, 64, generator=g).to(torch.bfloat16)
    raw[..., 528:656] = rope.view(torch.uint8).view(blocks, size, 128)
    return raw[..., :656] if pad else raw

def upconvert(raw):
    nope = raw[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scale = raw[:, 512:528].contiguous().view(torch.float32)
    out = torch.empty((raw.shape[0], 576), dtype=torch.bfloat16)
    out[:, :512] = (nope * scale.repeat_interleave(128, dim=1)).to(torch.bfloat16)
    out[:, 512:] = raw[:, 528:].contiguous().view(torch.bfloat16)
    return out

for start, pad in ((0, 0), (3, 16)):
    g = torch.Generator().manual_seed(11 + start)
    block_size, num_blocks, head_rows, shard_rows = 64, 24, 700, 720
    kv = cache(num_blocks, block_size, pad, g)
    table = torch.randperm(num_blocks, generator=g)[: -(-(head_rows - start) // 64)]
    table = table.to(torch.int32).view(1, -1).contiguous()
    starts = torch.tensor([start], dtype=torch.int32)
    mine = (torch.rand(head_rows, generator=g) < 0.3).to(torch.uint8)
    mine[:start] = 0
    other = (torch.rand(head_rows, generator=g) < 0.5).to(torch.uint8)
    union = torch.cat([other, mine])
    cum = union.cumsum(0, dtype=torch.int32)
    ends = torch.tensor([0, int(other.sum()), int(union.sum())], dtype=torch.int32)
    args = (kv, table, starts, union, cum, ends, 1, head_rows)
    wire = torch.full((head_rows + 5, 656), 0x5A, dtype=torch.uint8)
    region = torch.full((2 * shard_rows, 576), float("nan"), dtype=torch.bfloat16)
    pack_rows(*args, wire, region[shard_rows:])
    rows = torch.nonzero(mine).flatten()
    token = rows - start
    entries = kv[table[0, token // block_size].long(), token % block_size]
    u = rows.numel()
    assert torch.equal(wire[:u], entries) and bool((wire[u:] == 0x5A).all())
    slot = region[shard_rows:]
    assert torch.equal(slot[:u].view(torch.int16), upconvert(entries).view(torch.int16))
    assert bool(slot[u:].isnan().all()) and bool(region[:shard_rows].isnan().all())
print("ok")
"""


def test_union_kernels_match_the_reference():
    """Run under the Triton interpreter: no GPU needed."""
    env = {**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-c", inspect.getsource(launch) + KERNELS_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-2000:]


# CUDA only: the FP8 pack kernel, and the readback through the driver (`cuda_direct`).
cuda = pytest.mark.skipif(
    not (current_platform.is_cuda() and torch.cuda.is_available()),
    reason="needs a CUDA GPU",
)


@cuda
@pytest.mark.parametrize("start", [0, 3])
@pytest.mark.parametrize("stride_pad", [0, 16])
def test_pack_is_bitwise_the_serial_upconvert(start, stride_pad):
    """On the device, with arbitrary scales: a packed row is bitwise the row
    `cp_gather_and_upconvert_fp8_kv_cache` writes for it."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.ops.kv_ce_sel_union import ROW_DIM

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(7 + start)
    block_size, num_blocks, head_rows = 64, 80, 3000
    entry = 656
    raw = torch.empty(
        (num_blocks, block_size, entry + stride_pad), dtype=torch.uint8, device=device
    )
    values = torch.randn(
        num_blocks, block_size, 512, device=device, generator=generator
    )
    raw[..., :512] = (
        (values * 50).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    )
    scales = torch.rand(num_blocks, block_size, 4, device=device, generator=generator)
    raw[..., 512:528] = (scales * 1e-2).view(torch.uint8).view(num_blocks, -1, 16)
    rope = torch.randn(num_blocks, block_size, 64, device=device, generator=generator)
    raw[..., 528:656] = (
        rope.to(torch.bfloat16).view(torch.uint8).view(num_blocks, -1, 128)
    )
    kv = raw[..., :entry] if stride_pad else raw
    used = head_rows - start
    table = torch.randperm(num_blocks, device=device)[: -(-used // block_size)]
    table = table.to(torch.int32).view(1, -1).contiguous()
    starts = torch.tensor([start], dtype=torch.int32, device=device)
    full = torch.full(
        (head_rows, ROW_DIM), float("nan"), dtype=torch.bfloat16, device=device
    )
    ops.cp_gather_and_upconvert_fp8_kv_cache(kv, full, table, starts, 1)
    mine = (torch.rand(head_rows, device=device) < 0.3).to(torch.uint8)
    mine[:start] = 0
    other = (torch.rand(head_rows, device=device) < 0.5).to(torch.uint8)
    union = torch.cat([other, mine])
    cum = union.cumsum(0, dtype=torch.int32)
    ends = torch.tensor(
        [0, int(other.sum()), int(union.sum())], dtype=torch.int32, device=device
    )
    region = torch.zeros((head_rows, ROW_DIM), dtype=torch.bfloat16, device=device)
    wire = torch.empty((head_rows, entry), dtype=torch.uint8, device=device)
    launch.pack_rows(kv, table, starts, union, cum, ends, 1, head_rows, wire, region)
    rows = torch.nonzero(mine).flatten()
    u = rows.numel()
    assert torch.equal(region[:u].view(torch.int16), full[rows].view(torch.int16))
    assert int(region[u:].abs().sum()) == 0


@cuda
def test_the_union_sizes_reach_the_host_without_a_wait():
    from vllm.v1.attention.ops.kv_ce_sel_union import UnionSizes

    sizes = UnionSizes(4)
    assert sizes.exact() is None
    ends = torch.tensor([0, 5, 5, 12, 40], dtype=torch.int32, device="cuda")
    sizes.record(ends, torch.cuda.current_stream())
    torch.accelerator.synchronize()
    assert sizes.exact() == [5, 0, 7, 28]
