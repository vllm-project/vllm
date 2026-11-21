# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest
import torch

from vllm.model_executor.kernels.attention.dsa.dcp_indexer_rocm import (
    FP8_DTYPE,
    _pack_local_keys,
    _restore_global_keys,
    pack_dcp_candidates,
    select_dcp_candidates,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm DCP indexer"
)


@pytest.mark.parametrize("world_size", [2, 8])
@pytest.mark.parametrize("interleave", [1, 16])
@pytest.mark.parametrize("with_offsets", [False, True])
def test_global_candidates_match_score_and_token_order(
    world_size, interleave, with_offsets
):
    """Mixed signs, ties, short/empty shards and per-row offsets retain global ids."""
    k, rows, width = 8, 3, 19
    generator = torch.Generator().manual_seed(17)
    shards = []
    expected: list[list[tuple[float, int]]] = [[] for _ in range(rows)]
    for rank in range(world_size):
        scores = torch.randint(-3, 4, (rows, width), generator=generator).float()
        scores[1, 1] = -0.0
        scores[1, 2] = 0.0
        starts = (
            torch.tensor([2, 0, 1])
            if with_offsets
            else torch.zeros(rows, dtype=torch.long)
        )
        indices = torch.full((rows, k), -1, dtype=torch.int32)
        for row in range(rows):
            length = 0 if row == 2 else width - int(starts[row]) - rank
            candidates = [
                (float(scores[row, int(starts[row]) + i]), i) for i in range(length)
            ]
            candidates.sort(key=lambda item: (-item[0], item[1]))
            for column, (score, local) in enumerate(candidates[:k]):
                indices[row, column] = local
                token = (
                    local // interleave * world_size * interleave
                    + rank * interleave
                    + local % interleave
                )
                expected[row].append((score, token))
        shards.append(
            pack_dcp_candidates(
                scores.cuda(),
                indices.cuda(),
                rank,
                world_size,
                interleave,
                starts.cuda() if with_offsets else None,
            )
        )
    gathered = torch.cat(shards, dim=1)
    out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    select_dcp_candidates(gathered, out)
    reference = torch.full((rows, k), -1, dtype=torch.int32)
    for row, candidates in enumerate(expected):
        candidates.sort(key=lambda item: (-item[0], item[1]))
        reference[row, : min(k, len(candidates))] = torch.tensor(
            [token for _, token in candidates[:k]], dtype=torch.int32
        )
    # Selection is unordered within a row: compare the selected sets.
    torch.testing.assert_close(
        out.cpu().sort(dim=-1).values, reference.sort(dim=-1).values, rtol=0, atol=0
    )


def test_empty_shard_and_graph_replay():
    indices = torch.tensor([[0, 1, -1, 3]], dtype=torch.int32, device="cuda")
    scores = torch.tensor([[1.0, 4.0, -float("inf"), float("nan")]], device="cuda")
    out = torch.empty_like(indices)
    empty = pack_dcp_candidates(scores[:, :0], indices, 0, 2, 1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            select_dcp_candidates(
                torch.cat(
                    [empty, pack_dcp_candidates(scores, indices, 1, 2, 1)], dim=1
                ),
                out,
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select_dcp_candidates(
            torch.cat([empty, pack_dcp_candidates(scores, indices, 1, 2, 1)], dim=1),
            out,
        )
    scores[0, 0] = 6.0
    graph.replay()
    torch.testing.assert_close(
        out.cpu().sort(dim=-1).values,
        torch.tensor([[-1, -1, 1, 3]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test_strided_candidates_mask_nonfinite_and_preserve_output_padding():
    """Fused packing preserves signed-zero ties and writes only output slots."""
    scores = torch.tensor(
        [[0.0, 99.0, -0.0, 99.0, float("inf"), 99.0, float("nan"), 99.0, -2.0, 99.0]],
        device="cuda",
    )[:, ::2]
    indices = torch.tensor(
        [[0, 99, 1, 99, 2, 99, 3, 99, 4, 99, -1, 99, 5, 99]],
        device="cuda",
        dtype=torch.int32,
    )[:, ::2]
    packed = pack_dcp_candidates(scores, indices, 1, 2, 16)
    storage = torch.full((1, 18), 73, device="cuda", dtype=torch.int32)
    select_dcp_candidates(packed, storage[:, ::2])
    torch.testing.assert_close(
        storage[:, ::2].sort(-1).values.cpu(),
        torch.tensor([[-1, -1, -1, -1, -1, -1, 16, 17, 20]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    assert bool((storage[:, 1::2] == 73).all())


def _pack_rank_cache(keys, scales, cu_seq_lens, counts, rank, world_size, interleave):
    """Stage one rank's share of the global keys in its own paged cache."""
    head_dim = keys.shape[1]
    width = head_dim + 4
    page = 64
    rows_per_req = [int(counts[q].item()) for q in range(counts.shape[0])]
    pages_per_req = [max(1, -(-rows // page)) for rows in rows_per_req]
    table_stride = max(pages_per_req)
    block_table = torch.zeros(
        (counts.shape[0], table_stride), dtype=torch.int32, device=keys.device
    )
    cache = torch.zeros(
        (sum(pages_per_req), page, width), dtype=torch.uint8, device=keys.device
    )
    # A cache page holds `page` value rows of `head_dim` bytes followed by
    # `page` four-byte scales -- the layout the pack kernel reads.
    page_bytes = page * width
    flat = cache.view(-1)
    next_page = 0
    for req, rows in enumerate(rows_per_req):
        block_table[req, : pages_per_req[req]] = torch.arange(
            next_page, next_page + pages_per_req[req], dtype=torch.int32
        )
        start = next_page
        next_page += pages_per_req[req]
        seq_len = int(cu_seq_lens[req + 1].item()) - int(cu_seq_lens[req].item())
        for local in range(rows):
            # `dcp_local_to_global_index`, as a host-side scalar.
            pos = (
                local // interleave * (world_size * interleave)
                + rank * interleave
                + local % interleave
            )
            assert pos < seq_len
            token = int(cu_seq_lens[req].item()) + pos
            slot = local % page
            base = (start + local // page) * page_bytes
            flat[base + slot * head_dim : base + (slot + 1) * head_dim] = keys[token]
            scale_at = base + page * head_dim + slot * 4
            flat[scale_at : scale_at + 4] = scales[token : token + 1].view(torch.uint8)
    return cache, block_table


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("interleave", [1, 16])
def test_global_key_prefill_restores_global_token_order(world_size, interleave):
    """Pack and restore return every token's key and scale in global order.

    Each rank stages its interleaved share of one chunk's index keys, the
    per-rank buffers are concatenated the way the DCP `all_gather`
    concatenates them, and the restore must invert the sharding exactly.
    """
    device = torch.device("cuda")
    head_dim, page = 64, 64
    width = head_dim + 4
    seq_lens = torch.tensor([130, 5, 70], dtype=torch.int32, device=device)
    cu_seq_lens = torch.cat(
        (seq_lens.new_zeros(1), seq_lens.cumsum(0, dtype=torch.int32))
    )
    total = int(cu_seq_lens[-1].item())
    generator = torch.Generator(device=device).manual_seed(11)
    keys = torch.randint(
        0, 256, (total, head_dim), dtype=torch.uint8, generator=generator, device=device
    )
    scales = torch.rand(total, generator=generator, device=device)
    token_to_seq = torch.repeat_interleave(
        torch.arange(seq_lens.numel(), dtype=torch.int32, device=device), seq_lens
    )
    counts = get_dcp_local_seq_lens(seq_lens, world_size, None, interleave)
    all_cu = torch.cat(
        (counts.new_zeros((1, world_size)), counts.cumsum(0, dtype=torch.int32))
    )
    # The kernel keeps each rank's payload a whole number of 16-byte packs.
    align = 16 // math.gcd(width, 16)
    max_local_total = -(-int(counts.sum(dim=0).max().item()) // align) * align
    local_width = (
        -(-int(seq_lens.max().item()) // (world_size * interleave)) * interleave
    )

    packed = []
    for rank in range(world_size):
        cache, block_table = _pack_rank_cache(
            keys, scales, cu_seq_lens, counts[:, rank], rank, world_size, interleave
        )
        local_cu = torch.cat(
            (counts.new_zeros(1), counts[:, rank].cumsum(0, dtype=torch.int32))
        )
        rows = torch.zeros((max_local_total, width), dtype=torch.uint8, device=device)
        _pack_local_keys[seq_lens.numel(), max(1, -(-local_width // 16))](
            cache.view(torch.uint8),
            block_table,
            local_cu,
            rows,
            cache.stride(0) * cache.element_size(),
            block_table.stride(0),
            PAGE=page,
            HEAD_DIM=head_dim,
            WIDTH=width,
            TILE=triton.next_power_of_2(width),
        )
        packed.append(rows)
    gathered = torch.cat(packed, dim=0)

    out_keys = torch.empty((total, head_dim), dtype=FP8_DTYPE, device=device)
    out_scales = torch.empty(total, dtype=torch.float32, device=device)
    _restore_global_keys[(-(-total // 16),)](
        gathered,
        token_to_seq,
        cu_seq_lens,
        all_cu,
        out_keys.view(torch.uint8),
        out_scales.view(torch.uint8),
        total,
        max_local_total,
        World=world_size,
        Interleave=interleave,
        HEAD_DIM=head_dim,
        WIDTH=width,
    )
    torch.testing.assert_close(out_keys.view(torch.uint8), keys, rtol=0, atol=0)
    torch.testing.assert_close(out_scales, scales, rtol=0, atol=0)


def test_rowshard_prefill_is_declared_by_the_builder():
    """Only builders that keep one key per token opt into row sharding.

    The gate is a builder capability, not a model-name check, so a model that
    compresses its index keys keeps the score-every-row path.
    """
    from vllm.models.kimi_k3.common.dsa_cache import KimiK3IndexerMetadataBuilder
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )

    assert not DeepseekV32IndexerMetadataBuilder._dcp_prefill_rowshard_supported
    assert KimiK3IndexerMetadataBuilder._dcp_prefill_rowshard_supported
