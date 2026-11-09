# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.kernels.attention.dsa.dcp_indexer_rocm import (
    pack_dcp_candidates,
    select_dcp_candidates,
)
from vllm.platforms import current_platform

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
