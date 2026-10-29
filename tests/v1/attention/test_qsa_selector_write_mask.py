# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.models.qwen4_exp.common.qsa_cache import (
    _build_qsa_metadata_torch,
    build_qsa_metadata_triton,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="QSA metadata builders require CUDA"
)

NUM_TOKENS = 2048
COMPRESS_RATIO = 8
STORAGE_BLOCK_SIZE = 98
BUILDERS = [build_qsa_metadata_triton, _build_qsa_metadata_torch]


def _metadata(main_slots: torch.Tensor):
    query_start_loc = torch.tensor([0, NUM_TOKENS], dtype=torch.int32, device="cuda")
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([NUM_TOKENS], dtype=torch.int32, device="cuda"),
        num_reqs=1,
        num_actual_tokens=NUM_TOKENS,
        max_query_len=NUM_TOKENS,
        max_seq_len=NUM_TOKENS,
        block_table_tensor=torch.arange(
            1, 65, dtype=torch.int32, device="cuda"
        ).unsqueeze(0),
        slot_mapping=main_slots,
    )


def _slots(
    builder,
    metadata: CommonAttentionMetadata,
    compress_ratio: int,
) -> torch.Tensor:
    return builder(
        metadata,
        torch.zeros(NUM_TOKENS, dtype=torch.int32, device="cuda"),
        torch.zeros(NUM_TOKENS, dtype=torch.int64, device="cuda"),
        torch.zeros(NUM_TOKENS, dtype=torch.int32, device="cuda"),
        torch.zeros(NUM_TOKENS, dtype=torch.int64, device="cuda"),
        storage_block_size=STORAGE_BLOCK_SIZE,
        compress_ratio=compress_ratio,
    )[3]


@pytest.mark.parametrize("builder", BUILDERS, ids=["triton", "torch"])
@pytest.mark.parametrize("world_size,rank", [(2, 0), (2, 1), (4, 2)])
def test_replicated_selector_writes_each_rank(
    builder, world_size: int, rank: int
) -> None:
    positions = torch.arange(NUM_TOKENS, device="cuda")
    main_slots = torch.where(
        positions.remainder(world_size) == rank,
        positions.to(torch.int64),
        torch.full_like(positions, -1, dtype=torch.int64),
    )
    slots = _slots(builder, _metadata(main_slots), COMPRESS_RATIO)
    assert int((slots >= 0).sum()) == NUM_TOKENS // COMPRESS_RATIO


def test_compressed_qsa_cudagraph_replay_masks_null_block() -> None:
    main_slots = torch.arange(NUM_TOKENS, dtype=torch.int64, device="cuda")
    metadata = _metadata(main_slots)
    _slots(build_qsa_metadata_triton, metadata, COMPRESS_RATIO)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        slots = _slots(build_qsa_metadata_triton, metadata, COMPRESS_RATIO)

    graph.replay()
    assert int((slots >= 0).sum()) == NUM_TOKENS // COMPRESS_RATIO
    metadata.block_table_tensor.zero_()
    graph.replay()
    assert not torch.any(slots >= 0)
