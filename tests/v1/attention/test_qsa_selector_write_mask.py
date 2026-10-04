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


def _metadata(main_slots: torch.Tensor, is_dummy_batch: bool = False):
    query_start_loc = torch.tensor([0, NUM_TOKENS], dtype=torch.int32, device="cuda")
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([NUM_TOKENS], dtype=torch.int32, device="cuda"),
        num_reqs=1,
        num_actual_tokens=NUM_TOKENS,
        max_query_len=NUM_TOKENS,
        max_seq_len=NUM_TOKENS,
        block_table_tensor=torch.arange(64, dtype=torch.int32, device="cuda").unsqueeze(
            0
        ),
        slot_mapping=main_slots,
        is_dummy_batch=is_dummy_batch,
    )


def _slots(
    builder,
    metadata: CommonAttentionMetadata,
    compress_ratio: int,
    circular_buffer_size: int = 0,
) -> torch.Tensor:
    return builder(
        metadata,
        torch.zeros(NUM_TOKENS, dtype=torch.int32, device="cuda"),
        torch.zeros(NUM_TOKENS, dtype=torch.int64, device="cuda"),
        torch.zeros(NUM_TOKENS, dtype=torch.int32, device="cuda"),
        torch.zeros(NUM_TOKENS, dtype=torch.int64, device="cuda"),
        storage_block_size=STORAGE_BLOCK_SIZE,
        compress_ratio=compress_ratio,
        circular_buffer_size=circular_buffer_size,
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


@pytest.mark.parametrize("builder", BUILDERS, ids=["triton", "torch"])
@pytest.mark.parametrize("compress_ratio,circular_buffer_size", [(8, 0), (1, 64)])
def test_dummy_qsa_caches_do_not_write(
    builder, compress_ratio: int, circular_buffer_size: int
) -> None:
    main_slots = torch.full((NUM_TOKENS,), -1, dtype=torch.int64, device="cuda")
    slots = _slots(
        builder,
        _metadata(main_slots, is_dummy_batch=True),
        compress_ratio,
        circular_buffer_size,
    )
    assert not torch.any(slots >= 0)
