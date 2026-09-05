# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import torch

from vllm.models.qwen3_ksa.common.metadata import (
    KSAAttentionBackend,
    KSAAttentionMetadataBuilder,
    build_ksa_token_positions,
)
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec


def test_backend_declares_sliding_window_support() -> None:
    assert KSAAttentionBackend.supports_sliding_window()


def test_build_ksa_token_positions_handles_mixed_query_offsets() -> None:
    token_positions, token_to_request, computed = build_ksa_token_positions(
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens=torch.tensor([10, 7], dtype=torch.int32),
        num_actual_tokens=5,
    )

    torch.testing.assert_close(computed, torch.tensor([8, 4], dtype=torch.int32))
    torch.testing.assert_close(
        token_to_request, torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32)
    )
    torch.testing.assert_close(token_positions, torch.tensor([8, 9, 4, 5, 6]))


def _common_metadata(*, slot_mapping: torch.Tensor) -> CommonAttentionMetadata:
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens=torch.tensor([8, 9], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=5,
        max_query_len=3,
        max_seq_len=9,
        block_table_tensor=torch.tensor([[2, 3], [4, 5]], dtype=torch.int32),
        slot_mapping=slot_mapping,
        positions=torch.tensor([6, 7, 6, 7, 8], dtype=torch.int64),
    )


def test_text_metadata_keeps_normal_slot_mapping() -> None:
    slots = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int64)
    spec = FullAttentionSpec(
        block_size=8,
        num_kv_heads=2,
        head_size=4,
        dtype=torch.bfloat16,
    )
    builder = KSAAttentionMetadataBuilder(
        spec,
        ["layer.text_cache"],
        cast(Any, SimpleNamespace()),
        torch.device("cpu"),
    )
    metadata = builder.build(0, _common_metadata(slot_mapping=slots))

    torch.testing.assert_close(metadata.slot_mapping, slots)
    torch.testing.assert_close(metadata.token_positions, torch.tensor([6, 7, 6, 7, 8]))
    torch.testing.assert_close(
        metadata.boundary_mask,
        torch.tensor([False, True, False, True, False]),
    )


def test_summary_metadata_maps_only_chunk_boundaries() -> None:
    spec = FullAttentionSpec(
        block_size=8,
        num_kv_heads=2,
        head_size=4,
        dtype=torch.bfloat16,
        tokens_per_state=8,
    )
    builder = KSAAttentionMetadataBuilder(
        spec,
        ["layer"],
        cast(Any, SimpleNamespace()),
        torch.device("cpu"),
    )
    metadata = builder.build(
        0,
        _common_metadata(slot_mapping=torch.full((5,), -99, dtype=torch.int64)),
    )

    torch.testing.assert_close(
        metadata.slot_mapping,
        torch.tensor([-1, 2, -1, 4, -1], dtype=torch.int64),
    )


def test_sliding_metadata_uses_exact_chunk_aligned_transition() -> None:
    positions = torch.tensor([1023, 1024, 1025, 1031, 1032, 1033])
    num_reqs = positions.numel()
    common = CommonAttentionMetadata(
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32),
        query_start_loc_cpu=torch.arange(num_reqs + 1, dtype=torch.int32),
        seq_lens=(positions + 1).to(torch.int32),
        num_reqs=num_reqs,
        num_actual_tokens=num_reqs,
        max_query_len=1,
        max_seq_len=1034,
        block_table_tensor=torch.zeros((num_reqs, 65), dtype=torch.int32),
        slot_mapping=torch.arange(num_reqs, dtype=torch.int64),
        positions=positions,
    )
    spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=4,
        dtype=torch.bfloat16,
        sliding_window=1032,
    )
    builder = KSAAttentionMetadataBuilder(
        spec,
        ["layer.text_cache"],
        cast(Any, SimpleNamespace()),
        torch.device("cpu"),
    )
    metadata = builder.build(0, common)

    torch.testing.assert_close(
        metadata.chunk_indices,
        torch.tensor([127, 128, 128, 128, 129, 129]),
    )
    torch.testing.assert_close(
        metadata.text_start_positions,
        torch.tensor([0, 0, 0, 0, 8, 8]),
    )
    torch.testing.assert_close(
        metadata.visible_summary_lens,
        torch.tensor([0, 0, 0, 0, 1, 1]),
    )
    torch.testing.assert_close(
        metadata.token_to_request,
        torch.arange(num_reqs, dtype=torch.int32),
    )


def test_cudagraph_capture_metadata_binds_padded_runner_buffers() -> None:
    positions = torch.tensor([7, 0], dtype=torch.int64)
    slots = torch.tensor([15, -1], dtype=torch.int64)
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 1], dtype=torch.int32),
        seq_lens=torch.tensor([8, 0], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=128,
        block_table_tensor=torch.tensor([[2, 3], [0, 0]], dtype=torch.int32),
        slot_mapping=slots,
        positions=positions,
    )
    spec = FullAttentionSpec(
        block_size=8,
        num_kv_heads=2,
        head_size=4,
        dtype=torch.bfloat16,
    )
    builder = KSAAttentionMetadataBuilder(
        spec,
        ["layer.text_cache"],
        cast(Any, SimpleNamespace()),
        torch.device("cpu"),
    )

    metadata = builder.build_for_cudagraph_capture(common)

    assert metadata.is_cudagraph_capture
    assert metadata.positions.data_ptr() == positions.data_ptr()
    assert metadata.slot_mapping.data_ptr() == slots.data_ptr()
    torch.testing.assert_close(
        metadata.token_to_request,
        torch.tensor([0, 1], dtype=torch.int32),
    )
    assert (
        builder.get_cudagraph_support(cast(Any, SimpleNamespace()), spec)
        == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )


def test_runtime_metadata_accepts_full_graph_padding_rows() -> None:
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 1], dtype=torch.int32),
        seq_lens=torch.tensor([8, 0], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=128,
        block_table_tensor=torch.tensor([[2, 3], [0, 0]], dtype=torch.int32),
        slot_mapping=torch.tensor([15, -1], dtype=torch.int64),
        positions=torch.tensor([7, 0], dtype=torch.int64),
    )
    spec = FullAttentionSpec(
        block_size=8,
        num_kv_heads=2,
        head_size=4,
        dtype=torch.bfloat16,
    )
    builder = KSAAttentionMetadataBuilder(
        spec,
        ["layer.text_cache"],
        cast(Any, SimpleNamespace()),
        torch.device("cpu"),
    )

    metadata = builder.build(0, common)

    torch.testing.assert_close(metadata.token_positions, torch.tensor([7, 0]))
    torch.testing.assert_close(
        metadata.token_to_request,
        torch.tensor([0, 1], dtype=torch.int32),
    )
