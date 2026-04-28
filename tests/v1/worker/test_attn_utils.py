# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata


pytestmark = pytest.mark.skip_global_cleanup


class _RecordingBuilder:
    def __init__(self) -> None:
        self.seen_metadata: list[CommonAttentionMetadata] = []

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> CommonAttentionMetadata:
        assert common_prefix_len == 0
        _ = common_attn_metadata.seq_lens_cpu
        self.seen_metadata.append(common_attn_metadata)
        return common_attn_metadata


class _FakeAttentionGroup:
    def __init__(self, builder: _RecordingBuilder, layer_name: str) -> None:
        self._builder = builder
        self.layer_names = [layer_name]

    def get_metadata_builder(self, common_prefix_len: int) -> _RecordingBuilder:
        assert common_prefix_len == 0
        return self._builder


def test_build_attn_metadata_seeds_seq_lens_cpu_for_all_kv_groups():
    original_seq_lens_prop = CommonAttentionMetadata.seq_lens_cpu
    original_seq_lens_fget = original_seq_lens_prop.fget
    original_num_computed_prop = CommonAttentionMetadata.num_computed_tokens_cpu
    original_num_computed_fget = original_num_computed_prop.fget
    lazy_seq_lens_init_count = 0
    lazy_num_computed_init_count = 0

    def tracking_seq_lens_cpu(self: CommonAttentionMetadata) -> torch.Tensor:
        nonlocal lazy_seq_lens_init_count
        if self._seq_lens_cpu is None:
            lazy_seq_lens_init_count += 1
        assert original_seq_lens_fget is not None
        return original_seq_lens_fget(self)

    def tracking_num_computed_tokens_cpu(
        self: CommonAttentionMetadata,
    ) -> torch.Tensor:
        nonlocal lazy_num_computed_init_count
        if self._num_computed_tokens_cpu is None:
            lazy_num_computed_init_count += 1
        assert original_num_computed_fget is not None
        return original_num_computed_fget(self)

    CommonAttentionMetadata.seq_lens_cpu = property(tracking_seq_lens_cpu)
    CommonAttentionMetadata.num_computed_tokens_cpu = property(
        tracking_num_computed_tokens_cpu
    )
    try:
        builder0 = _RecordingBuilder()
        builder1 = _RecordingBuilder()
        attn_groups = [
            [_FakeAttentionGroup(builder0, "layer0")],
            [_FakeAttentionGroup(builder1, "layer1")],
        ]
        query_start_loc_cpu = torch.tensor([0, 2, 5], dtype=torch.int32)
        query_start_loc_gpu = query_start_loc_cpu.clone()
        seq_lens = torch.tensor([8, 16], dtype=torch.int32)
        shared_block_table = torch.zeros((2, 2), dtype=torch.int32)
        shared_slot_mapping = torch.zeros(5, dtype=torch.int64)

        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=2,
            num_tokens=5,
            query_start_loc_gpu=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=3,
            seq_lens=seq_lens,
            max_seq_len=16,
            block_tables=(shared_block_table, shared_block_table),
            slot_mappings=(shared_slot_mapping, shared_slot_mapping),
            kv_cache_config=SimpleNamespace(kv_cache_groups=[object(), object()]),
        )
    finally:
        CommonAttentionMetadata.seq_lens_cpu = original_seq_lens_prop
        CommonAttentionMetadata.num_computed_tokens_cpu = original_num_computed_prop

    for metadata in builder0.seen_metadata + builder1.seen_metadata:
        _ = metadata.num_computed_tokens_cpu

    assert lazy_seq_lens_init_count == 0
    assert lazy_num_computed_init_count == 0
    assert set(attn_metadata) == {"layer0", "layer1"}
    assert len(builder0.seen_metadata) == 1
    assert len(builder1.seen_metadata) == 1
    assert builder0.seen_metadata[0]._seq_lens_cpu is not None
    assert builder0.seen_metadata[0]._seq_lens_cpu is builder1.seen_metadata[0]._seq_lens_cpu
    assert builder0.seen_metadata[0]._num_computed_tokens_cpu is not None
    assert (
        builder0.seen_metadata[0]._num_computed_tokens_cpu
        is builder1.seen_metadata[0]._num_computed_tokens_cpu
    )