# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.glm5next.common.sparse_indexer import _kpool_flat_page_view
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MLAAttentionSpec,
)
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    prepare_kernel_block_sizes,
)

ROW = 132  # 128 fp8 bytes + 4 scale bytes per indexer state


class _RecordingBuilder:
    requires_block_table_width = False
    requires_block_stride_bytes = True

    def __init__(self, kv_cache_spec, *_args, block_stride_bytes=None, **_kwargs):
        self.kv_cache_spec = kv_cache_spec
        self.block_stride_bytes = block_stride_bytes

    def set_kernel_block_size(self, block_size):
        self.kernel_block_size = block_size


class _RecordingBackend:
    @staticmethod
    def get_builder_cls():
        return _RecordingBuilder


@pytest.mark.parametrize(("page_states", "stride_pages"), [(32, 7), (64, 5)])
def test_kernel_page_table_uses_physical_stride(page_states, stride_pages):
    pages_per_block = 4
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.kv_cache_spec = MLAAttentionSpec(
        block_size=pages_per_block * page_states * 4,
        num_kv_heads=1,
        head_size=128,
        head_size_v=0,
        dtype=torch.uint8,
        state_content_bytes=ROW,
        tokens_per_state=4,
        kernel_page_size=page_states * 4,
    )
    builder._kernel_page_size = builder.kv_cache_spec.kernel_page_size
    builder.block_stride_bytes = stride_pages * page_states * ROW
    builder.arange_buffer = torch.arange(16, dtype=torch.int32)

    page_size, page_table = builder._expand_kernel_page_table(
        torch.tensor([[0, 2]], dtype=torch.int32)
    )
    assert page_size == page_states
    assert page_table.tolist() == [
        list(range(pages_per_block))
        + list(range(2 * stride_pages, 2 * stride_pages + pages_per_block))
    ]


def test_single_page_table_uses_strided_cache_block_ids():
    page_states, row_width, num_blocks = 64, ROW, 3
    page_bytes = page_states * row_width
    block_stride_bytes = 2 * page_bytes + 4
    backing = torch.empty(
        (num_blocks - 1) * block_stride_bytes + page_bytes, dtype=torch.uint8
    )
    cache = backing.as_strided(
        (num_blocks, page_states, row_width),
        (block_stride_bytes, row_width, 1),
    )
    kernel_cache = _kpool_flat_page_view(cache)

    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.kv_cache_spec = MLAAttentionSpec(
        block_size=page_states * 4,
        num_kv_heads=1,
        head_size=128,
        head_size_v=0,
        dtype=torch.uint8,
        state_content_bytes=ROW,
        tokens_per_state=4,
        kernel_page_size=page_states * 4,
    )
    builder._kernel_page_size = builder.kv_cache_spec.kernel_page_size
    builder.block_stride_bytes = block_stride_bytes
    block_table = torch.tensor([[0, 2]], dtype=torch.int32)

    page_size, page_table = builder._expand_kernel_page_table(block_table)

    assert block_stride_bytes % 4 == 0
    assert block_stride_bytes % page_bytes != 0
    assert kernel_cache is cache
    assert page_size == page_states
    assert page_table is block_table
    assert kernel_cache[page_table[0, 1]].storage_offset() == 2 * block_stride_bytes


@pytest.mark.parametrize(
    ("block_size", "expected_kernel_page", "expected_alignment"),
    [
        (640, 128, 32 * ROW),
        (1024, 256, 64 * ROW),
        (1152, 128, 32 * ROW),
        (256, 256, None),
    ],
)
def test_kpool_spec_declares_repage_alignment(
    default_vllm_config, block_size, expected_kernel_page, expected_alignment
):
    from vllm.models.glm5next.common.attention import Glm5NextIndexerCache

    default_vllm_config.cache_config.block_size = block_size
    cache = Glm5NextIndexerCache(
        head_dim=ROW,
        dtype=torch.uint8,
        prefix=f"indexer_{block_size}",
        cache_config=default_vllm_config.cache_config,
        index_kpool=4,
    )
    spec = cache.get_kv_cache_spec(default_vllm_config)
    assert spec.kernel_page_size == expected_kernel_page
    assert spec.block_stride_alignment == expected_alignment
    assert DeepseekV32IndexerBackend.get_supported_kernel_block_sizes(spec) == [64]
    assert DeepseekV32IndexerBackend.get_strided_block_page_rows(spec) == (
        expected_kernel_page // 4
    )


def test_dense_kpool_cache_uses_declared_kernel_page_size():
    spec = MLAAttentionSpec(
        block_size=1024,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        state_content_bytes=ROW,
        tokens_per_state=4,
        kernel_page_size=256,
    )
    page_size = spec.page_size_bytes
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * page_size,
                layers=["indexer"],
                layer_stride=2 * page_size,
                block_stride=page_size,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["indexer"], spec)],
    )
    groups = [[AttentionGroup(DeepseekV32IndexerBackend, ["indexer"], spec, 0)]]

    kernel_block_sizes = prepare_kernel_block_sizes(config, groups)
    assert kernel_block_sizes == [64]

    caches = allocate_kv_cache(
        config, torch.device("cpu"), KVCacheLayout.LBHNC, kernel_block_sizes
    )
    assert caches["indexer"].shape == (8, 1, 64, ROW)

    manager_caches = allocate_kv_cache(
        config, torch.device("cpu"), KVCacheLayout.LBHNC, [spec.block_size]
    )
    assert manager_caches["indexer"].shape == (2, 1, 256, ROW)


@pytest.mark.parametrize(
    (
        "manager_block_size",
        "stride_pages",
        "kernel_block_size",
        "builder_block_size",
        "kernel_page_size",
    ),
    [
        (1024, 1, 64, 256, None),
        (256, 1, 64, 256, None),
        (256, 1, 256, 256, 256),
        (1024, 1, 1024, 1024, 256),
        (1024, 2, 1024, 1024, 256),
    ],
)
def test_kpool_builder_uses_the_cache_view_geometry(
    manager_block_size,
    stride_pages,
    kernel_block_size,
    builder_block_size,
    kernel_page_size,
):
    spec = MLAAttentionSpec(
        block_size=manager_block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        state_content_bytes=ROW,
        tokens_per_state=4,
        kernel_page_size=256,
    )
    group = AttentionGroup(_RecordingBackend, ["indexer"], spec, 0)
    group.create_metadata_builders(
        SimpleNamespace(),
        torch.device("cpu"),
        kernel_block_size,
        block_stride_bytes=stride_pages * spec.page_size_bytes,
    )

    builder = group.get_metadata_builder()
    assert builder.kv_cache_spec.block_size == builder_block_size
    assert builder.kv_cache_spec.kernel_page_size == kernel_page_size
    assert builder.kernel_block_size == kernel_block_size
    assert builder.block_stride_bytes == stride_pages * spec.page_size_bytes
