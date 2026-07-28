# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the DSA indexer's expanded block table."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    get_block_table_width,
)
from vllm.v1.worker.block_table import MultiGroupBlockTable


def _make_builder(block_table_width: int, max_num_batched_tokens: int = 16):
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.expanded_block_table_buffer = torch.zeros(
        (max_num_batched_tokens, block_table_width), dtype=torch.int32
    )
    builder.decode_seq_lens_buffer = torch.zeros(
        max_num_batched_tokens, dtype=torch.int32
    )
    builder.arange_buffer = torch.arange(max_num_batched_tokens, dtype=torch.int32)
    builder.decode_lens_buffer = torch.zeros(max_num_batched_tokens, dtype=torch.int32)
    return builder


def test_nonuniform_decode_uses_finalized_block_table_width():
    block_tables = MultiGroupBlockTable(
        max_num_reqs=2,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[64],
        kernel_block_sizes=[64],
        max_num_blocks=[1875],
    )
    block_table = block_tables[0].get_device_tensor(2)
    assert block_table.shape == (2, 1876)
    indexer_width = get_block_table_width(1875, 64, 64)
    assert indexer_width == block_table.shape[1]
    builder = _make_builder(indexer_width)
    block_table.copy_(torch.arange(2 * 1876, dtype=torch.int32).view(2, 1876))
    decode_lens_cpu = torch.tensor([4, 2], dtype=torch.int32)

    _, expanded_block_table, _, _, _ = builder._prepare_decode_tensors(
        seq_lens=torch.tensor([100, 100], dtype=torch.int32),
        block_table=block_table,
        decode_lens=decode_lens_cpu,
        decode_lens_cpu=decode_lens_cpu,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        num_decodes=2,
        num_decode_tokens=8,
        use_native=False,
        next_n=4,
        max_decode_len=4,
    )

    expected = torch.repeat_interleave(block_table, decode_lens_cpu, dim=0)
    torch.testing.assert_close(expanded_block_table[:6], expected)
    assert expanded_block_table.shape == (8, 1876)


def test_block_table_width_aligns_before_kernel_block_splitting():
    block_tables = MultiGroupBlockTable(
        max_num_reqs=1,
        max_num_batched_tokens=1,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[256],
        kernel_block_sizes=[64],
        max_num_blocks=[235],
    )

    expected_width = get_block_table_width(235, 256, 64)
    assert expected_width == 940
    assert block_tables[0].get_device_tensor(1).shape[1] == expected_width


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_buffer_accounts_for_dcp_and_kernel_block_splitting(monkeypatch):
    kv_cache_block_size = 256
    kernel_block_size = 64
    vllm_config = create_vllm_config(max_model_len=1200, block_size=kv_cache_block_size)
    vllm_config.parallel_config.decode_context_parallel_size = 2
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.indexer.get_dcp_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )
    kv_cache_spec = MLAAttentionSpec(
        block_size=kv_cache_block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    ).copy_with_new_block_size(kernel_block_size)

    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=torch.device("cuda"),
    )

    max_num_kv_blocks = kv_cache_spec.max_num_blocks_per_req(vllm_config, 1200)
    expected_width = get_block_table_width(
        max_num_kv_blocks, kv_cache_block_size, kernel_block_size
    )
    assert expected_width == 12
    assert builder.expanded_block_table_buffer.shape[1] == expected_width
