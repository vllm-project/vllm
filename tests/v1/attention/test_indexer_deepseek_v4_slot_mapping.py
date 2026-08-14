# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla import indexer as indexer_module
from vllm.v1.attention.backends.mla.indexer import (
    BuildPrefillChunkMetadataKernel,
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.block_table import get_block_table_width


def test_indexer_warmup_normalizes_zero_compress_ratios():
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=[0, 0, 4, 128, 0])
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )

    keys = BuildPrefillChunkMetadataKernel().get_warmup_keys(config)

    assert {key.COMPRESS_RATIO for key in keys} == {1, 4, 128}


def test_zero_token_pcp_rank_participates_in_compressed_mapping_gather(monkeypatch):
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.compress_ratio = 4
    builder.pcp_world_size = 4
    builder.kv_cache_spec = SimpleNamespace(storage_block_size=64)
    builder.compressed_slot_mapping_buffer = torch.zeros(8, dtype=torch.int64)

    def fake_get_compressed_slot_mapping(
        num_tokens,
        query_start_loc,
        seq_lens,
        block_table,
        block_size,
        compress_ratio,
        out,
    ):
        assert num_tokens == 0
        assert block_size == 64
        assert compress_ratio == 4
        out.fill_(-1)
        return out[:num_tokens]

    gathered_slot_mapping = torch.tensor([123, -1, -1, -1], dtype=torch.int64)

    class FakePCPGroup:
        def __init__(self):
            self.calls = 0

        def all_gather(self, tensor, dim=0):
            self.calls += 1
            assert dim == 0
            torch.testing.assert_close(tensor, torch.tensor([-1]))
            return gathered_slot_mapping

    fake_pcp_group = FakePCPGroup()
    monkeypatch.setattr(
        indexer_module,
        "get_compressed_slot_mapping",
        fake_get_compressed_slot_mapping,
    )
    monkeypatch.setattr(indexer_module, "get_pcp_group", lambda: fake_pcp_group)

    query_start_loc = torch.tensor([0, 0], dtype=torch.int32)
    seq_lens = torch.tensor([16], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens,
        num_reqs=1,
        num_actual_tokens=0,
        max_query_len=0,
        max_seq_len=16,
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([11, 22, 33, 44], dtype=torch.int64),
        causal=True,
    )

    metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert fake_pcp_group.calls == 1
    assert metadata.num_decodes == 0
    assert metadata.num_decode_tokens == 0
    assert metadata.num_prefills == 0
    assert metadata.num_prefill_tokens == 0
    torch.testing.assert_close(metadata.slot_mapping, gathered_slot_mapping)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_builder_deepseek_v4_compressed_slot_mapping_uses_storage_block_size():
    """Regression test: DeepseekV4 compression path must compute slot_mapping from
    compressed positions, not reuse the uncompressed common metadata mapping.
    """
    device = torch.device("cuda")

    # storage_block_size = block_size // compress_ratio = 256 // 4 = 64
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    vllm_config = create_vllm_config(max_model_len=1024)
    max_num_blocks = kv_cache_spec.max_num_blocks_per_req(vllm_config, 1024)
    block_table_width = get_block_table_width(max_num_blocks, kv_cache_spec.block_size)
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
        block_table_width=block_table_width,
    )

    # Construct a single request where:
    # - num_computed = 240 (=> compressed_pos_start = 60)
    # - query_len = 40 (=> num_groups = 10)
    # => compressed positions are 60..69 which cross the storage block boundary at 64.
    query_start_loc = torch.tensor([0, 40], dtype=torch.int32, device=device)
    query_start_loc_cpu = query_start_loc.cpu()
    seq_lens = torch.tensor([280], dtype=torch.int32, device=device)  # 240 + 40

    # Two blocks: compressed positions 0..63 map to block 5, 64..127 map to block 7.
    block_table_tensor = torch.tensor([[5, 7]], dtype=torch.int32, device=device)

    # Dummy uncompressed slot mapping (length == uncompressed num_actual_tokens).
    slot_mapping = torch.full((40,), -123, dtype=torch.int64, device=device)

    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
        num_reqs=1,
        num_actual_tokens=40,
        max_query_len=40,
        max_seq_len=280,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )

    md = builder.build(common_prefix_len=0, common_attn_metadata=common)

    # The compressed slot_mapping retains the original uncompressed size (40).
    # Only every compress_ratio-th position gets a valid slot; the rest are -1.
    assert md.slot_mapping.numel() == 40
    valid_slots = md.slot_mapping[md.slot_mapping >= 0]
    assert valid_slots.numel() == 10  # 40 tokens / compress_ratio 4

    storage_bs = kv_cache_spec.storage_block_size  # 64
    # Compressed positions 60..63 land in block 5, positions 64..69 in block 7.
    expected = torch.tensor(
        [
            5 * storage_bs + 60,
            5 * storage_bs + 61,
            5 * storage_bs + 62,
            5 * storage_bs + 63,
        ]
        + [
            7 * storage_bs + 0,
            7 * storage_bs + 1,
            7 * storage_bs + 2,
            7 * storage_bs + 3,
            7 * storage_bs + 4,
            7 * storage_bs + 5,
        ],
        dtype=torch.int64,
        device=device,
    )
    torch.testing.assert_close(valid_slots, expected)
