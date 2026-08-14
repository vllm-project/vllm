# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLASparseMetadataBuilder,
    DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla import indexer
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    DeepseekV4IndexerMetadataBuilder,
)


def test_deepseek_v4_rocm_adaptive_builders_support_varlen_full_graphs():
    adaptive_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=True)
    )
    fixed_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=False)
    )

    for builder_cls in (
        DeepseekV4ROCMAiterMLASparseMetadataBuilder,
        DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
    ):
        assert (
            builder_cls.get_cudagraph_support(adaptive_config, SimpleNamespace())
            == AttentionCGSupport.ALWAYS
        )
        assert (
            builder_cls.get_cudagraph_support(fixed_config, SimpleNamespace())
            == AttentionCGSupport.UNIFORM_BATCH
        )


def test_deepseek_v4_rocm_adaptive_indexer_support(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(indexer.current_platform, "is_rocm", lambda: True)
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=True)
    )

    assert DeepseekV4IndexerBackend.supports_device_cpu_query_lens_mismatch()
    assert (
        DeepseekV4IndexerMetadataBuilder.get_cudagraph_support(
            vllm_config, SimpleNamespace()
        )
        == AttentionCGSupport.ALWAYS
    )


def test_rocm_indexer_flattens_device_cpu_query_length_mismatch():
    builder = DeepseekV4IndexerMetadataBuilder.__new__(DeepseekV4IndexerMetadataBuilder)
    builder.supports_varlen = False
    builder.force_varlen_flattening = True
    builder.decode_seq_lens_buffer = torch.zeros(8, dtype=torch.int32)
    builder.expanded_block_table_buffer = torch.zeros((8, 2), dtype=torch.int32)
    builder.decode_lens_buffer = torch.zeros(8, dtype=torch.int32)
    builder.arange_buffer = torch.arange(8, dtype=torch.int32)

    seq_lens, block_table, decode_lens, batch_size, requires_padding = (
        builder._prepare_decode_tensors(
            seq_lens=torch.tensor([10, 10], dtype=torch.int32),
            block_table=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            decode_lens=torch.tensor([3, 1], dtype=torch.int32),
            # The CPU distributes the same four-token budget evenly.
            decode_lens_cpu=torch.tensor([2, 2], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            num_decodes=2,
            num_decode_tokens=4,
            use_native=False,
            next_n=8,
            max_decode_len=2,
        )
    )

    torch.testing.assert_close(
        seq_lens, torch.tensor([8, 9, 10, 10], dtype=torch.int32)
    )
    torch.testing.assert_close(
        block_table,
        torch.tensor([[1, 2], [1, 2], [1, 2], [3, 4]], dtype=torch.int32),
    )
    torch.testing.assert_close(decode_lens, torch.ones(4, dtype=torch.int32))
    assert batch_size == 4
    assert not requires_padding
