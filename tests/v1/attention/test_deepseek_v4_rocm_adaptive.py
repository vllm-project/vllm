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


def _make_indexer_builder(*, adaptive: bool, capacity: int = 12):
    builder = DeepseekV4IndexerMetadataBuilder.__new__(DeepseekV4IndexerMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=adaptive)
    )
    builder.supports_varlen = False
    builder.decode_seq_lens_buffer = torch.zeros(capacity, dtype=torch.int32)
    builder.expanded_block_table_buffer = torch.zeros((capacity, 2), dtype=torch.int32)
    builder.decode_lens_buffer = torch.zeros(capacity, dtype=torch.int32)
    builder.arange_buffer = torch.arange(capacity, dtype=torch.int32)
    return builder


@pytest.mark.cpu_test
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


@pytest.mark.cpu_test
def test_deepseek_v4_rocm_adaptive_indexer_support(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(indexer.current_platform, "is_rocm", lambda: True)
    vllm_config = SimpleNamespace(
        num_speculative_tokens=7,
        speculative_config=SimpleNamespace(enable_adaptive_verification=True),
    )

    assert DeepseekV4IndexerBackend.supports_device_cpu_query_lens_mismatch()
    assert DeepseekV4IndexerMetadataBuilder._use_flattening(vllm_config)
    assert (
        DeepseekV4IndexerMetadataBuilder.get_cudagraph_support(
            vllm_config, SimpleNamespace()
        )
        == AttentionCGSupport.ALWAYS
    )


@pytest.mark.cpu_test
def test_rocm_adaptive_indexer_preserves_single_request_uniform_path(
    monkeypatch: pytest.MonkeyPatch,
):
    builder = _make_indexer_builder(adaptive=True, capacity=8)

    class FakeUniformKernel:
        called = False

        def __getitem__(self, grid):
            assert grid == (4,)

            def launch(
                seq_lens,
                decode_seq_lens,
                block_table,
                block_table_stride,
                expanded_block_table,
                expanded_block_table_stride,
                decode_lens,
                max_decode_len,
                *,
                BLOCK_SIZE,
            ):
                self.called = True
                assert block_table_stride == expanded_block_table_stride == 2
                assert BLOCK_SIZE == 1024
                decode_seq_lens[:max_decode_len] = torch.arange(
                    seq_lens[0] - max_decode_len + 1,
                    seq_lens[0] + 1,
                    dtype=torch.int32,
                )
                expanded_block_table[:max_decode_len] = block_table[0]
                decode_lens[:max_decode_len] = 1

            return launch

    fake_kernel = FakeUniformKernel()
    monkeypatch.setattr(indexer, "_prepare_uniform_decode_kernel", fake_kernel)

    seq_lens, block_table, decode_lens, batch_size, requires_padding = (
        builder._prepare_decode_tensors(
            seq_lens=torch.tensor([10], dtype=torch.int32),
            block_table=torch.tensor([[1, 2]], dtype=torch.int32),
            decode_lens=torch.tensor([4], dtype=torch.int32),
            decode_lens_cpu=torch.tensor([4], dtype=torch.int32),
            query_start_loc=torch.tensor([0], dtype=torch.int32),
            num_decodes=1,
            num_decode_tokens=4,
            use_native=False,
            next_n=8,
            max_decode_len=4,
        )
    )

    assert fake_kernel.called
    torch.testing.assert_close(seq_lens, torch.tensor([7, 8, 9, 10], dtype=torch.int32))
    torch.testing.assert_close(
        block_table,
        torch.tensor([[1, 2], [1, 2], [1, 2], [1, 2]], dtype=torch.int32),
    )
    torch.testing.assert_close(decode_lens, torch.ones(4, dtype=torch.int32))
    assert batch_size == 4
    assert not requires_padding


@pytest.mark.cpu_test
def test_rocm_adaptive_indexer_replays_changed_allocations_with_stable_buffers():
    builder = _make_indexer_builder(adaptive=True)
    buffer_ptrs = (
        builder.decode_seq_lens_buffer.data_ptr(),
        builder.expanded_block_table_buffer.data_ptr(),
        builder.decode_lens_buffer.data_ptr(),
    )
    block_table = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    decode_lens_cpu = torch.tensor([4, 4, 0], dtype=torch.int32)

    first = builder._prepare_decode_tensors(
        seq_lens=torch.tensor([20, 20, 0], dtype=torch.int32),
        block_table=block_table,
        decode_lens=torch.tensor([7, 1, 0], dtype=torch.int32),
        decode_lens_cpu=decode_lens_cpu,
        query_start_loc=torch.tensor([0, 7, 8], dtype=torch.int32),
        num_decodes=3,
        num_decode_tokens=10,
        use_native=False,
        next_n=8,
        max_decode_len=4,
    )
    torch.testing.assert_close(
        first[0],
        torch.tensor([14, 15, 16, 17, 18, 19, 20, 20, 0, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        first[1][:, 0],
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 3, 0, 0], dtype=torch.int32),
    )

    second = builder._prepare_decode_tensors(
        seq_lens=torch.tensor([20, 20, 0], dtype=torch.int32),
        block_table=block_table,
        decode_lens=torch.tensor([1, 7, 0], dtype=torch.int32),
        decode_lens_cpu=decode_lens_cpu,
        query_start_loc=torch.tensor([0, 1, 8], dtype=torch.int32),
        num_decodes=3,
        num_decode_tokens=10,
        use_native=False,
        next_n=8,
        max_decode_len=4,
    )
    torch.testing.assert_close(
        second[0],
        torch.tensor([20, 14, 15, 16, 17, 18, 19, 20, 0, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        second[1][:, 0],
        torch.tensor([1, 3, 3, 3, 3, 3, 3, 3, 0, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(second[2], torch.ones(10, dtype=torch.int32))
    assert second[3:] == (10, False)
    assert buffer_ptrs == (
        builder.decode_seq_lens_buffer.data_ptr(),
        builder.expanded_block_table_buffer.data_ptr(),
        builder.decode_lens_buffer.data_ptr(),
    )
